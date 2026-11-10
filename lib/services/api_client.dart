import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:cross_file/cross_file.dart';
import 'package:http/http.dart' as http;
import 'package:image_classifier/config.dart';
import 'package:image_classifier/models/model_info.dart';
import 'package:image_classifier/models/prediction.dart';

/// Custom exception for API errors.
///
/// The status code names the failure class. When the backend answered, the
/// status code of the response is kept. When the failure happened before a
/// response arrived, a fixed code names the class instead: 408 for a request
/// that never reached an answer, 400 for a local file fault, 502 for a
/// response that could not be read or parsed, and 503 for a transport
/// failure.
///
/// Every failure is named this way. A failure that carries its own text (a
/// socket error, a client exception, a state error) never reaches the user:
/// the message is fixed and the status code names the class, so no raw error
/// text can escape the client.
class ApiException implements Exception {
  const ApiException({required this.message, required this.statusCode});

  final String message;
  final int statusCode;

  @override
  String toString() => 'ApiException($statusCode): $message';
}

/// Client for communicating with the FastAPI backend.
class ApiClient {
  /// Creates an [ApiClient] with the given [client], optional [baseUrl], and [timeout].
  ///
  /// The [client] is owned by the caller and should be disposed by the caller.
  /// If [baseUrl] is not provided, it is resolved from [Config.resolveApiBaseUrl]
  /// (compile-time override > runtime override > platform default).
  /// If [timeout] is not provided, uses [Config.requestTimeout].
  ApiClient({required this.client, String? baseUrl, Duration? timeout})
    : _explicitBaseUrl = baseUrl,
      _timeout = timeout ?? Config.requestTimeout;

  final http.Client client;
  final String? _explicitBaseUrl;
  final Duration _timeout;

  String? _resolvedBaseUrl;

  /// The effective base URL for requests.
  ///
  /// Resolved lazily and cached. When no URL was passed to the constructor,
  /// this consults [Config.resolveApiBaseUrl], which honours the runtime
  /// override stored by the settings screen.
  Future<String> get baseUrl async {
    final explicit = _explicitBaseUrl;
    if (explicit != null) return Config.normalizeApiBaseUrl(explicit);
    final cached = _resolvedBaseUrl;
    if (cached != null) return cached;
    final resolved = await Config.resolveApiBaseUrl();
    _resolvedBaseUrl = resolved;
    return resolved;
  }

  /// Discards the cached base URL so the next request re-resolves it.
  ///
  /// Call after changing the runtime API base URL so the new value takes
  /// effect without restarting the app.
  void invalidateBaseUrlCache() {
    _resolvedBaseUrl = null;
  }

  /// Checks if the backend is healthy.
  Future<bool> healthCheck() async {
    final statusCode = await healthStatusCode();
    return statusCode == 200;
  }

  /// Returns the HTTP status code from `/health`, or null when the backend
  /// cannot be reached.
  ///
  /// Unlike [healthCheck], this lets callers distinguish a reachable but
  /// unhealthy backend (non-200) from an unreachable one (null).
  Future<int?> healthStatusCode() async {
    final url = await baseUrl;
    try {
      final response = await client
          .get(Uri.parse('$url/health'))
          .timeout(_timeout);
      return response.statusCode;
    } on TimeoutException {
      return null;
    } on SocketException {
      return null;
    } on http.ClientException {
      return null;
    } on HttpException {
      return null;
    } on FormatException {
      return null;
    } on ArgumentError {
      return null;
    } on IOException {
      return null;
    } on Exception {
      // The client failed in a way this list does not name. The health check
      // only answers with a status code, so an unnamed failure means the
      // backend cannot be reached.
      return null;
    } on Error {
      return null;
    }
  }

  /// Gets model metadata from the backend.
  Future<ModelInfo> getModelInfo() async {
    final url = await baseUrl;
    final http.Response response = await _handleRequest(url, () async {
      // The body is read through the bounded reader, so a model info response
      // that is cut short or never ends is reported the same way a prediction
      // response is: the backend answered, so the transfer failed after that.
      final stopwatch = Stopwatch()..start();
      final streamedResponse = await client
          .send(http.Request('GET', Uri.parse('$url/model-info')))
          .timeout(_timeout);
      // The reader reports every failure of the body as the same response
      // failure, so no transport error text can reach the user.
      return _readResponseBody(
        streamedResponse,
        _remainingBudget(stopwatch),
        url,
      );
    });

    // The body is read outside the transport wrapper: the backend answered,
    // so a body that cannot be read into the expected shape is a response
    // problem and never a transport failure.
    if (response.statusCode == 200) {
      return _parseResponse(response.body, url, ModelInfo.fromJson);
    }
    throw _toApiException(response);
  }

  /// Classifies an image and returns ranked predictions.
  Future<PredictionResponse> predict(File imageFile) async {
    final url = await baseUrl;
    // The file size is resolved before the request is sent, so a missing
    // image file is reported as a file problem instead of being mistaken for
    // a connection failure. A file that exists but cannot be read fails while
    // the request body is streamed; that failure is reported as a file
    // problem too, so the backend is never blamed for it.
    final http.MultipartFile upload;
    try {
      upload = await http.MultipartFile.fromPath('file', imageFile.path);
    } on IOException {
      // The file cannot be read. The message names the problem only: the
      // local path stays out of the text that reaches the screen. The
      // request was never sent, so the status names the local input fault
      // (400) instead of a status of the backend.
      throw ApiException(
        message: 'Could not read the image file.',
        statusCode: 400,
      );
    }
    final http.Response response = await _handleRequest(url, () async {
      final request = http.MultipartRequest('POST', Uri.parse('$url/predict'))
        ..files.add(upload);

      final stopwatch = Stopwatch()..start();
      final http.StreamedResponse streamedResponse;
      try {
        streamedResponse = await client.send(request).timeout(_timeout);
      } on FileSystemException {
        // The file passed the size check but could not be read while the
        // body was streamed. That is a problem with the local file, not with
        // the connection to the backend. The path stays out of the message.
        // The request was never delivered, so the status names the local
        // input fault (400) instead of a status of the backend.
        throw ApiException(
          message: 'Could not read the image file.',
          statusCode: 400,
        );
      }
      // The body read is bounded too: a backend that sends the status code
      // and then stalls would otherwise keep the request pending forever.
      // The reader reports every failure of the body as the same response
      // failure, so no transport error text can reach the user.
      return _readResponseBody(
        streamedResponse,
        _remainingBudget(stopwatch),
        url,
      );
    });

    // The body is read outside the transport wrapper: the backend answered,
    // so a body that cannot be read into the expected shape is a response
    // problem and never a transport failure.
    if (response.statusCode == 200) {
      return _parseResponse(response.body, url, PredictionResponse.fromJson);
    }
    throw _toApiException(response);
  }

  /// Predicts using an [XFile] (from image_picker/file_picker).
  ///
  /// Uses [XFile.path] for file upload. On platforms where path is unavailable
  /// (e.g., web), the caller should use [predictBytes] instead.
  Future<PredictionResponse> predictXFile(XFile xFile) async {
    return predict(File(xFile.path));
  }

  /// Predicts using raw image bytes (for web platforms where XFile.path is unavailable).
  ///
  /// Accepts [bytes] (Uint8List) and a [fileName] for the multipart upload.
  /// This is the recommended method for web platforms where [XFile.path] returns
  /// null or throws. On mobile/desktop, prefer [predictXFile] or [predict].
  Future<PredictionResponse> predictBytes(
    Uint8List bytes,
    String fileName,
  ) async {
    final url = await baseUrl;
    final http.Response response = await _handleRequest(url, () async {
      final request = http.MultipartRequest('POST', Uri.parse('$url/predict'))
        ..files.add(
          http.MultipartFile.fromBytes('file', bytes, filename: fileName),
        );

      final stopwatch = Stopwatch()..start();
      final http.StreamedResponse streamedResponse;
      try {
        streamedResponse = await client.send(request).timeout(_timeout);
      } on FileSystemException {
        // A custom client can read the request body from the file system.
        // The failure is about the local file, not about the connection to
        // the backend, so the backend is not blamed for it. The request was
        // never delivered, so the status names the local input fault (400)
        // instead of a status of the backend.
        throw ApiException(
          message: 'Could not read the image file.',
          statusCode: 400,
        );
      }
      // The body read is bounded too: a backend that sends the status code
      // and then stalls would otherwise keep the request pending forever.
      // The reader reports every failure of the body as the same response
      // failure, so no transport error text can reach the user.
      return _readResponseBody(
        streamedResponse,
        _remainingBudget(stopwatch),
        url,
      );
    });

    // The body is read outside the transport wrapper: the backend answered,
    // so a body that cannot be read into the expected shape is a response
    // problem and never a transport failure.
    if (response.statusCode == 200) {
      return _parseResponse(response.body, url, PredictionResponse.fromJson);
    }
    throw _toApiException(response);
  }

  /// Releases the resources held by this client.
  ///
  /// This client never owns the underlying [http.Client]: it is created by the
  /// caller and passed in through the constructor, so closing it here would
  /// break other users of the same client. The method therefore does nothing
  /// and the caller stays responsible for disposing the client.
  Future<void> dispose() async {
    // No-op: the client lifecycle belongs to the caller.
  }

  /// Runs [request], translating transport failures into [ApiException].
  ///
  /// [url] is the base URL the request targets; it is included in the error
  /// message so a connection failure points at the address that failed
  /// instead of reporting a bare "Network error".
  ///
  /// The closure only sends the request and reads the body. A response that
  /// arrives but cannot be read into the expected shape is a response
  /// problem, so the body is parsed outside of this method and can never be
  /// reported as a transport failure.
  Future<http.Response> _handleRequest(
    String url,
    Future<http.Response> Function() request,
  ) async {
    try {
      return await request();
    } on ApiException {
      // The request already named the failure class (a local file problem or
      // a response that could not be read). The catch all below must not
      // rename it as a transport failure.
      rethrow;
    } on TimeoutException {
      throw ApiException(
        message:
            'Request to $url timed out after ${_timeout.inSeconds}s. '
            'Is the backend running at $url?',
        statusCode: 408,
      );
    } on SocketException {
      throw _transportFailure(url);
    } on http.ClientException {
      throw _transportFailure(url);
    } on HttpException {
      throw _transportFailure(url);
    } on IOException {
      throw _transportFailure(url);
    } on FormatException {
      throw ApiException(
        message:
            'Invalid response from $url. '
            'The response could not be parsed as the expected JSON.',
        statusCode: 502,
      );
    } on TypeError {
      throw ApiException(
        message:
            'Invalid response from $url: unexpected response shape. '
            'Is the backend the image classifier API?',
        statusCode: 502,
      );
    } on ArgumentError {
      // The address itself is validated before the request, so an
      // ArgumentError here comes from the request, not from the URL. The raw
      // exception text is dropped: a custom client can carry any text in it.
      throw ApiException(message: 'Invalid request to $url.', statusCode: 502);
    } on Exception {
      // The client failed while the request was sent or the body was read,
      // in a way this list does not name. The request never reached a
      // complete answer, so it is a transport failure. The raw error text is
      // dropped: a custom client can carry any text in it.
      throw _transportFailure(url);
    } on Error {
      // The same holds for an error that is not an exception.
      throw _transportFailure(url);
    }
  }

  /// The time that is left of the request budget.
  ///
  /// The send and the body read share one budget, so a request never takes
  /// longer than the configured timeout.
  Duration _remainingBudget(Stopwatch stopwatch) {
    final remaining = _timeout - stopwatch.elapsed;
    return remaining.isNegative ? Duration.zero : remaining;
  }

  /// Reads the body of [streamedResponse] within [budget].
  ///
  /// The read is bounded and the subscription is released when the deadline
  /// expires. A future timeout alone is not enough: it gives up on the
  /// response but leaves the body stream subscribed, so the transfer would
  /// keep running and the connection would stay open after the request was
  /// abandoned.
  ///
  /// Every failure of the read is reported as the same response failure: the
  /// status line arrived, so the backend answered and the transfer failed
  /// after that. The error that ended the read never leaves this method.
  Future<http.Response> _readResponseBody(
    http.StreamedResponse streamedResponse,
    Duration budget,
    String url,
  ) async {
    if (budget <= Duration.zero) {
      // The budget is spent, so the body cannot be read any more.
      throw _responseReadFailure(url);
    }
    final completer = Completer<http.Response>();
    final buffer = BytesBuilder(copy: false);
    final timer = Timer(budget, () {
      if (!completer.isCompleted) {
        completer.completeError(_responseReadFailure(url));
      }
    });
    final subscription = streamedResponse.stream.listen(
      buffer.add,
      onError: (Object error, StackTrace stackTrace) {
        if (!completer.isCompleted) {
          // The transport failed after the status line arrived. The error is
          // dropped: it can carry any text, and the status code names the
          // failure class.
          completer.completeError(_responseReadFailure(url), stackTrace);
        }
      },
      onDone: () {
        if (!completer.isCompleted) {
          completer.complete(
            http.Response.bytes(
              buffer.takeBytes(),
              streamedResponse.statusCode,
              headers: streamedResponse.headers,
              request: streamedResponse.request,
              reasonPhrase: streamedResponse.reasonPhrase,
              isRedirect: streamedResponse.isRedirect,
              persistentConnection: streamedResponse.persistentConnection,
            ),
          );
        }
      },
      cancelOnError: true,
    );
    try {
      return await completer.future;
    } finally {
      timer.cancel();
      try {
        await subscription.cancel();
      } catch (_) {
        // A failure to release the subscription must not replace the failure
        // that ended the read: the user has to see why the response could
        // not be read.
      }
    }
  }

  /// The failure of a response body that could not be read.
  ///
  /// The status line arrived, so the backend answered and the connection was
  /// established. The transfer failed after that, which is not a connection
  /// problem, so the backend must not be blamed for it.
  ApiException _responseReadFailure(String url) {
    return ApiException(
      message: 'Could not read the response from $url.',
      statusCode: 502,
    );
  }

  /// The failure of a request that never reached an answer.
  ///
  /// Every transport failure - a refused socket, a broken connection, an
  /// HTTP error and an I/O error - carries the same status code and the same
  /// message, so one problem can never be reported under two different
  /// codes. The raw exception text stays out of the message: it can name
  /// internal details of the TLS stack or of the file system.
  ApiException _transportFailure(String url) {
    return ApiException(
      message:
          'Cannot reach $url. Check that the backend is running and that '
          'the address is reachable from this device.',
      statusCode: 503,
    );
  }

  ApiException _toApiException(http.Response response) {
    return ApiException(
      message: _parseErrorMessage(response.body, response.statusCode),
      statusCode: response.statusCode,
    );
  }

  /// Reads the body of a successful response into the expected shape.
  ///
  /// The backend answered, so a body that cannot be read is a response
  /// problem and not a transport failure. Every failure is reported with a
  /// fixed message, so no raw error text - an assertion message or a cast
  /// error - can reach the user.
  T _parseResponse<T>(
    String body,
    String url,
    T Function(Map<String, dynamic>) fromJson,
  ) {
    final Map<String, dynamic> json;
    try {
      json = _parseJsonMap(body);
    } on FormatException {
      throw ApiException(
        message:
            'Invalid response from $url. '
            'The response could not be parsed as the expected JSON.',
        statusCode: 502,
      );
    }
    try {
      return fromJson(json);
    } on TypeError {
      throw ApiException(
        message:
            'Invalid response from $url: unexpected response shape. '
            'Is the backend the image classifier API?',
        statusCode: 502,
      );
    } on Exception {
      // The body has the JSON shape the model expects, but the model cannot
      // read it - for example because a value is outside its range. The
      // failure is still a response problem.
      throw ApiException(
        message:
            'Invalid response from $url: unexpected response shape. '
            'Is the backend the image classifier API?',
        statusCode: 502,
      );
    } on Error {
      // The same holds for an error that is not an exception, such as the
      // assertion that guards the confidence range.
      throw ApiException(
        message:
            'Invalid response from $url: unexpected response shape. '
            'Is the backend the image classifier API?',
        statusCode: 502,
      );
    }
  }

  dynamic _parseJson(String body) {
    try {
      if (body.isEmpty) return <String, dynamic>{};
      return jsonDecode(body);
    } catch (_) {
      throw const FormatException('Invalid JSON response');
    }
  }

  Map<String, dynamic> _parseJsonMap(String body) {
    try {
      final parsed = _parseJson(body);
      if (parsed is Map<String, dynamic>) return parsed;
      throw const FormatException('Response is not a JSON object');
    } catch (_) {
      throw const FormatException('Invalid JSON response');
    }
  }

  /// Builds the message of a failed request.
  ///
  /// The `detail` field of a JSON error body is written for the user, so it
  /// is shown. Every other body is dropped: the raw response body can carry
  /// anything the server sent, including internal details and payload.
  String _parseErrorMessage(String body, int statusCode) {
    try {
      final json = _parseJson(body);
      if (json is Map<String, dynamic> && json['detail'] != null) {
        final detail = json['detail'];
        if (detail is String) {
          return _sanitizeErrorMessage(detail);
        }
      }
    } catch (_) {
      // Fall through to default message
    }
    return _sanitizeErrorMessage('Request failed with status $statusCode');
  }

  /// Sanitizes error messages to prevent leaking sensitive data.
  /// Limits length and strips control characters.
  String _sanitizeErrorMessage(String message) {
    const maxLength = 500;
    // Strip control characters except newlines/tabs
    final sanitized = message.replaceAll(
      RegExp(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]'),
      '',
    );
    if (sanitized.length > maxLength) {
      return '${sanitized.substring(0, maxLength)}... [truncated]';
    }
    return sanitized;
  }
}
