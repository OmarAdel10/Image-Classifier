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
  /// If [baseUrl] is not provided, uses [Config.apiBaseUrl].
  /// If [timeout] is not provided, uses [Config.requestTimeout].
  ApiClient({required this.client, String? baseUrl, Duration? timeout})
    : _baseUrl = baseUrl ?? Config.apiBaseUrl,
      _timeout = timeout ?? Config.requestTimeout;

  final http.Client client;
  final String _baseUrl;
  final Duration _timeout;

  /// Checks if the backend is healthy.
  Future<bool> healthCheck() async {
    try {
      final response = await client
          .get(Uri.parse('$_baseUrl/health'))
          .timeout(_timeout);
      return response.statusCode == 200;
    } on TimeoutException catch (_) {
      return false;
    } on SocketException catch (_) {
      return false;
    } on HttpException catch (_) {
      return false;
    } on FormatException catch (_) {
      return false;
    }
  }

  /// Gets model metadata from the backend.
  Future<ModelInfo> getModelInfo() async {
    return _handleRequest(() async {
      final response = await client
          .get(Uri.parse('$_baseUrl/model-info'))
          .timeout(_timeout);

      if (response.statusCode == 200) {
        return ModelInfo.fromJson(_parseJsonMap(response.body));
      }
      throw _toApiException(response);
    });
  }

  /// Classifies an image and returns ranked predictions.
  Future<PredictionResponse> predict(File imageFile) async {
    return _handleRequest(() async {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$_baseUrl/predict'),
      )..files.add(await http.MultipartFile.fromPath('file', imageFile.path));

      final streamedResponse = await client.send(request).timeout(_timeout);
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        return PredictionResponse.fromJson(_parseJsonMap(response.body));
      }
      throw _toApiException(response);
    });
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
    return _handleRequest(() async {
      final request =
          http.MultipartRequest('POST', Uri.parse('$_baseUrl/predict'))
            ..files.add(
              http.MultipartFile.fromBytes('file', bytes, filename: fileName),
            );

      final streamedResponse = await client.send(request).timeout(_timeout);
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        return PredictionResponse.fromJson(_parseJsonMap(response.body));
      }
      throw _toApiException(response);
    });
  }

  /// Closes the underlying [http.Client] if it was created by this instance.
  ///
  /// Call this when the [ApiClient] is no longer needed to free resources.
  /// Note: If the [client] was provided by the caller, the caller is responsible
  /// for disposing it. This method only disposes if the client was created
  /// internally (which this class does not do by default).
  Future<void> dispose() async {
    // No-op by default — caller owns the client lifecycle.
    // Override in a subclass if internal client management is added.
  }

  Future<T> _handleRequest<T>(Future<T> Function() request) async {
    try {
      return await request();
    } on TimeoutException catch (_) {
      throw const ApiException(message: 'Request timed out', statusCode: 408);
    } on SocketException catch (_) {
      throw const ApiException(message: 'Network error', statusCode: 503);
    } on HttpException catch (_) {
      throw const ApiException(message: 'HTTP error', statusCode: 502);
    }
  }

  ApiException _toApiException(http.Response response) {
    return ApiException(
      message: _parseErrorMessage(response.body),
      statusCode: response.statusCode,
    );
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

  String _parseErrorMessage(String body) {
    try {
      final json = _parseJson(body);
      if (json is Map<String, dynamic> && json['detail'] != null) {
        final detail = json['detail'];
        if (detail is String) {
          // Sanitize: limit length and strip control characters (IC-57)
          return _sanitizeErrorMessage(detail);
        }
      }
    } catch (_) {
      // Fall through to default message
    }
    // Sanitize fallback message to avoid leaking sensitive data (IC-57)
    return _sanitizeErrorMessage('Request failed with status $body');
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
