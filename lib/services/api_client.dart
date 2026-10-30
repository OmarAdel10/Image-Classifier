import 'dart:async';
import 'dart:convert';
import 'dart:io';

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
    try {
      final response = await client
          .get(Uri.parse('$_baseUrl/model-info'))
          .timeout(_timeout);

      if (response.statusCode == 200) {
        return ModelInfo.fromJson(_parseJsonMap(response.body));
      }
      throw ApiException(
        message: _parseErrorMessage(response.body),
        statusCode: response.statusCode,
      );
    } on TimeoutException catch (_) {
      throw const ApiException(message: 'Request timed out', statusCode: 408);
    } on SocketException catch (_) {
      throw const ApiException(message: 'Network error', statusCode: 503);
    } on HttpException catch (_) {
      throw const ApiException(message: 'HTTP error', statusCode: 502);
    }
  }

  /// Classifies an image and returns ranked predictions.
  Future<PredictionResponse> predict(File imageFile) async {
    final request = http.MultipartRequest(
      'POST',
      Uri.parse('$_baseUrl/predict'),
    )..files.add(await http.MultipartFile.fromPath('file', imageFile.path));

    try {
      final streamedResponse = await client.send(request).timeout(_timeout);
      final response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        return PredictionResponse.fromJson(_parseJsonMap(response.body));
      }
      throw ApiException(
        message: _parseErrorMessage(response.body),
        statusCode: response.statusCode,
      );
    } on TimeoutException catch (_) {
      throw const ApiException(message: 'Request timed out', statusCode: 408);
    } on SocketException catch (_) {
      throw const ApiException(message: 'Network error', statusCode: 503);
    } on HttpException catch (_) {
      throw const ApiException(message: 'HTTP error', statusCode: 502);
    }
  }

  /// Predicts using an [XFile] (from image_picker/file_picker).
  Future<PredictionResponse> predictXFile(XFile xFile) async {
    return predict(File(xFile.path));
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
        return json['detail'] as String;
      }
    } catch (_) {
      // Fall through to default message
    }
    return 'Request failed with status $body';
  }
}
