import 'dart:io';
import 'dart:convert';

import 'package:cross_file/cross_file.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image_classifier/models/prediction.dart';
import 'package:image_classifier/models/model_info.dart';
import 'package:image_classifier/services/api_client.dart';

File _createTempImageFile() {
  final tempDir = Directory.systemTemp.createTempSync('api_client_test_');
  final file = File('${tempDir.path}/test_image.png');
  // Create a minimal valid PNG (1x1 pixel)
  file.writeAsBytesSync([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
    0x00, 0x00, 0x00, 0x0D, // IHDR chunk length
    0x49, 0x48, 0x44, 0x52, // IHDR
    0x00, 0x00, 0x00, 0x01, // width: 1
    0x00, 0x00, 0x00, 0x01, // height: 1
    0x08,
    0x02,
    0x00,
    0x00,
    0x00, // bit depth, color type, compression, filter, interlace
    0x90, 0x77, 0x53, 0xDE, // CRC
    0x00, 0x00, 0x00, 0x0C, // IDAT chunk length
    0x49, 0x44, 0x41, 0x54, // IDAT
    0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x03, 0x01, 0x01, 0x00,
    0x18, 0xDD, 0x8D, 0xB4, // compressed data + CRC
    0x00, 0x00, 0x00, 0x00, // IEND chunk length
    0x49, 0x45, 0x4E, 0x44, // IEND
    0xAE, 0x42, 0x60, 0x82, // CRC
  ]);
  return file;
}

void main() {
  group('ApiClient', () {
    late MockClient mockClient;
    late ApiClient apiClient;

    setUp(() {
      mockClient = MockClient((request) async {
        return http.Response('{"error": "not implemented"}', 501);
      });
      apiClient = ApiClient(client: mockClient);
    });

    test('healthCheck returns true on 200', () async {
      mockClient = MockClient((request) async {
        expect(request.url.toString(), endsWith('/health'));
        expect(request.method, 'GET');
        return http.Response('{"status": "ok"}', 200);
      });
      apiClient = ApiClient(client: mockClient);

      final result = await apiClient.healthCheck();
      expect(result, isTrue);
    });

    test('healthCheck returns false on non-200', () async {
      mockClient = MockClient((request) async {
        return http.Response('{"status": "degraded"}', 503);
      });
      apiClient = ApiClient(client: mockClient);

      final result = await apiClient.healthCheck();
      expect(result, isFalse);
    });

    test('healthCheck returns false on exception', () async {
      mockClient = MockClient((request) async {
        throw SocketException('Connection refused');
      });
      apiClient = ApiClient(client: mockClient);

      final result = await apiClient.healthCheck();
      expect(result, isFalse);
    });

    test('getModelInfo returns parsed ModelInfo on 200', () async {
      mockClient = MockClient((request) async {
        expect(request.url.toString(), endsWith('/model-info'));
        expect(request.method, 'GET');
        return http.Response(
          jsonEncode({
            'model': 'cnn_scratch',
            'input_size': '32x32',
            'classes': [
              'airplane',
              'automobile',
              'bird',
              'cat',
              'deer',
              'dog',
              'frog',
              'horse',
              'ship',
              'truck',
            ],
            'device': 'cpu',
            'trained_at': '2026-10-29T10:00:00.000Z',
          }),
          200,
        );
      });
      apiClient = ApiClient(client: mockClient);

      final result = await apiClient.getModelInfo();

      expect(result, isA<ModelInfo>());
      expect(result.modelName, 'cnn_scratch');
      expect(result.inputSize, '32x32');
      expect(result.classes, hasLength(10));
      expect(result.classes.first, 'airplane');
      expect(result.device, 'cpu');
      expect(result.trainedAt, isNotNull);
      expect(result.trainedAt!.year, 2026);
    });

    test('getModelInfo throws ApiException on non-200', () async {
      mockClient = MockClient((request) async {
        return http.Response('{"detail": "model unavailable"}', 503);
      });
      apiClient = ApiClient(client: mockClient);

      expect(() => apiClient.getModelInfo(), throwsA(isA<ApiException>()));
    });

    test('predict returns parsed PredictionResponse on 200', () async {
      final xFile = XFile(_createTempImageFile().path);
      mockClient = MockClient((request) async {
        expect(request.url.toString(), endsWith('/predict'));
        expect(request.method, 'POST');
        expect(
          request.headers['content-type'],
          contains('multipart/form-data'),
        );
        return http.Response(
          jsonEncode({
            'predictions': [
              {'class_name': 'cat', 'confidence': 0.9},
              {'class_name': 'dog', 'confidence': 0.1},
            ],
            'latency_ms': 42.5,
            'model': 'cnn_scratch',
          }),
          200,
        );
      });
      apiClient = ApiClient(client: mockClient);

      final result = await apiClient.predictXFile(xFile);

      expect(result, isA<PredictionResponse>());
      expect(result.predictions, hasLength(2));
      expect(result.predictions.first.className, 'cat');
      expect(result.predictions.first.confidence, 0.9);
      expect(result.predictions.last.className, 'dog');
      expect(result.predictions.last.confidence, 0.1);
      expect(result.latencyMs, 42.5);
      expect(result.model, 'cnn_scratch');
    });

    test('predict throws ApiException on 400', () async {
      final xFile = XFile(_createTempImageFile().path);
      mockClient = MockClient((request) async {
        return http.Response('{"detail": "file must be an image"}', 400);
      });
      apiClient = ApiClient(client: mockClient);

      expect(() => apiClient.predictXFile(xFile), throwsA(isA<ApiException>()));
    });

    test('predict throws ApiException on 413', () async {
      final xFile = XFile(_createTempImageFile().path);
      mockClient = MockClient((request) async {
        return http.Response('{"detail": "image file too large"}', 413);
      });
      apiClient = ApiClient(client: mockClient);

      expect(() => apiClient.predictXFile(xFile), throwsA(isA<ApiException>()));
    });

    test('predict throws ApiException on 500', () async {
      final xFile = XFile(_createTempImageFile().path);
      mockClient = MockClient((request) async {
        return http.Response('{"detail": "model inference failed"}', 500);
      });
      apiClient = ApiClient(client: mockClient);

      expect(() => apiClient.predictXFile(xFile), throwsA(isA<ApiException>()));
    });

    test('predict throws ApiException on timeout', () async {
      final xFile = XFile(_createTempImageFile().path);
      mockClient = MockClient((request) async {
        await Future.delayed(const Duration(seconds: 35));
        return http.Response('{}', 200);
      });
      apiClient = ApiClient(client: mockClient);

      expect(() => apiClient.predictXFile(xFile), throwsA(isA<ApiException>()));
    });

    test('predict throws ApiException on network error', () async {
      final xFile = XFile(_createTempImageFile().path);
      mockClient = MockClient((request) async {
        throw SocketException('Connection refused');
      });
      apiClient = ApiClient(client: mockClient);

      expect(() => apiClient.predictXFile(xFile), throwsA(isA<ApiException>()));
    });
  });

  group('PredictionItem', () {
    test('fromJson creates correct object', () {
      final json = {'class_name': 'cat', 'confidence': 0.95};
      final item = PredictionItem.fromJson(json);

      expect(item.className, 'cat');
      expect(item.confidence, 0.95);
    });

    test('toJson returns correct map', () {
      const item = PredictionItem(className: 'dog', confidence: 0.8);
      final json = item.toJson();

      expect(json, {'class_name': 'dog', 'confidence': 0.8});
    });
  });

  group('PredictionResponse', () {
    test('fromJson creates correct object with multiple predictions', () {
      final json = {
        'predictions': [
          {'class_name': 'cat', 'confidence': 0.9},
          {'class_name': 'dog', 'confidence': 0.1},
        ],
        'latency_ms': 42.0,
        'model': 'cnn_scratch',
      };
      final response = PredictionResponse.fromJson(json);

      expect(response.predictions, hasLength(2));
      expect(response.latencyMs, 42.0);
      expect(response.model, 'cnn_scratch');
    });

    test('toJson returns correct map', () {
      const response = PredictionResponse(
        predictions: [
          PredictionItem(className: 'cat', confidence: 0.9),
          PredictionItem(className: 'dog', confidence: 0.1),
        ],
        latencyMs: 42.0,
        model: 'cnn_scratch',
      );
      final json = response.toJson();

      expect(json['latency_ms'], 42.0);
      expect(json['model'], 'cnn_scratch');
      expect(json['predictions'], isA<List>());
    });
  });

  group('ModelInfo', () {
    test('fromJson creates correct object with trained_at', () {
      final json = {
        'model': 'cnn_scratch',
        'input_size': '32x32',
        'classes': ['cat', 'dog'],
        'device': 'cpu',
        'trained_at': '2026-10-29T10:00:00.000Z',
      };
      final info = ModelInfo.fromJson(json);

      expect(info.modelName, 'cnn_scratch');
      expect(info.inputSize, '32x32');
      expect(info.classes, ['cat', 'dog']);
      expect(info.device, 'cpu');
      expect(info.trainedAt, isNotNull);
    });

    test('fromJson handles null trained_at', () {
      final json = {
        'model': 'cnn_scratch',
        'input_size': '32x32',
        'classes': ['cat', 'dog'],
        'device': 'cpu',
        'trained_at': null,
      };
      final info = ModelInfo.fromJson(json);

      expect(info.trainedAt, isNull);
    });

    test('toJson returns correct map', () {
      final info = ModelInfo(
        modelName: 'cnn_scratch',
        inputSize: '32x32',
        classes: ['cat', 'dog'],
        device: 'cpu',
        trainedAt: DateTime.parse('2026-10-29T10:00:00.000Z'),
      );
      final json = info.toJson();

      expect(json['model'], 'cnn_scratch');
      expect(json['input_size'], '32x32');
      expect(json['trained_at'], '2026-10-29T10:00:00.000Z');
    });
  });

  group('ApiException', () {
    test('includes status code and message', () {
      const exception = ApiException(
        message: 'model unavailable',
        statusCode: 503,
      );

      expect(exception.message, 'model unavailable');
      expect(exception.statusCode, 503);
      expect(exception.toString(), contains('503'));
      expect(exception.toString(), contains('model unavailable'));
    });
  });
}
