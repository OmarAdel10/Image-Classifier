import 'package:cross_file/cross_file.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mockito/annotations.dart';
import 'package:mockito/mockito.dart';
import 'package:image_classifier/models/prediction.dart';
import 'package:image_classifier/services/api_client.dart';
import 'package:image_classifier/providers/prediction_provider.dart';

@GenerateMocks([ApiClient])
import 'prediction_provider_test.mocks.dart';

void main() {
  group('PredictionProvider', () {
    late MockApiClient mockApiClient;
    late PredictionProvider provider;

    setUp(() {
      mockApiClient = MockApiClient();
      provider = PredictionProvider(apiClient: mockApiClient);
    });

    test('initial state is idle', () {
      expect(provider.state, isA<PredictionStateIdle>());
    });

    test('predict sets state to loading then result on success', () async {
      final file = XFile('test.png');
      final mockResponse = PredictionResponse(
        predictions: [
          PredictionItem(className: 'cat', confidence: 0.9),
          PredictionItem(className: 'dog', confidence: 0.1),
        ],
        latencyMs: 42.0,
        model: 'cnn_scratch',
      );

      when(mockApiClient.predictXFile(any))
          .thenAnswer((_) async => mockResponse);

      final future = provider.predict(file);

      // Should immediately transition to loading
      expect(provider.state, isA<PredictionStateLoading>());

      await future;

      // Should transition to result with correct data
      expect(provider.state, isA<PredictionStateResult>());
      final resultState = provider.state as PredictionStateResult;
      expect(resultState.response, mockResponse);
    });

    test('predict sets state to error on ApiException', () async {
      final file = XFile('test.png');
      const exception = ApiException(
        message: 'model unavailable',
        statusCode: 503,
      );

      when(mockApiClient.predictXFile(any))
          .thenAnswer((_) => Future.error(exception));

      final future = provider.predict(file);

      expect(provider.state, isA<PredictionStateLoading>());

      await future;

      expect(provider.state, isA<PredictionStateError>());
      final errorState = provider.state as PredictionStateError;
      expect(errorState.message, 'model unavailable');
    });

    test('predict sets state to error on unexpected exception', () async {
      final file = XFile('test.png');

      when(mockApiClient.predictXFile(any))
          .thenAnswer((_) => Future.error(Exception('network error')));

      final future = provider.predict(file);

      await future;

      expect(provider.state, isA<PredictionStateError>());
      final errorState = provider.state as PredictionStateError;
      expect(errorState.message, contains('network error'));
    });

    test('clear resets state to idle', () async {
      final file = XFile('test.png');
      final mockResponse = PredictionResponse(
        predictions: [PredictionItem(className: 'cat', confidence: 0.9)],
        latencyMs: 42.0,
        model: 'cnn_scratch',
      );

      when(mockApiClient.predictXFile(any))
          .thenAnswer((_) async => mockResponse);

      await provider.predict(file);
      expect(provider.state, isA<PredictionStateResult>());

      provider.clear();
      expect(provider.state, isA<PredictionStateIdle>());
    });

    test('predict while loading cancels previous and starts new', () async {
      final file1 = XFile('test1.png');
      final file2 = XFile('test2.png');
      final mockResponse = PredictionResponse(
        predictions: [PredictionItem(className: 'cat', confidence: 0.9)],
        latencyMs: 42.0,
        model: 'cnn_scratch',
      );

      when(mockApiClient.predictXFile(any))
          .thenAnswer((_) async => mockResponse);

      final future1 = provider.predict(file1);
      expect(provider.state, isA<PredictionStateLoading>());

      // Start second prediction while first is loading
      final future2 = provider.predict(file2);
      expect(provider.state, isA<PredictionStateLoading>());

      await Future.wait([future1, future2]);

      // Should end up with result from last call
      expect(provider.state, isA<PredictionStateResult>());
    });
  });
}
