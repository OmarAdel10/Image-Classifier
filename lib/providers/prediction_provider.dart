import 'package:cross_file/cross_file.dart';
import 'package:flutter/foundation.dart';
import 'package:image_classifier/models/prediction.dart';
import 'package:image_classifier/services/api_client.dart';

/// Represents the current state of a prediction operation.
sealed class PredictionState {
  const PredictionState();
}

/// No prediction in progress.
class PredictionStateIdle extends PredictionState {
  const PredictionStateIdle();
}

/// Prediction is in progress.
class PredictionStateLoading extends PredictionState {
  const PredictionStateLoading();
}

/// Prediction completed successfully with results.
class PredictionStateResult extends PredictionState {
  const PredictionStateResult({required this.response});

  final PredictionResponse response;
}

/// Prediction failed with an error message.
class PredictionStateError extends PredictionState {
  const PredictionStateError({required this.message});

  final String message;
}

/// Manages prediction state and coordinates with the API client.
class PredictionProvider extends ChangeNotifier {
  PredictionProvider({required this.apiClient});

  final ApiClient apiClient;

  PredictionState _state = const PredictionStateIdle();

  PredictionState get state => _state;

  /// Predicts the class of the given image file.
  ///
  /// Updates state through: idle -> loading -> result|error
  Future<void> predict(XFile file) async {
    // If already loading, we still start a new prediction
    // The previous one will be abandoned when state updates
    _state = const PredictionStateLoading();
    notifyListeners();

    try {
      final response = await apiClient.predictXFile(file);
      _state = PredictionStateResult(response: response);
    } on ApiException catch (e) {
      _state = PredictionStateError(message: e.message);
    } on Exception catch (e) {
      _state = PredictionStateError(message: e.toString());
    }
    notifyListeners();
  }

  /// Clears the current prediction state back to idle.
  void clear() {
    _state = const PredictionStateIdle();
    notifyListeners();
  }
}
