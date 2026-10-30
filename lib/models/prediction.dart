/// Represents a single class prediction with its confidence score.
class PredictionItem {
  const PredictionItem({required this.className, required this.confidence});

  factory PredictionItem.fromJson(Map<String, dynamic> json) {
    final confidence = (json['confidence'] as num).toDouble();
    return PredictionItem(
      className: json['class_name'] as String,
      confidence: confidence.clamp(0.0, 1.0),
    );
  }

  final String className;
  final double confidence;

  Map<String, dynamic> toJson() {
    return {'class_name': className, 'confidence': confidence};
  }

  @override
  String toString() =>
      'PredictionItem(className: $className, confidence: $confidence)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is PredictionItem &&
          runtimeType == other.runtimeType &&
          className == other.className &&
          confidence == other.confidence;

  @override
  int get hashCode => className.hashCode ^ confidence.hashCode;
}

/// Represents the full ranked prediction response for one image.
class PredictionResponse {
  const PredictionResponse({
    required this.predictions,
    required this.latencyMs,
    required this.model,
  });

  factory PredictionResponse.fromJson(Map<String, dynamic> json) {
    final predictionsJson = json['predictions'] as List<dynamic>;
    final predictions = predictionsJson
        .map((e) => PredictionItem.fromJson(e as Map<String, dynamic>))
        .toList();

    return PredictionResponse(
      predictions: predictions,
      latencyMs: (json['latency_ms'] as num).toDouble(),
      model: json['model'] as String,
    );
  }

  final List<PredictionItem> predictions;
  final double latencyMs;
  final String model;

  Map<String, dynamic> toJson() {
    return {
      'predictions': predictions.map((e) => e.toJson()).toList(),
      'latency_ms': latencyMs,
      'model': model,
    };
  }

  @override
  String toString() =>
      'PredictionResponse(predictions: $predictions, latencyMs: $latencyMs, model: $model)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is PredictionResponse &&
          runtimeType == other.runtimeType &&
          predictions == other.predictions &&
          latencyMs == other.latencyMs &&
          model == other.model;

  @override
  int get hashCode =>
      predictions.hashCode ^ latencyMs.hashCode ^ model.hashCode;
}
