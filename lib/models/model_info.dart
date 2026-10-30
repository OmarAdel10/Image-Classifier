/// Represents model metadata returned by the /model-info endpoint.
class ModelInfo {
  const ModelInfo({
    required this.modelName,
    required this.inputSize,
    required this.classes,
    required this.device,
    this.trainedAt,
  });

  factory ModelInfo.fromJson(Map<String, dynamic> json) {
    DateTime? trainedAt;
    final trainedAtJson = json['trained_at'];
    if (trainedAtJson != null) {
      trainedAt = DateTime.parse(trainedAtJson as String).toLocal();
    }

    return ModelInfo(
      modelName: json['model'] as String,
      inputSize: json['input_size'] as String,
      classes: (json['classes'] as List<dynamic>)
          .map((e) => e as String)
          .toList(),
      device: json['device'] as String,
      trainedAt: trainedAt,
    );
  }

  final String modelName;
  final String inputSize;
  final List<String> classes;
  final String device;
  final DateTime? trainedAt;

  Map<String, dynamic> toJson() {
    return {
      'model': modelName,
      'input_size': inputSize,
      'classes': classes,
      'device': device,
      'trained_at': trainedAt?.toIso8601String(),
    };
  }

  @override
  String toString() =>
      'ModelInfo(modelName: $modelName, inputSize: $inputSize, classes: $classes, device: $device, trainedAt: $trainedAt)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ModelInfo &&
          runtimeType == other.runtimeType &&
          modelName == other.modelName &&
          inputSize == other.inputSize &&
          classes == other.classes &&
          device == other.device &&
          trainedAt == other.trainedAt;

  @override
  int get hashCode =>
      modelName.hashCode ^
      inputSize.hashCode ^
      classes.hashCode ^
      device.hashCode ^
      trainedAt.hashCode;
}
