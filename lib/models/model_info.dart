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
      trainedAt = DateTime.tryParse(trainedAtJson as String)?.toLocal();
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
          _listEquals(classes, other.classes) &&
          device == other.device &&
          trainedAt == other.trainedAt;

  bool _listEquals(List<String> a, List<String> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  @override
  int get hashCode {
    int result = modelName.hashCode;
    result = 31 * result + inputSize.hashCode;
    for (final cls in classes) {
      result = 31 * result + cls.hashCode;
    }
    result = 31 * result + device.hashCode;
    result = 31 * result + (trainedAt?.hashCode ?? 0);
    return result;
  }
}
