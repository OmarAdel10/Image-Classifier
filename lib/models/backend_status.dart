/// Backend status and connection mode.
enum BackendConnectionMode {
  /// Backend URL comes from --dart-define at compile time.
  compileTime,

  /// Backend URL saved in SharedPreferences by the user.
  custom,

  /// Backend is managed by the app (auto-started).
  managed,
}

/// Status of the managed backend.
enum ManagedBackendStatus {
  /// Not started yet.
  stopped,

  /// Process is starting.
  starting,

  /// Backend process running, waiting for /health.
  probing,

  /// Backend healthy, waiting for /ready (model loading).
  modelLoading,

  /// Backend ready, model loaded.
  ready,

  /// Backend failed to start or model failed to load.
  failed,
}

/// Overall backend status.
class BackendStatus {
  final BackendConnectionMode mode;
  final ManagedBackendStatus? managedStatus;
  final String? baseUrl;
  final int? port;
  final String? error;
  final String? logPath;
  final String? modelInfo;

  const BackendStatus({
    required this.mode,
    this.managedStatus,
    this.baseUrl,
    this.port,
    this.error,
    this.logPath,
    this.modelInfo,
  });

  factory BackendStatus.managed({
    required ManagedBackendStatus status,
    String? baseUrl,
    int? port,
    String? error,
    String? logPath,
    String? modelInfo,
  }) {
    return BackendStatus(
      mode: BackendConnectionMode.managed,
      managedStatus: status,
      baseUrl: baseUrl,
      port: port,
      error: error,
      logPath: logPath,
      modelInfo: modelInfo,
    );
  }

  factory BackendStatus.custom({required String baseUrl}) {
    return BackendStatus(mode: BackendConnectionMode.custom, baseUrl: baseUrl);
  }

  factory BackendStatus.compileTime({required String baseUrl}) {
    return BackendStatus(
      mode: BackendConnectionMode.compileTime,
      baseUrl: baseUrl,
    );
  }

  bool get isManaged => mode == BackendConnectionMode.managed;
  bool get isExternal => mode != BackendConnectionMode.managed;
  bool get isReady => managedStatus == ManagedBackendStatus.ready;
  bool get isFailed => managedStatus == ManagedBackendStatus.failed;

  BackendStatus copyWith({
    BackendConnectionMode? mode,
    ManagedBackendStatus? managedStatus,
    String? baseUrl,
    int? port,
    String? error,
    String? logPath,
    String? modelInfo,
  }) {
    return BackendStatus(
      mode: mode ?? this.mode,
      managedStatus: managedStatus ?? this.managedStatus,
      baseUrl: baseUrl ?? this.baseUrl,
      port: port ?? this.port,
      error: error ?? this.error,
      logPath: logPath ?? this.logPath,
      modelInfo: modelInfo ?? this.modelInfo,
    );
  }
}
