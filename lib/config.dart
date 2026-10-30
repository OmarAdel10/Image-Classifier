/// Application configuration constants.
class Config {
  Config._();

  /// Base URL for the FastAPI backend.
  static const String apiBaseUrl = 'http://localhost:8000';

  /// Timeout for HTTP requests to the backend.
  static const Duration requestTimeout = Duration(seconds: 30);

  /// Maximum allowed image file size in bytes (10 MiB).
  static const int maxImageSizeBytes = 10 * 1024 * 1024;
}
