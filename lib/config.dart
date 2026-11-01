/// Application configuration constants.
class Config {
  Config._();

  /// Base URL for the FastAPI backend.
  /// Can be overridden via `--dart-define=API_BASE_URL=https://example.com`
  static const String apiBaseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://localhost:8000',
  );

  /// Timeout for HTTP requests to the backend.
  /// Can be overridden via `--dart-define=REQUEST_TIMEOUT_SEC=15`
  static const Duration requestTimeout = Duration(
    seconds: int.fromEnvironment('REQUEST_TIMEOUT_SEC', defaultValue: 15),
  );

  /// Maximum allowed image file size in bytes (10 MiB).
  static const int maxImageSizeBytes = 10 * _kMebibyte;

  /// 1 MiB in bytes.
  static const int _kMebibyte = 1024 * 1024;

  /// Returns true when [url] is a well-formed http(s) base URL
  /// (scheme http/https and a non-empty host). Guards against URL
  /// injection via a malformed `--dart-define=API_BASE_URL=...`
  /// (IC-60).
  static bool isValidApiBaseUrl(String url) {
    final uri = Uri.tryParse(url);
    return uri != null &&
        (uri.scheme == 'http' || uri.scheme == 'https') &&
        uri.host.isNotEmpty;
  }

  /// Enforcement half of [isValidApiBaseUrl] (IC-60): throws [StateError]
  /// when [url] is not a well-formed http(s) base URL. main() calls this
  /// so a malformed `--dart-define=API_BASE_URL=...` fails fast at startup
  /// instead of silently pointing requests at an attacker-controlled URL.
  static void ensureValidApiBaseUrl(String url) {
    if (!isValidApiBaseUrl(url)) {
      throw StateError(
        'Invalid API_BASE_URL "$url": '
        'must be a well-formed http/https URL with a host.',
      );
    }
  }
}
