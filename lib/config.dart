import 'dart:io';

import 'package:shared_preferences/shared_preferences.dart';

/// Application configuration constants.
class Config {
  Config._();

  /// SharedPreferences key holding the runtime API base URL override.
  static const String apiBaseUrlPrefsKey = 'api_base_url';

  /// Compile-time API base URL override.
  ///
  /// Set via `--dart-define=API_BASE_URL=https://example.com`. Empty when
  /// not supplied, in which case the runtime override or the
  /// platform-appropriate default is used.
  static const String compileTimeApiBaseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: '',
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

  /// Platform-appropriate default API base URL.
  ///
  /// On Android the host machine's loopback is reachable at `10.0.2.2`, not
  /// `localhost` (which resolves to the emulator itself). On every other
  /// platform `localhost` addresses the host directly.
  static String get platformDefaultApiBaseUrl {
    if (Platform.isAndroid) {
      return 'http://10.0.2.2:8000';
    }
    return 'http://localhost:8000';
  }

  /// Effective API base URL without consulting runtime storage.
  ///
  /// Priority: compile-time `--dart-define` override, then the
  /// platform-appropriate default. Use [resolveApiBaseUrl] to also honour a
  /// runtime override stored via [setApiBaseUrl].
  static String get apiBaseUrl {
    if (compileTimeApiBaseUrl.isNotEmpty) {
      return compileTimeApiBaseUrl;
    }
    return platformDefaultApiBaseUrl;
  }

  /// Resolves the API base URL, honouring the runtime override.
  ///
  /// Priority: compile-time `--dart-define` override, then the runtime
  /// SharedPreferences override, then the platform-appropriate default.
  ///
  /// The runtime override is ignored when a compile-time override is present
  /// so a production build cannot be silently redirected by stored state.
  static Future<String> resolveApiBaseUrl() async {
    if (compileTimeApiBaseUrl.isNotEmpty) {
      return compileTimeApiBaseUrl;
    }
    try {
      final prefs = await SharedPreferences.getInstance();
      final runtimeUrl = prefs.getString(apiBaseUrlPrefsKey);
      if (runtimeUrl != null && runtimeUrl.isNotEmpty) {
        return runtimeUrl;
      }
    } catch (_) {
      // Storage unavailable (e.g. no platform plugin) — use the default.
    }
    return platformDefaultApiBaseUrl;
  }

  /// Stores a runtime API base URL override.
  ///
  /// Throws [ArgumentError] when [url] is not a well-formed http(s) URL.
  /// Pass `null` or an empty string to clear the override.
  static Future<void> setApiBaseUrl(String? url) async {
    if (url == null || url.isEmpty) {
      await clearApiBaseUrl();
      return;
    }
    if (!isValidApiBaseUrl(url)) {
      throw ArgumentError(
        'Invalid API base URL "$url": '
        'must be a well-formed http/https URL with a host.',
      );
    }
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(apiBaseUrlPrefsKey, url);
  }

  /// Clears any runtime API base URL override.
  static Future<void> clearApiBaseUrl() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove(apiBaseUrlPrefsKey);
    } catch (_) {
      // Storage unavailable — nothing to clear.
    }
  }

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
