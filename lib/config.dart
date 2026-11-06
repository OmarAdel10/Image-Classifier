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

  /// Platform-appropriate default API base URL for [isAndroid].
  ///
  /// On Android the host machine's loopback is reachable at `10.0.2.2`, not
  /// `localhost` (which resolves to the emulator itself). On every other
  /// platform `localhost` addresses the host directly.
  static String platformDefaultApiBaseUrlFor({required bool isAndroid}) {
    return isAndroid ? 'http://10.0.2.2:8000' : 'http://localhost:8000';
  }

  /// Platform-appropriate default API base URL for the current platform.
  static String get platformDefaultApiBaseUrl =>
      platformDefaultApiBaseUrlFor(isAndroid: Platform.isAndroid);

  /// Trims whitespace and removes trailing slashes from [url].
  ///
  /// Callers append paths such as `/health` to the base URL, so a trailing
  /// slash would otherwise produce `//health`.
  static String normalizeApiBaseUrl(String url) {
    var normalized = url.trim();
    while (normalized.endsWith('/')) {
      normalized = normalized.substring(0, normalized.length - 1);
    }
    return normalized;
  }

  /// Resolves the API base URL, honouring the runtime override.
  ///
  /// Priority: compile-time `--dart-define` override, then the runtime
  /// SharedPreferences override, then the platform-appropriate default.
  ///
  /// The runtime override is ignored when a compile-time override is present
  /// so a production build cannot be silently redirected by stored state.
  /// A stored value that is no longer a valid base URL is ignored.
  static Future<String> resolveApiBaseUrl({
    String compileTimeOverride = compileTimeApiBaseUrl,
  }) async {
    if (compileTimeOverride.isNotEmpty) {
      return normalizeApiBaseUrl(compileTimeOverride);
    }
    try {
      final prefs = await SharedPreferences.getInstance();
      final runtimeUrl = prefs.getString(apiBaseUrlPrefsKey);
      if (runtimeUrl != null && runtimeUrl.isNotEmpty) {
        final normalized = normalizeApiBaseUrl(runtimeUrl);
        if (isValidApiBaseUrl(normalized)) {
          return normalized;
        }
      }
    } catch (_) {
      // Storage unavailable (e.g. no platform plugin) — use the default.
    }
    return platformDefaultApiBaseUrl;
  }

  /// Stores a runtime API base URL override.
  ///
  /// The URL is trimmed and stripped of trailing slashes before it is stored.
  /// Throws [ArgumentError] when [url] is not a well-formed http(s) URL, and
  /// [StateError] when the store refuses the write or when
  /// [compileTimeOverride] pins the build: a value that cannot take effect
  /// now would be a dormant redirect for a later unpinned build.
  /// A refused write restores the previously stored URL instead of dropping
  /// it, so a failed save never discards a working backend address.
  /// Pass `null`, an empty string, or whitespace to clear the override. The
  /// clear path runs before the pinned check, so a pinned build may still
  /// remove a stored override; it can never create one.
  /// The URL is validated before the pinned check, so a malformed URL is
  /// reported as malformed even in a pinned build.
  static Future<void> setApiBaseUrl(
    String? url, {
    String compileTimeOverride = compileTimeApiBaseUrl,
  }) async {
    if (url == null || url.trim().isEmpty) {
      // Clearing is allowed even in a pinned build: a stored override can
      // never take effect there, so removing it cannot become a dormant
      // redirect. A refused removal is reported instead of being swallowed,
      // so callers never claim a reset that did not happen.
      if (!await clearApiBaseUrl()) {
        throw StateError('Could not clear the stored API base URL.');
      }
      return;
    }
    final normalized = normalizeApiBaseUrl(url);
    if (!isValidApiBaseUrl(normalized)) {
      throw ArgumentError(
        'Invalid API base URL: must be a well-formed http/https URL '
        'with a host and no query or fragment.',
      );
    }
    if (compileTimeOverride.isNotEmpty) {
      throw StateError(
        'Refusing to store a runtime API base URL while '
        '--dart-define=API_BASE_URL is set.',
      );
    }
    final SharedPreferences prefs;
    try {
      prefs = await SharedPreferences.getInstance();
    } catch (_) {
      throw StateError('Could not store the API base URL.');
    }
    // The value that is in use before the write, so the rollback can put it
    // back if the platform cannot be read any more.
    final previous = prefs.getString(apiBaseUrlPrefsKey);
    final bool stored;
    try {
      stored = await prefs.setString(apiBaseUrlPrefsKey, normalized);
    } catch (_) {
      await _rollbackStoredUrl(prefs, previous);
      throw StateError('Could not store the API base URL.');
    }
    if (!stored) {
      // The platform refused the write. Reload the persisted state, so the
      // app keeps using the URL that is actually stored instead of the one
      // that was never saved.
      await _rollbackStoredUrl(prefs, previous);
      throw StateError('Could not store the API base URL.');
    }
  }

  /// Restores the persisted override after a failed write.
  ///
  /// `shared_preferences` writes to its in-memory cache before it calls the
  /// platform store, so a write that the platform refuses still leaves the
  /// new value visible for the rest of the session. Reloading rebuilds the
  /// cache from the platform, which restores the previously stored URL
  /// instead of deleting it: removing the key would throw away a working
  /// backend address the user configured earlier.
  ///
  /// When the platform cannot be read any more, the reload fails and the
  /// cache would keep the refused URL. In that case [previous] is written
  /// back, which restores the cache even when the platform refuses that
  /// write as well, because `shared_preferences` updates the cache before it
  /// calls the store.
  static Future<void> _rollbackStoredUrl(
    SharedPreferences prefs,
    String? previous,
  ) async {
    try {
      await prefs.reload();
      return;
    } catch (_) {
      // The platform is unreadable as well: fall through and put the
      // previous value back into the cache.
    }
    try {
      if (previous == null) {
        await prefs.remove(apiBaseUrlPrefsKey);
      } else {
        await prefs.setString(apiBaseUrlPrefsKey, previous);
      }
    } catch (_) {
      // The write-back failed as well. `shared_preferences` updates its
      // cache before it calls the store, so the cache already holds the
      // restored address (or no address at all); the caller reports the
      // storage failure.
    }
  }

  /// Clears any runtime API base URL override.
  ///
  /// Returns true when the override is gone (including when none was stored)
  /// and false when storage is unavailable or the platform refused the
  /// removal, so callers do not report a successful reset while the platform
  /// may still hold the old URL.
  ///
  /// A refused removal still clears the in-memory cache, so the override is
  /// not active in this session, but the platform may restore it after a
  /// restart.
  static Future<bool> clearApiBaseUrl() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      return await prefs.remove(apiBaseUrlPrefsKey);
    } catch (_) {
      return false;
    }
  }

  /// Returns true when [url] is a well-formed http(s) base URL.
  ///
  /// Requires scheme http/https, a non-empty host, a port in 1-65535, no
  /// embedded credentials, no query or fragment, and no whitespace or
  /// percent-encoded characters anywhere in the URL. Guards against URL
  /// injection via a malformed `--dart-define=API_BASE_URL=...` (IC-60) and
  /// against base URLs that would break path appending (`$url/health`).
  static bool isValidApiBaseUrl(String url) {
    final trimmed = url.trim();
    if (trimmed.isEmpty || RegExp(r'\s').hasMatch(trimmed)) return false;
    // Percent-encoding is rejected anywhere, not just in the host: the base
    // URL is concatenated with '/health' and '/predict', so an encoded
    // segment in the path could resolve to a different effective path than
    // the one that was reviewed.
    if (trimmed.contains('%')) return false;
    final uri = Uri.tryParse(trimmed);
    if (uri == null) return false;
    if (uri.scheme != 'http' && uri.scheme != 'https') return false;
    if (uri.host.isEmpty || uri.userInfo.isNotEmpty) return false;
    if (uri.hasQuery || uri.hasFragment) return false;
    if (uri.port < 1 || uri.port > 65535) return false;
    return true;
  }

  /// Enforcement half of [isValidApiBaseUrl] (IC-60): throws [StateError]
  /// when [url] is not a well-formed http(s) base URL.
  ///
  /// The error message is deliberately generic so a credential-bearing
  /// override is never echoed into crash logs.
  static void ensureValidApiBaseUrl(String url) {
    if (!isValidApiBaseUrl(url)) {
      throw StateError(
        'Invalid API_BASE_URL: must be a well-formed http/https URL with '
        'a host and no query or fragment.',
      );
    }
  }

  /// Validates the compile-time override at startup.
  ///
  /// No-op when no override was supplied. A malformed
  /// `--dart-define=API_BASE_URL=...` must fail fast instead of silently
  /// pointing requests at an attacker-controlled URL (IC-60).
  ///
  /// [compileTimeOverride] defaults to [compileTimeApiBaseUrl]; tests pass an
  /// explicit value.
  static void validateStartupApiBaseUrl({
    String compileTimeOverride = compileTimeApiBaseUrl,
  }) {
    if (compileTimeOverride.isNotEmpty) {
      ensureValidApiBaseUrl(normalizeApiBaseUrl(compileTimeOverride));
    }
  }
}
