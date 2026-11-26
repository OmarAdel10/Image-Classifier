import 'dart:async';
import 'dart:convert';
import 'dart:developer';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:image_classifier/config.dart';
import 'package:image_classifier/models/backend_status.dart';
import 'package:image_classifier/services/api_client.dart';
import 'package:image_classifier/services/backend_process_manager.dart';

/// Controller for the backend lifecycle.
///
/// Handles:
/// - Determining connection mode (compile-time, custom, managed)
/// - Starting/stopping the managed backend
/// - Polling /health and /ready endpoints
/// - Providing the resolved base URL to the API client
class BackendController extends ChangeNotifier {
  BackendController({required this.apiClient, this.onStatusChange});

  final ApiClient apiClient;
  final void Function(BackendStatus)? onStatusChange;

  BackendStatus _status = const BackendStatus(
    mode: BackendConnectionMode.managed,
    managedStatus: ManagedBackendStatus.stopped,
  );
  final BackendProcessManager _processManager = BackendProcessManager();
  Timer? _pollTimer;
  String? _resolvedBaseUrl;

  BackendStatus get status => _status;
  String? get resolvedBaseUrl => _resolvedBaseUrl;
  bool get isManaged => _status.isManaged;
  bool get isReady => _status.isReady;

  /// Initializes the backend connection.
  ///
  /// Returns the resolved base URL for the API client.
  Future<String> initialize() async {
    // 1. Check for compile-time override
    if (Config.compileTimeApiBaseUrl.isNotEmpty) {
      final compileTimeUrl = Config.compileTimeApiBaseUrl;
      _status = BackendStatus.compileTime(baseUrl: compileTimeUrl);
      _resolvedBaseUrl = compileTimeUrl;
      notifyListeners();
      onStatusChange?.call(_status);
      await _verifyExternalBackend(compileTimeUrl);
      return compileTimeUrl;
    }

    // 2. Check for saved custom URL by resolving the effective URL
    final resolvedUrl = await Config.resolveApiBaseUrl();
    // If the resolved URL differs from the platform default, user has set a custom URL
    final isCustom = resolvedUrl != Config.platformDefaultApiBaseUrl;
    if (isCustom) {
      if (await _isManagedDefault(resolvedUrl)) {
        // It's a loopback default, treat as managed
        return _startManagedBackend();
      }
      _status = BackendStatus.custom(baseUrl: resolvedUrl);
      _resolvedBaseUrl = resolvedUrl;
      notifyListeners();
      onStatusChange?.call(_status);
      await _verifyExternalBackend(resolvedUrl);
      return resolvedUrl;
    }

    // 3. Default: managed backend
    return _startManagedBackend();
  }

  /// Starts the managed backend and waits for it to be ready.
  Future<String> _startManagedBackend() async {
    _status = _status.copyWith(managedStatus: ManagedBackendStatus.starting);
    notifyListeners();
    onStatusChange?.call(_status);

    try {
      // Start the backend process
      final result = await _processManager.start();
      _status = _status.copyWith(
        managedStatus: ManagedBackendStatus.probing,
        baseUrl: result.baseUrl,
        port: result.port,
      );
      notifyListeners();
      onStatusChange?.call(_status);

      // Poll /health until backend responds
      await _pollHealth(result.baseUrl);

      _status = _status.copyWith(
        managedStatus: ManagedBackendStatus.modelLoading,
      );
      notifyListeners();
      onStatusChange?.call(_status);

      // Poll /ready until model is loaded
      await _pollReady(result.baseUrl);

      _resolvedBaseUrl = result.baseUrl;
      _status = _status.copyWith(
        managedStatus: ManagedBackendStatus.ready,
        modelInfo: 'Model loaded',
      );
      notifyListeners();
      onStatusChange?.call(_status);

      return result.baseUrl;
    } catch (e) {
      _status = _status.copyWith(
        managedStatus: ManagedBackendStatus.failed,
        error: e.toString(),
      );
      notifyListeners();
      onStatusChange?.call(_status);
      rethrow;
    }
  }

  /// Verifies an external backend is reachable.
  Future<void> _verifyExternalBackend(String url) async {
    try {
      final healthy = await _checkHealth(url);
      if (!healthy) {
        throw Exception('Backend at $url returned non-200 health check');
      }
      // Also check it's our backend (service marker)
      final isOurs = await _checkServiceMarker(url);
      if (!isOurs) {
        // Not our backend, but we'll still use it if user configured it
        log(
          'Warning: Backend at $url does not have cnn-classifier service marker',
        );
      }
    } catch (e) {
      _status = _status.copyWith(error: e.toString());
      notifyListeners();
      onStatusChange?.call(_status);
      rethrow;
    }
  }

  /// Polls /health until it returns 200 or timeout.
  Future<void> _pollHealth(String baseUrl) async {
    const maxAttempts = 120; // 30 seconds at 250ms intervals
    const interval = Duration(milliseconds: 250);

    for (int i = 0; i < maxAttempts; i++) {
      try {
        final healthy = await _checkHealth(baseUrl);
        if (healthy) return;
      } catch (_) {
        // Ignore, keep polling
      }
      await Future.delayed(interval);
    }
    throw Exception('Backend health check timed out after 30s');
  }

  /// Polls /ready until model is loaded or timeout.
  Future<void> _pollReady(String baseUrl) async {
    const maxAttempts = 240; // 60 seconds at 250ms intervals
    const interval = Duration(milliseconds: 250);

    for (int i = 0; i < maxAttempts; i++) {
      try {
        final ready = await _checkReady(baseUrl);
        if (ready) return;
        // Update progress
        _status = _status.copyWith(
          modelInfo: 'Loading model... (${i * 250}ms)',
        );
        notifyListeners();
      } catch (_) {
        // Ignore, keep polling
      }
      await Future.delayed(interval);
    }
    throw Exception('Model load timed out after 60s');
  }

  /// Checks /health endpoint.
  Future<bool> _checkHealth(String baseUrl) async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Checks /ready endpoint.
  Future<bool> _checkReady(String baseUrl) async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/ready'))
          .timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        final json = jsonDecode(response.body);
        return json['status'] == 'ready' && json['model_loaded'] == true;
      }
      return false;
    } catch (_) {
      return false;
    }
  }

  /// Checks if the backend has our service marker.
  Future<bool> _checkServiceMarker(String baseUrl) async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/health'))
          .timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        final json = jsonDecode(response.body);
        return json['service'] == 'cnn-classifier';
      }
    } catch (_) {}
    return false;
  }

  /// Checks if a URL is a managed default (loopback).
  Future<bool> _isManagedDefault(String url) async {
    try {
      final uri = Uri.parse(url);
      return uri.host == '127.0.0.1' || uri.host == 'localhost';
    } catch (_) {
      return false;
    }
  }

  /// Switches to a custom backend URL.
  Future<void> setCustomUrl(String url) async {
    await Config.setApiBaseUrl(url);
    await stopManagedBackend();
    _status = BackendStatus.custom(baseUrl: url);
    _resolvedBaseUrl = url;
    apiClient.invalidateBaseUrlCache();
    notifyListeners();
    onStatusChange?.call(_status);
  }

  /// Resets to managed backend mode.
  Future<void> resetToManaged() async {
    await Config.clearApiBaseUrl();
    await stopManagedBackend();
    await _startManagedBackend();
    apiClient.invalidateBaseUrlCache();
    // _status updated by _startManagedBackend
  }

  /// Stops the managed backend if running.
  Future<void> stopManagedBackend() async {
    _pollTimer?.cancel();
    _pollTimer = null;
    await _processManager.stop();
  }

  /// Restarts the managed backend.
  Future<void> restartManagedBackend() async {
    await stopManagedBackend();
    await _startManagedBackend();
    apiClient.invalidateBaseUrlCache();
  }

  @override
  void dispose() {
    stopManagedBackend();
    super.dispose();
  }
}
