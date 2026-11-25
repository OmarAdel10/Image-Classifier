import 'dart:async';
import 'dart:io';
import 'dart:convert';
import 'dart:developer';

import 'package:path/path.dart' as p;
import 'package:image_classifier/services/backend_locator.dart';

/// Result of starting the backend.
class BackendStartResult {
  final Process process;
  final int port;
  final String baseUrl;

  BackendStartResult({
    required this.process,
    required this.port,
    required this.baseUrl,
  });
}

/// Manages the backend subprocess.
class BackendProcessManager {
  Process? _process;
  int? _port;
  String? _baseUrl;

  /// The current backend process, if running.
  Process? get process => _process;

  /// The port the backend is running on.
  int? get port => _port;

  /// The base URL of the backend.
  String? get baseUrl => _baseUrl;

  /// Whether the backend is currently running.
  bool get isRunning => _process != null;

  /// Starts the backend on the given port (or finds a free one).
  ///
  /// [preferredPort] is the port to try first (default 8000).
  /// Returns the start result with the actual port and base URL.
  Future<BackendStartResult> start({int preferredPort = 8000}) async {
    if (_process != null) {
      throw StateError('Backend already running');
    }

    final location = await locateBackend();
    _port = await _findFreePort(preferredPort);
    _baseUrl = 'http://127.0.0.1:$_port';

    final env = <String, String>{
      ...Platform.environment,
      'CNN_HOST': '127.0.0.1',
      'CNN_PORT': '$_port',
      'PYTHONPATH': p.join(location.backendDir.path, 'src'),
    };

    _process = await Process.start(
      location.pythonExecutable,
      [location.scriptPath],
      workingDirectory: location.backendDir.path,
      environment: env,
      mode: ProcessStartMode.normal,
    );

    // Set up stdout/stderr piping to the app's console
    _process!.stdout
        .transform(utf8.decoder)
        .listen((line) => log('[backend] $line'));
    _process!.stderr
        .transform(utf8.decoder)
        .listen((line) => log('[backend] $line'));

    // Watch for process exit
    _process!.exitCode.then((code) {
      log('[backend] Process exited with code $code');
      _process = null;
      _port = null;
      _baseUrl = null;
    });

    // Give the process a moment to start
    await Future.delayed(const Duration(milliseconds: 500));

    if (_process == null) {
      throw BackendStartException('Backend process terminated immediately');
    }

    return BackendStartResult(
      process: _process!,
      port: _port!,
      baseUrl: _baseUrl!,
    );
  }

  /// Stops the backend gracefully.
  Future<void> stop({Duration gracePeriod = const Duration(seconds: 5)}) async {
    if (_process == null) return;

    // Close stdin first (our launcher watches for this)
    _process!.stdin.close();

    try {
      await _process!.exitCode.timeout(gracePeriod);
    } on TimeoutException {
      // Force kill if graceful shutdown fails
      _process!.kill(ProcessSignal.sigterm);
      await Future.delayed(const Duration(milliseconds: 500));
      if (_process != null) {
        _process!.kill(ProcessSignal.sigkill);
        await _process!.exitCode;
      }
    }

    _process = null;
    _port = null;
    _baseUrl = null;
  }

  /// Restarts the backend.
  Future<BackendStartResult> restart({int preferredPort = 8000}) async {
    await stop();
    return start(preferredPort: preferredPort);
  }

  /// Finds a free port starting from [preferredPort].
  Future<int> _findFreePort(int preferredPort) async {
    // Try preferred port first
    if (await _isPortFree(preferredPort)) return preferredPort;

    // Try a few ephemeral ports
    for (int i = 0; i < 10; i++) {
      final port = preferredPort + 1 + i;
      if (await _isPortFree(port)) return port;
    }

    // Fall back to port 0 (OS assigns)
    final server = await ServerSocket.bind('127.0.0.1', 0);
    final port = server.port;
    await server.close();
    return port;
  }

  Future<bool> _isPortFree(int port) async {
    try {
      final server = await ServerSocket.bind('127.0.0.1', port);
      await server.close();
      return true;
    } catch (_) {
      return false;
    }
  }
}

class BackendStartException implements Exception {
  final String message;
  BackendStartException(this.message);
  @override
  String toString() => 'BackendStartException: $message';
}
