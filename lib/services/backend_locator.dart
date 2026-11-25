import 'dart:io';

import 'package:path/path.dart' as p;

/// Result of locating the backend.
class BackendLocation {
  final Directory backendDir;
  final String pythonExecutable;
  final String scriptPath;

  BackendLocation({
    required this.backendDir,
    required this.pythonExecutable,
    required this.scriptPath,
  });
}

/// Locates the backend directory relative to the app executable.
///
/// The backend is expected at `<app_dir>/../backend` when running from the
/// repo, or bundled next to the executable in a packaged build.
Future<BackendLocation> locateBackend() async {
  final exe = Platform.resolvedExecutable;
  final appDir = File(exe).parent;

  // 1. Check if running from repo: backend is at <app_dir>/../backend
  final repoBackend = Directory(p.join(appDir.path, '..', 'backend'));
  if (await repoBackend.exists()) {
    final script = File(p.join(repoBackend.path, 'scripts', 'run_backend.py'));
    if (await script.exists()) {
      return BackendLocation(
        backendDir: repoBackend,
        pythonExecutable: _findPython(),
        scriptPath: script.path,
      );
    }
  }

  // 2. Check if backend is bundled next to the executable
  final bundledBackend = Directory(p.join(appDir.path, 'backend'));
  if (await bundledBackend.exists()) {
    final script = File(
      p.join(bundledBackend.path, 'scripts', 'run_backend.py'),
    );
    if (await script.exists()) {
      return BackendLocation(
        backendDir: bundledBackend,
        pythonExecutable: _findPython(),
        scriptPath: script.path,
      );
    }
  }

  // 3. Fallback: try to find python and assume backend is in a known location
  // This is a last resort for development
  final fallbackBackend = Directory(
    '/mnt/ALL/UniProjects/deep-learning/backend',
  );
  if (await fallbackBackend.exists()) {
    final script = File(
      p.join(fallbackBackend.path, 'scripts', 'run_backend.py'),
    );
    if (await script.exists()) {
      return BackendLocation(
        backendDir: fallbackBackend,
        pythonExecutable: _findPython(),
        scriptPath: script.path,
      );
    }
  }

  throw BackendNotFoundException('Could not locate backend directory');
}

String _findPython() {
  // Try common Python executables
  const candidates = ['python3', 'python'];
  for (final cmd in candidates) {
    if (_which(cmd)) return cmd;
  }
  return 'python3'; // Default, will fail with clear error if not found
}

bool _which(String command) {
  try {
    final result = Process.runSync('which', [command]);
    return result.exitCode == 0;
  } catch (_) {
    return false;
  }
}

class BackendNotFoundException implements Exception {
  final String message;
  BackendNotFoundException(this.message);
  @override
  String toString() => 'BackendNotFoundException: $message';
}
