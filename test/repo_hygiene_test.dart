import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

void main() {
  group('Repo Hygiene - .gitignore enforcement', () {
    final root = Directory.current;
    final gitignore = File('${root.path}/.gitignore');

    setUpAll(() {
      if (!gitignore.existsSync()) {
        fail('.gitignore not found at repo root');
      }
    });

    test('.gitignore ignores Python bytecode directories', () {
      final patterns = gitignore.readAsLinesSync();
      // Should match __pycache__/ directories anywhere
      expect(patterns, contains('__pycache__/'));
    });

    test('.gitignore ignores pytest cache', () {
      final patterns = gitignore.readAsLinesSync();
      expect(patterns, contains('.pytest_cache/'));
    });

    test('.gitignore ignores virtual environments', () {
      final patterns = gitignore.readAsLinesSync();
      expect(patterns, contains('.venv/'));
    });

    test('.gitignore ignores coverage output', () {
      final patterns = gitignore.readAsLinesSync();
      expect(patterns, contains('.coverage'));
    });

    test('.gitignore ignores egg-info directories', () {
      final patterns = gitignore.readAsLinesSync();
      expect(patterns, contains('*.egg-info/'));
    });

    test('.gitignore ignores backend checkpoints', () {
      final patterns = gitignore.readAsLinesSync();
      expect(patterns, contains('backend/checkpoints/'));
    });

    test('.gitignore ignores backend .env', () {
      final patterns = gitignore.readAsLinesSync();
      expect(patterns, contains('backend/.env'));
    });

    test('.gitignore ignores root .env files', () {
      final patterns = gitignore.readAsLinesSync();
      expect(patterns, contains('/.env'));
      expect(patterns, contains('.env.local'));
      expect(patterns, contains('.env.*.local'));
    });

    test('.gitignore ignores docs/ directory', () {
      final patterns = gitignore.readAsLinesSync();
      expect(patterns, contains('docs/'));
    });

    test('.gitignore ignores Flutter build artifacts', () {
      final patterns = gitignore.readAsLinesSync();
      expect(patterns, contains('/build/'));
      expect(patterns, contains('.dart_tool/'));
    });

    test('git check-ignore confirms patterns work', () {
      // Verify git itself recognizes these patterns
      final testPaths = [
        'backend/__pycache__/foo.pyc',
        'backend/.pytest_cache/v/cache',
        'backend/.venv/bin/python',
        '.coverage',
        'backend/some.egg-info/PKG-INFO',
        'backend/checkpoints/best_model.pt',
        'backend/.env',
        '.env',
        '.env.local',
        '.env.prod.local',
        'docs/superpowers/plan.md',
        'build/app.apk',
        '.dart_tool/package_config.json',
      ];

      for (final path in testPaths) {
        final result = Process.runSync('git', [
          'check-ignore',
          path,
        ], workingDirectory: root.path);
        // Exit code 0 = ignored, 1 = not ignored, 128 = error
        expect(
          result.exitCode,
          0,
          reason: 'Expected "$path" to be ignored by .gitignore',
        );
      }
    });
  });
}
