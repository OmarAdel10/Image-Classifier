import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:image_classifier/config.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:shared_preferences_platform_interface/shared_preferences_platform_interface.dart';

import 'support/throwing_prefs_store.dart';

void main() {
  group('Config runtime API base URL', () {
    setUp(() {
      // Seed an empty mock store so every test starts without an override.
      SharedPreferences.setMockInitialValues({});
    });

    group('platformDefaultApiBaseUrlFor', () {
      test('uses the Android emulator host loopback for Android', () {
        expect(
          Config.platformDefaultApiBaseUrlFor(isAndroid: true),
          'http://10.0.2.2:8000',
        );
      });

      test('uses localhost on non-Android platforms', () {
        expect(
          Config.platformDefaultApiBaseUrlFor(isAndroid: false),
          'http://localhost:8000',
        );
      });
    });

    group('platformDefaultApiBaseUrl', () {
      test('is http://localhost:8000 on Linux', () {
        expect(Platform.isLinux, isTrue);
        expect(Config.platformDefaultApiBaseUrl, 'http://localhost:8000');
      });
    });

    group('normalizeApiBaseUrl', () {
      test('trims whitespace and strips trailing slashes', () {
        expect(
          Config.normalizeApiBaseUrl('  http://host:8000/  '),
          'http://host:8000',
        );
      });

      test('strips multiple trailing slashes', () {
        expect(
          Config.normalizeApiBaseUrl('http://host:8000///'),
          'http://host:8000',
        );
      });

      test('leaves a URL without a trailing slash unchanged', () {
        expect(
          Config.normalizeApiBaseUrl('http://host:8000'),
          'http://host:8000',
        );
      });
    });

    group('apiBaseUrl', () {
      test('returns platform default when no compile-time override', () {
        expect(Config.compileTimeApiBaseUrl, isEmpty);
        expect(Config.apiBaseUrl, Config.platformDefaultApiBaseUrl);
        expect(Config.apiBaseUrl, 'http://localhost:8000');
      });

      test('effectiveApiBaseUrl uses a non-empty override, normalized', () {
        expect(
          Config.effectiveApiBaseUrl(
            compileTimeOverride: '  http://compile.example.com:8000/  ',
          ),
          'http://compile.example.com:8000',
        );
      });

      test('effectiveApiBaseUrl falls back to the platform default', () {
        expect(
          Config.effectiveApiBaseUrl(compileTimeOverride: ''),
          Config.platformDefaultApiBaseUrl,
        );
        expect(
          Config.effectiveApiBaseUrl(compileTimeOverride: ''),
          'http://localhost:8000',
        );
      });

      test('ignores a runtime override (storage is not consulted)', () {
        SharedPreferences.setMockInitialValues({
          Config.apiBaseUrlPrefsKey: 'http://runtime.example.com:1234',
        });
        expect(Config.apiBaseUrl, Config.platformDefaultApiBaseUrl);
      });
    });

    group('resolveApiBaseUrl', () {
      test('returns the SharedPreferences override when one is set', () async {
        SharedPreferences.setMockInitialValues({
          Config.apiBaseUrlPrefsKey: 'http://runtime.example.com:1234',
        });

        expect(
          await Config.resolveApiBaseUrl(),
          'http://runtime.example.com:1234',
        );
      });

      test(
        'falls back to the platform default when no override is set',
        () async {
          expect(
            await Config.resolveApiBaseUrl(),
            Config.platformDefaultApiBaseUrl,
          );
          expect(await Config.resolveApiBaseUrl(), 'http://localhost:8000');
        },
      );

      test(
        'falls back to the platform default for an empty override',
        () async {
          SharedPreferences.setMockInitialValues({
            Config.apiBaseUrlPrefsKey: '',
          });

          expect(
            await Config.resolveApiBaseUrl(),
            Config.platformDefaultApiBaseUrl,
          );
        },
      );

      test('compile-time override wins over a stored override', () async {
        SharedPreferences.setMockInitialValues({
          Config.apiBaseUrlPrefsKey: 'http://stored.example.com:1234',
        });

        expect(
          await Config.resolveApiBaseUrl(
            compileTimeOverride: 'http://compile.example.com:8000',
          ),
          'http://compile.example.com:8000',
        );
      });

      test('ignores a stored value that is not a valid base URL', () async {
        SharedPreferences.setMockInitialValues({
          Config.apiBaseUrlPrefsKey: 'ftp://bad',
        });

        expect(
          await Config.resolveApiBaseUrl(),
          Config.platformDefaultApiBaseUrl,
        );
      });
    });

    group('isValidApiBaseUrl', () {
      test('rejects a URL with embedded credentials', () {
        expect(
          Config.isValidApiBaseUrl('http://user:pass@example.com'),
          isFalse,
        );
      });

      test('accepts a well-formed http URL', () {
        expect(Config.isValidApiBaseUrl('http://example.com'), isTrue);
      });
    });

    group('validateStartupApiBaseUrl', () {
      test('throws StateError for a non-empty invalid override', () {
        expect(
          () => Config.validateStartupApiBaseUrl(
            compileTimeOverride: 'ftp://bad',
          ),
          throwsA(isA<StateError>()),
        );
      });

      test('does nothing for an empty override', () {
        expect(
          () => Config.validateStartupApiBaseUrl(compileTimeOverride: ''),
          returnsNormally,
        );
      });
    });

    group('storage failures', () {
      setUp(() {
        SharedPreferencesStorePlatform.instance = ThrowingPrefsStore();
        SharedPreferences.resetStatic();
      });

      tearDown(() {
        // Restore a working in-memory store for the remaining tests.
        SharedPreferences.setMockInitialValues({});
      });

      test('resolveApiBaseUrl falls back to the platform default', () async {
        expect(
          await Config.resolveApiBaseUrl(),
          Config.platformDefaultApiBaseUrl,
        );
      });

      test('clearApiBaseUrl does not throw', () async {
        await expectLater(Config.clearApiBaseUrl(), completes);
      });

      test('setApiBaseUrl throws StateError', () async {
        await expectLater(
          Config.setApiBaseUrl('http://saved.example.com:9000'),
          throwsA(isA<StateError>()),
        );
      });
    });

    group('setApiBaseUrl', () {
      test(
        'persists the value and resolveApiBaseUrl then returns it',
        () async {
          await Config.setApiBaseUrl('http://saved.example.com:9000');

          expect(
            await Config.resolveApiBaseUrl(),
            'http://saved.example.com:9000',
          );
          final prefs = await SharedPreferences.getInstance();
          expect(
            prefs.getString(Config.apiBaseUrlPrefsKey),
            'http://saved.example.com:9000',
          );
        },
      );

      test('throws ArgumentError for an invalid URL', () async {
        await expectLater(
          Config.setApiBaseUrl('not a url'),
          throwsA(isA<ArgumentError>()),
        );
        await expectLater(
          Config.setApiBaseUrl('ftp://x'),
          throwsA(isA<ArgumentError>()),
        );
      });

      test('does not persist an invalid URL', () async {
        await expectLater(
          Config.setApiBaseUrl('not a url'),
          throwsA(isA<ArgumentError>()),
        );

        expect(
          await Config.resolveApiBaseUrl(),
          Config.platformDefaultApiBaseUrl,
        );
      });

      test('null clears the override', () async {
        await Config.setApiBaseUrl('http://saved.example.com:9000');
        await Config.setApiBaseUrl(null);

        expect(
          await Config.resolveApiBaseUrl(),
          Config.platformDefaultApiBaseUrl,
        );
      });

      test('empty string clears the override', () async {
        await Config.setApiBaseUrl('http://saved.example.com:9000');
        await Config.setApiBaseUrl('');

        expect(
          await Config.resolveApiBaseUrl(),
          Config.platformDefaultApiBaseUrl,
        );
      });
    });

    group('clearApiBaseUrl', () {
      test('removes the override so the default is used again', () async {
        await Config.setApiBaseUrl('http://saved.example.com:9000');
        await Config.clearApiBaseUrl();

        expect(
          await Config.resolveApiBaseUrl(),
          Config.platformDefaultApiBaseUrl,
        );
        final prefs = await SharedPreferences.getInstance();
        expect(prefs.getString(Config.apiBaseUrlPrefsKey), isNull);
      });

      test('is a no-op when no override is stored', () async {
        await Config.clearApiBaseUrl();

        expect(
          await Config.resolveApiBaseUrl(),
          Config.platformDefaultApiBaseUrl,
        );
      });
    });
  });
}
