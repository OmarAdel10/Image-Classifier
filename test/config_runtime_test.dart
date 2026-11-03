import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:image_classifier/config.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  group('Config runtime API base URL', () {
    setUp(() {
      // Seed an empty mock store so every test starts without an override.
      SharedPreferences.setMockInitialValues({});
    });

    group('platformDefaultApiBaseUrl', () {
      test('is http://localhost:8000 on Linux', () {
        expect(Platform.isLinux, isTrue);
        expect(Config.platformDefaultApiBaseUrl, 'http://localhost:8000');
      });
    });

    group('apiBaseUrl', () {
      test('returns platform default when no compile-time override', () {
        expect(Config.compileTimeApiBaseUrl, isEmpty);
        expect(Config.apiBaseUrl, Config.platformDefaultApiBaseUrl);
        expect(Config.apiBaseUrl, 'http://localhost:8000');
      });

      test('honours the compile-time override over the platform default', () {
        // The compile-time value is a `const` and cannot be changed at
        // runtime, so assert the documented priority relationship:
        // compile-time override when non-empty, platform default otherwise.
        expect(
          Config.apiBaseUrl,
          Config.compileTimeApiBaseUrl.isNotEmpty
              ? Config.compileTimeApiBaseUrl
              : Config.platformDefaultApiBaseUrl,
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
