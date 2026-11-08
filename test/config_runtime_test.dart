import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:image_classifier/config.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:shared_preferences_platform_interface/shared_preferences_platform_interface.dart';

import 'support/false_prefs_store.dart';
import 'support/false_removing_prefs_store.dart';
import 'support/refusing_write_prefs_store.dart';
import 'support/reload_throwing_prefs_store.dart';
import 'support/throwing_write_back_prefs_store.dart';
import 'support/throwing_prefs_store.dart';
import 'support/throwing_removing_prefs_store.dart';
import 'support/throwing_write_prefs_store.dart';

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
      test(
        'matches the non-Android platform default',
        skip: Platform.isAndroid ? 'platform default differs on Android' : null,
        () {
          expect(Config.platformDefaultApiBaseUrl, 'http://localhost:8000');
        },
      );
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

    group('compile-time override precedence', () {
      test('uses the platform default when no override is set', () async {
        expect(Config.compileTimeApiBaseUrl, isEmpty);
        expect(
          await Config.resolveApiBaseUrl(),
          Config.platformDefaultApiBaseUrl,
        );
        expect(await Config.resolveApiBaseUrl(), 'http://localhost:8000');
      });

      test('uses a non-empty override, normalized', () async {
        expect(
          await Config.resolveApiBaseUrl(
            compileTimeOverride: '  http://compile.example.com:8000/  ',
          ),
          'http://compile.example.com:8000',
        );
      });

      test('ignores a runtime override under a pin', () async {
        SharedPreferences.setMockInitialValues({
          Config.apiBaseUrlPrefsKey: 'http://runtime.example.com:1234',
        });
        expect(
          await Config.resolveApiBaseUrl(
            compileTimeOverride: 'http://compile.example.com:8000',
          ),
          'http://compile.example.com:8000',
        );
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

      test('rejects a URL with a query string', () {
        expect(Config.isValidApiBaseUrl('http://example.com/?x=1'), isFalse);
      });

      test('rejects a URL with a fragment', () {
        expect(Config.isValidApiBaseUrl('http://example.com/#frag'), isFalse);
      });

      test('rejects a URL with internal whitespace', () {
        expect(Config.isValidApiBaseUrl('http://exa mple.com'), isFalse);
      });

      test('rejects a URL whose port is out of range', () {
        expect(Config.isValidApiBaseUrl('http://example.com:99999'), isFalse);
      });

      test('accepts the highest valid port', () {
        expect(Config.isValidApiBaseUrl('http://example.com:65535'), isTrue);
      });

      test('rejects a port one above the maximum', () {
        expect(Config.isValidApiBaseUrl('http://example.com:65536'), isFalse);
      });

      test('rejects a URL with an invalid percent escape in the host', () {
        expect(Config.isValidApiBaseUrl('http://exa%zzmple.com'), isFalse);
      });

      test('rejects an empty URL', () {
        expect(Config.isValidApiBaseUrl(''), isFalse);
      });

      test('rejects a whitespace-only URL', () {
        expect(Config.isValidApiBaseUrl('   '), isFalse);
      });

      test('rejects a URL whose port is not a number', () {
        expect(Config.isValidApiBaseUrl('http://host:notaport'), isFalse);
      });

      test('rejects a URL with port 0', () {
        expect(Config.isValidApiBaseUrl('http://example.com:0'), isFalse);
      });

      test('rejects a URL with a negative port', () {
        // The Uri parser keeps a negative port, so the port range check has
        // to reject it explicitly.
        expect(Config.isValidApiBaseUrl('http://example.com:-1'), isFalse);
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

      test('validates the normalized override', () {
        expect(
          () => Config.validateStartupApiBaseUrl(
            compileTimeOverride: '  http://ok.example.com:8000/  ',
          ),
          returnsNormally,
        );
      });

      test('throws after normalization for an invalid override', () {
        expect(
          () => Config.validateStartupApiBaseUrl(
            compileTimeOverride: '  ftp://bad/  ',
          ),
          throwsA(isA<StateError>()),
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

      test('clearApiBaseUrl reports false when storage fails', () async {
        expect(await Config.clearApiBaseUrl(), isFalse);
      });

      test('setApiBaseUrl reports a storage failure as StateError', () async {
        await expectLater(
          Config.setApiBaseUrl('http://saved.example.com:9000'),
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              allOf(
                contains('Could not store the API base URL'),
                isNot(contains('Refusing to store a runtime API base URL')),
              ),
            ),
          ),
        );
      });

      test('setApiBaseUrl refuses a pinned build', () async {
        await expectLater(
          Config.setApiBaseUrl(
            'http://saved.example.com:9000',
            compileTimeOverride: 'http://compile.example.com:8000',
          ),
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              contains('Refusing to store a runtime API base URL'),
            ),
          ),
        );
      });
    });

    group('setApiBaseUrl when the store write reports failure', () {
      setUp(() {
        SharedPreferencesStorePlatform.instance = FalseReturningPrefsStore();
        SharedPreferences.resetStatic();
      });

      tearDown(() {
        // Restore a working in-memory store for the remaining tests.
        SharedPreferences.setMockInitialValues({});
      });

      test('throws StateError with the store-failure message', () async {
        await expectLater(
          Config.setApiBaseUrl('http://saved.example.com:9000'),
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              'Could not store the API base URL.',
            ),
          ),
        );
      });

      test('clearApiBaseUrl reports false when removal is refused', () async {
        expect(await Config.clearApiBaseUrl(), isFalse);
      });

      test('setApiBaseUrl(null) reports a refused removal', () async {
        await expectLater(
          Config.setApiBaseUrl(null),
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              'Could not clear the stored API base URL.',
            ),
          ),
        );
      });

      test('a refused write leaves no effective override behind', () async {
        await expectLater(
          Config.setApiBaseUrl('http://saved.example.com:9000'),
          throwsA(isA<StateError>()),
        );

        // shared_preferences updates its in-memory cache before the platform
        // write completes, so the refused value has to be rolled back.
        final prefs = await SharedPreferences.getInstance();
        expect(prefs.getString(Config.apiBaseUrlPrefsKey), isNull);
        expect(
          await Config.resolveApiBaseUrl(),
          Config.platformDefaultApiBaseUrl,
        );
      });
    });

    group('setApiBaseUrl when the store refuses the write', () {
      setUp(() {
        SharedPreferencesStorePlatform.instance = RefusingWritePrefsStore({
          'flutter.${Config.apiBaseUrlPrefsKey}':
              'http://stored.example.com:1234',
        });
        SharedPreferences.resetStatic();
      });

      tearDown(() {
        // Restore a working in-memory store for the remaining tests.
        SharedPreferences.setMockInitialValues({});
      });

      test('keeps the previously stored URL in effect', () async {
        await expectLater(
          Config.setApiBaseUrl('http://new.example.com:9000'),
          throwsA(isA<StateError>()),
        );

        // The refused write must not throw away the backend address that is
        // still stored: the app keeps using the one that is persisted.
        expect(
          await Config.resolveApiBaseUrl(),
          'http://stored.example.com:1234',
        );
      });
    });

    group('setApiBaseUrl when the store write throws', () {
      setUp(() {
        SharedPreferencesStorePlatform.instance = ThrowingWritePrefsStore();
        SharedPreferences.resetStatic();
      });

      tearDown(() {
        // Restore a working in-memory store for the remaining tests.
        SharedPreferences.setMockInitialValues({});
      });

      test('reports the failure as a storage error', () async {
        await expectLater(
          Config.setApiBaseUrl('http://saved.example.com:9000'),
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              'Could not store the API base URL.',
            ),
          ),
        );
      });

      test('leaves no effective override behind', () async {
        await expectLater(
          Config.setApiBaseUrl('http://saved.example.com:9000'),
          throwsA(isA<StateError>()),
        );

        final prefs = await SharedPreferences.getInstance();
        expect(prefs.getString(Config.apiBaseUrlPrefsKey), isNull);
        expect(
          await Config.resolveApiBaseUrl(),
          Config.platformDefaultApiBaseUrl,
        );
      });
    });

    group('setApiBaseUrl when the rollback cannot read the store', () {
      setUp(() {
        SharedPreferencesStorePlatform.instance = ReloadThrowingPrefsStore({
          'flutter.${Config.apiBaseUrlPrefsKey}':
              'http://stored.example.com:1234',
        });
        SharedPreferences.resetStatic();
      });

      tearDown(() {
        // Restore a working in-memory store for the remaining tests.
        SharedPreferences.setMockInitialValues({});
      });

      test('reports the storage error and keeps the cached URL', () async {
        await expectLater(
          Config.setApiBaseUrl('http://new.example.com:9000'),
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              'Could not store the API base URL.',
            ),
          ),
        );

        // The rollback could not read the store, so the address that is
        // still cached stays in use instead of being lost.
        expect(
          await Config.resolveApiBaseUrl(),
          'http://stored.example.com:1234',
        );
      });
    });

    group('setApiBaseUrl when the rollback fails and nothing was stored', () {
      setUp(() {
        SharedPreferencesStorePlatform.instance = ReloadThrowingPrefsStore({});
        SharedPreferences.resetStatic();
      });

      tearDown(() {
        // Restore a working in-memory store for the remaining tests.
        SharedPreferences.setMockInitialValues({});
      });

      test('drops the refused URL', () async {
        await expectLater(
          Config.setApiBaseUrl('http://new.example.com:9000'),
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              'Could not store the API base URL.',
            ),
          ),
        );

        // Nothing was stored before, so the refused URL must not stay in
        // effect: the rollback has to remove it from the in-memory cache.
        final prefs = await SharedPreferences.getInstance();
        expect(prefs.getString(Config.apiBaseUrlPrefsKey), isNull);
        expect(
          await Config.resolveApiBaseUrl(),
          Config.platformDefaultApiBaseUrl,
        );
      });
    });

    group('setApiBaseUrl when the rollback cannot write the value back', () {
      tearDown(() {
        // Restore a working in-memory store for the remaining tests.
        SharedPreferences.setMockInitialValues({});
      });

      void installStore(Map<String, Object> values) {
        SharedPreferencesStorePlatform.instance = ThrowingWriteBackPrefsStore(
          values,
        );
        SharedPreferences.resetStatic();
      }

      test('drops the refused URL when nothing was stored', () async {
        installStore({});

        await expectLater(
          Config.setApiBaseUrl('http://new.example.com:9000'),
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              'Could not store the API base URL.',
            ),
          ),
        );

        // The write-back failed as well, so the rollback cannot restore
        // anything. The refused URL must still not stay in effect.
        expect(
          await Config.resolveApiBaseUrl(),
          Config.platformDefaultApiBaseUrl,
        );
      });

      test('keeps the previously stored URL', () async {
        installStore({
          'flutter.${Config.apiBaseUrlPrefsKey}':
              'http://stored.example.com:1234',
        });

        await expectLater(
          Config.setApiBaseUrl('http://new.example.com:9000'),
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              'Could not store the API base URL.',
            ),
          ),
        );

        // The working backend address that was configured earlier survives a
        // refused save.
        expect(
          await Config.resolveApiBaseUrl(),
          'http://stored.example.com:1234',
        );
      });
    });

    group('clearApiBaseUrl when the platform refuses the removal', () {
      setUp(() {
        SharedPreferencesStorePlatform.instance = FalseRemovingPrefsStore({
          'flutter.${Config.apiBaseUrlPrefsKey}':
              'http://stored.example.com:1234',
        });
        SharedPreferences.resetStatic();
      });

      tearDown(() {
        // Restore a working in-memory store for the remaining tests.
        SharedPreferences.setMockInitialValues({});
      });

      test('reports the failure', () async {
        expect(await Config.clearApiBaseUrl(), isFalse);
      });

      test('the stored URL is not active after the refused removal', () async {
        expect(await Config.clearApiBaseUrl(), isFalse);

        expect(
          await Config.resolveApiBaseUrl(),
          Config.platformDefaultApiBaseUrl,
        );
      });
    });

    group('clearApiBaseUrl when the platform throws while removing', () {
      setUp(() {
        SharedPreferencesStorePlatform.instance = ThrowingRemovingPrefsStore({
          'flutter.${Config.apiBaseUrlPrefsKey}':
              'http://stored.example.com:1234',
        });
        SharedPreferences.resetStatic();
      });

      tearDown(() {
        // Restore a working in-memory store for the remaining tests.
        SharedPreferences.setMockInitialValues({});
      });

      test('reports the failure without throwing', () async {
        expect(await Config.clearApiBaseUrl(), isFalse);
      });

      test('the stored URL is not active after the failed removal', () async {
        expect(await Config.clearApiBaseUrl(), isFalse);

        expect(
          await Config.resolveApiBaseUrl(),
          Config.platformDefaultApiBaseUrl,
        );
      });
    });

    group('clearApiBaseUrl return value', () {
      test('reports true when the override is removed', () async {
        SharedPreferences.setMockInitialValues({
          Config.apiBaseUrlPrefsKey: 'http://stored.example.com:1234',
        });

        expect(await Config.clearApiBaseUrl(), isTrue);
        expect(
          await Config.resolveApiBaseUrl(),
          isNot('http://stored.example.com:1234'),
        );
      });

      test('reports true when no override was stored', () async {
        SharedPreferences.setMockInitialValues({});

        expect(await Config.clearApiBaseUrl(), isTrue);
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

      test('stores the normalized URL', () async {
        await Config.setApiBaseUrl('  http://saved.example.com:9000/  ');

        final prefs = await SharedPreferences.getInstance();
        expect(
          prefs.getString(Config.apiBaseUrlPrefsKey),
          'http://saved.example.com:9000',
        );
      });

      test(
        'a stored value of another type does not escape as a TypeError',
        () async {
          // shared_preferences casts the cached value to a string, so a value
          // of another type must not escape the documented error contract.
          SharedPreferences.setMockInitialValues({
            Config.apiBaseUrlPrefsKey: 42,
          });
          SharedPreferences.resetStatic();

          await Config.setApiBaseUrl('http://saved.example.com:9000');

          expect(
            await Config.resolveApiBaseUrl(),
            'http://saved.example.com:9000',
          );
        },
      );

      test('a pinned build stores nothing', () async {
        await expectLater(
          Config.setApiBaseUrl(
            'http://saved.example.com:9000',
            compileTimeOverride: 'http://compile.example.com:8000',
          ),
          throwsA(isA<StateError>()),
        );

        // A refused write must not leave a dormant redirect behind.
        final prefs = await SharedPreferences.getInstance();
        expect(prefs.getString(Config.apiBaseUrlPrefsKey), isNull);
        expect(
          await Config.resolveApiBaseUrl(),
          Config.platformDefaultApiBaseUrl,
        );
      });

      test('throws ArgumentError for an invalid URL', () async {
        await expectLater(
          Config.setApiBaseUrl('not a url'),
          throwsA(
            isA<ArgumentError>().having(
              (e) => e.message,
              'message',
              contains('must be a well-formed http/https URL'),
            ),
          ),
        );
        await expectLater(
          Config.setApiBaseUrl('ftp://x'),
          throwsA(
            isA<ArgumentError>().having(
              (e) => e.message,
              'message',
              contains('must be a well-formed http/https URL'),
            ),
          ),
        );
      });

      test('reports an invalid URL before the pinned-build refusal', () async {
        await expectLater(
          Config.setApiBaseUrl(
            'not a url',
            compileTimeOverride: 'http://compile.example.com:8000',
          ),
          throwsA(
            isA<ArgumentError>().having(
              (e) => e.message,
              'message',
              contains('must be a well-formed http/https URL'),
            ),
          ),
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

      test('null clears the override even in a pinned build', () async {
        await Config.setApiBaseUrl('http://saved.example.com:9000');

        // The clear path runs before the pinned check: a pinned build can
        // never create a runtime override, so removing one is always safe.
        await Config.setApiBaseUrl(
          null,
          compileTimeOverride: 'http://compile.example.com:8000',
        );

        final prefs = await SharedPreferences.getInstance();
        expect(prefs.getString(Config.apiBaseUrlPrefsKey), isNull);
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
