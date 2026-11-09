import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image_classifier/config.dart';
import 'package:image_classifier/screens/settings_screen.dart';
import 'package:image_classifier/services/api_client.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:shared_preferences_platform_interface/shared_preferences_platform_interface.dart';

import 'support/delayed_mutation_prefs_store.dart';
import 'support/delayed_prefs_store.dart';
import 'support/false_prefs_store.dart';
import 'support/false_removing_prefs_store.dart';
import 'support/refusing_write_prefs_store.dart';
import 'support/throwing_prefs_store.dart';

/// A built settings screen together with the [ApiClient] instance it uses.
typedef SettingsHarness = ({Widget widget, ApiClient client});

/// Builds the settings screen with an [ApiClient] backed by [httpClient].
SettingsHarness _buildSettingsScreen(
  http.Client httpClient, {
  String compileTimeOverride = Config.compileTimeApiBaseUrl,
}) {
  final client = ApiClient(client: httpClient);
  return (
    widget: Provider<ApiClient>.value(
      value: client,
      child: MaterialApp(
        home: SettingsScreen(compileTimeOverride: compileTimeOverride),
      ),
    ),
    client: client,
  );
}

/// Finds the button labelled [label], matching the icon-button variants
/// (`FilledButton.icon` / `OutlinedButton.icon`) whose runtime type differs
/// from the base button class.
Finder _buttonWithLabel(String label) {
  return find.ancestor(
    of: find.text(label),
    matching: find.byWidgetPredicate((widget) => widget is ButtonStyleButton),
  );
}

void main() {
  group('SettingsScreen', () {
    setUp(() {
      SharedPreferences.setMockInitialValues({});
    });

    tearDown(() {
      // Undo any ThrowingPrefsStore installed by a test.
      SharedPreferences.setMockInitialValues({});
    });

    testWidgets('renders the title and the three action buttons', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        _buildSettingsScreen(
          MockClient((request) async => http.Response('{}', 200)),
        ).widget,
      );
      await tester.pumpAndSettle();

      expect(find.text('Backend Settings'), findsOneWidget);
      expect(find.text('Test Connection'), findsOneWidget);
      expect(find.text('Save'), findsOneWidget);
      expect(find.text('Reset to Default'), findsOneWidget);
    });

    testWidgets('pre-fills the text field with the platform default', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        _buildSettingsScreen(
          MockClient((request) async => http.Response('{}', 200)),
        ).widget,
      );
      await tester.pumpAndSettle();

      final field = tester.widget<TextField>(find.byType(TextField));
      expect(field.controller?.text, Config.platformDefaultApiBaseUrl);
      expect(field.controller?.text, 'http://localhost:8000');
    });

    testWidgets('pre-fills the text field with the stored override', (
      WidgetTester tester,
    ) async {
      SharedPreferences.setMockInitialValues({
        Config.apiBaseUrlPrefsKey: 'http://stored.example.com:1234',
      });

      await tester.pumpWidget(
        _buildSettingsScreen(
          MockClient((request) async => http.Response('{}', 200)),
        ).widget,
      );
      await tester.pumpAndSettle();

      final field = tester.widget<TextField>(find.byType(TextField));
      expect(field.controller?.text, 'http://stored.example.com:1234');
    });

    testWidgets('keeps text typed while the stored URL resolves', (
      WidgetTester tester,
    ) async {
      // The store read is held open so the text is typed *before* the stored
      // URL arrives: the non-clobber guard in _loadCurrentUrl only matters
      // when the resolve completes after the user has typed.
      final store = DelayedPrefsStore({
        'flutter.${Config.apiBaseUrlPrefsKey}':
            'http://stored.example.com:1234',
      });
      SharedPreferencesStorePlatform.instance = store;
      SharedPreferences.resetStatic();

      await tester.pumpWidget(
        _buildSettingsScreen(
          MockClient((request) async => http.Response('{}', 200)),
        ).widget,
      );
      await tester.enterText(
        find.byType(TextField),
        'http://typed.example.com:9000',
      );
      await tester.pump();

      // The stored URL only now becomes available.
      store.release();
      await tester.pumpAndSettle();

      final field = tester.widget<TextField>(find.byType(TextField));
      expect(field.controller?.text, 'http://typed.example.com:9000');
    });

    testWidgets('shows a validation error and disables Save for a bad URL', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        _buildSettingsScreen(
          MockClient((request) async => http.Response('{}', 200)),
        ).widget,
      );
      await tester.pumpAndSettle();

      await tester.enterText(find.byType(TextField), 'not a url');
      await tester.pump();

      expect(find.textContaining('Not a valid URL'), findsOneWidget);

      final saveButton = tester.widget<FilledButton>(_buttonWithLabel('Save'));
      expect(saveButton.onPressed, isNull);
    });

    testWidgets('shows an empty-field error and disables Save', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        _buildSettingsScreen(
          MockClient((request) async => http.Response('{}', 200)),
        ).widget,
      );
      await tester.pumpAndSettle();

      await tester.enterText(find.byType(TextField), '');
      await tester.pump();

      expect(
        find.text('Enter a URL such as http://localhost:8000'),
        findsOneWidget,
      );

      final saveButton = tester.widget<FilledButton>(_buttonWithLabel('Save'));
      expect(saveButton.onPressed, isNull);
    });

    testWidgets('Save persists the typed URL', (WidgetTester tester) async {
      await tester.pumpWidget(
        _buildSettingsScreen(
          MockClient((request) async => http.Response('{}', 200)),
        ).widget,
      );
      await tester.pumpAndSettle();

      await tester.enterText(
        find.byType(TextField),
        'http://saved.example.com:9000',
      );
      await tester.pump();

      await tester.tap(_buttonWithLabel('Save'));
      await tester.pumpAndSettle();

      expect(await Config.resolveApiBaseUrl(), 'http://saved.example.com:9000');
      expect(
        find.textContaining('Saved. Using http://saved.example.com:9000'),
        findsOneWidget,
      );
    });

    testWidgets('probes the normalized URL', (WidgetTester tester) async {
      final probed = <String>[];
      final harness = _buildSettingsScreen(
        MockClient((request) async {
          probed.add(request.url.toString());
          return http.Response('{}', 200);
        }),
      );
      await tester.pumpWidget(harness.widget);
      await tester.pumpAndSettle();

      // A trailing slash would produce '//health' if it were not stripped.
      await tester.enterText(
        find.byType(TextField),
        'http://saved.example.com:9000/',
      );
      await tester.pump();
      await tester.tap(_buttonWithLabel('Test Connection'));
      await tester.pumpAndSettle();

      expect(probed, ['http://saved.example.com:9000/health']);
      // The banner shows the normalized URL, so a trailing slash cannot hide
      // in the message the user reads.
      expect(
        find.text('Connected to http://saved.example.com:9000'),
        findsOneWidget,
      );
    });

    testWidgets('Save makes the live client use the new URL', (
      WidgetTester tester,
    ) async {
      final harness = _buildSettingsScreen(
        MockClient((request) async => http.Response('{}', 200)),
      );
      await tester.pumpWidget(harness.widget);
      await tester.pumpAndSettle();

      // Resolve (and cache) the default before saving.
      expect(await harness.client.baseUrl, Config.platformDefaultApiBaseUrl);

      await tester.enterText(
        find.byType(TextField),
        'http://saved.example.com:9000',
      );
      await tester.pump();

      await tester.tap(_buttonWithLabel('Save'));
      await tester.pumpAndSettle();

      expect(await harness.client.baseUrl, 'http://saved.example.com:9000');
    });

    testWidgets('Reset to Default makes the live client use the default URL', (
      WidgetTester tester,
    ) async {
      SharedPreferences.setMockInitialValues({
        Config.apiBaseUrlPrefsKey: 'http://stored.example.com:1234',
      });

      final harness = _buildSettingsScreen(
        MockClient((request) async => http.Response('{}', 200)),
      );
      await tester.pumpWidget(harness.widget);
      await tester.pumpAndSettle();

      // Resolve (and cache) the stored URL before resetting.
      expect(await harness.client.baseUrl, 'http://stored.example.com:1234');

      await tester.tap(_buttonWithLabel('Reset to Default'));
      await tester.pumpAndSettle();

      // The client must not keep using the cached (now cleared) URL.
      expect(await harness.client.baseUrl, Config.platformDefaultApiBaseUrl);
    });

    testWidgets('reports a missing ApiClient provider instead of throwing', (
      WidgetTester tester,
    ) async {
      // The shipped app always provides the client, but a missing provider
      // must not escape the handler or leave the Test Connection button
      // spinning forever.
      await tester.pumpWidget(const MaterialApp(home: SettingsScreen()));
      await tester.pumpAndSettle();

      // The status banner grows with the error text, so the buttons can be
      // scrolled out of the lazily built list between the actions.
      Future<void> tapButton(String label) async {
        await tester.scrollUntilVisible(
          find.text(label),
          100,
          scrollable: find.byType(Scrollable).first,
        );
        await tester.tap(_buttonWithLabel(label));
        await tester.pumpAndSettle();
      }

      await tester.enterText(
        find.byType(TextField),
        'http://saved.example.com:9000',
      );
      await tester.pump();

      await tapButton('Test Connection');

      expect(find.textContaining('Could not test'), findsOneWidget);
      expect(find.byType(CircularProgressIndicator), findsNothing);

      await tapButton('Save');

      expect(find.textContaining('Could not save'), findsOneWidget);
      // The generic error arm must name the error that caused it, so a
      // missing provider is not reported as an anonymous failure.
      expect(find.textContaining('Provider<ApiClient>'), findsOneWidget);

      await tapButton('Reset to Default');

      expect(find.textContaining('Could not reset'), findsOneWidget);
      // The reset failure names the error that caused it, like the other
      // failure messages on this screen.
      expect(find.textContaining('Provider<ApiClient>'), findsOneWidget);
    });

    testWidgets('reports a storage refusal instead of claiming success', (
      WidgetTester tester,
    ) async {
      SharedPreferencesStorePlatform.instance = FalseReturningPrefsStore();
      SharedPreferences.resetStatic();

      final harness = _buildSettingsScreen(
        MockClient((request) async => http.Response('{}', 200)),
      );
      await tester.pumpWidget(harness.widget);
      await tester.pumpAndSettle();

      await tester.enterText(
        find.byType(TextField),
        'http://saved.example.com:9000',
      );
      await tester.pump();
      await tester.tap(_buttonWithLabel('Save'));
      await tester.pumpAndSettle();

      expect(find.textContaining('Could not save'), findsOneWidget);
      expect(find.textContaining('Saved. Using'), findsNothing);
      // The message names the cause instead of leaking the raw exception.
      expect(
        find.textContaining('Could not store the API base URL'),
        findsOneWidget,
      );
      expect(find.textContaining('Bad state:'), findsNothing);
      // The refused URL must not be used by the app either.
      expect(await harness.client.baseUrl, Config.platformDefaultApiBaseUrl);

      await tester.scrollUntilVisible(
        find.text('Reset to Default'),
        100,
        scrollable: find.byType(Scrollable).first,
      );
      await tester.tap(_buttonWithLabel('Reset to Default'));
      await tester.pumpAndSettle();

      expect(
        find.textContaining('Could not remove the stored URL'),
        findsOneWidget,
      );
      expect(find.textContaining('still in use'), findsNothing);
    });

    testWidgets('a refused save keeps the stored backend address in use', (
      WidgetTester tester,
    ) async {
      SharedPreferencesStorePlatform.instance = RefusingWritePrefsStore({
        'flutter.${Config.apiBaseUrlPrefsKey}':
            'http://stored.example.com:1234',
      });
      SharedPreferences.resetStatic();

      final harness = _buildSettingsScreen(
        MockClient((request) async => http.Response('{}', 200)),
      );
      await tester.pumpWidget(harness.widget);
      await tester.pumpAndSettle();

      await tester.enterText(
        find.byType(TextField),
        'http://new.example.com:9000',
      );
      await tester.pump();
      await tester.tap(_buttonWithLabel('Save'));
      await tester.pumpAndSettle();

      expect(find.textContaining('Could not save'), findsOneWidget);
      // The refused write must not throw away the backend address that is
      // still stored.
      expect(await harness.client.baseUrl, 'http://stored.example.com:1234');
    });

    testWidgets('a refused save re-resolves the client from the store', (
      WidgetTester tester,
    ) async {
      final store = RefusingWritePrefsStore({
        'flutter.${Config.apiBaseUrlPrefsKey}':
            'http://stored.example.com:1234',
      });
      SharedPreferencesStorePlatform.instance = store;
      SharedPreferences.resetStatic();

      final harness = _buildSettingsScreen(
        MockClient((request) async => http.Response('{}', 200)),
      );
      await tester.pumpWidget(harness.widget);
      await tester.pumpAndSettle();

      // Resolve the URL first, so the client caches it.
      expect(await harness.client.baseUrl, 'http://stored.example.com:1234');

      // The persisted address changes underneath the running app.
      store.setPersisted(
        'flutter.${Config.apiBaseUrlPrefsKey}',
        'http://changed.example.com:9999',
      );

      await tester.enterText(
        find.byType(TextField),
        'http://new.example.com:9000',
      );
      await tester.pump();
      await tester.tap(_buttonWithLabel('Save'));
      await tester.pumpAndSettle();

      expect(find.textContaining('Could not save'), findsOneWidget);
      // The client must drop its cached address and read the persisted one.
      expect(await harness.client.baseUrl, 'http://changed.example.com:9999');
    });

    testWidgets('a refused reset says the stored URL is not active now', (
      WidgetTester tester,
    ) async {
      SharedPreferencesStorePlatform.instance = FalseRemovingPrefsStore({
        'flutter.${Config.apiBaseUrlPrefsKey}':
            'http://stored.example.com:1234',
      });
      SharedPreferences.resetStatic();

      final harness = _buildSettingsScreen(
        MockClient((request) async => http.Response('{}', 200)),
      );
      await tester.pumpWidget(harness.widget);
      await tester.pumpAndSettle();

      await tester.scrollUntilVisible(
        find.text('Reset to Default'),
        100,
        scrollable: find.byType(Scrollable).first,
      );
      await tester.tap(_buttonWithLabel('Reset to Default'));
      await tester.pumpAndSettle();

      // The platform refused the removal, but the in-memory cache is gone, so
      // the stored URL is not in use - the message must not claim otherwise.
      expect(find.textContaining('not active now'), findsOneWidget);
      expect(find.textContaining('still in use'), findsNothing);
      expect(await harness.client.baseUrl, Config.platformDefaultApiBaseUrl);
    });

    testWidgets('pinned build shows the banner and disables Save', (
      WidgetTester tester,
    ) async {
      SharedPreferences.setMockInitialValues({
        Config.apiBaseUrlPrefsKey: 'http://saved.example.com:9000',
      });

      await tester.pumpWidget(
        _buildSettingsScreen(
          MockClient((request) async => http.Response('{}', 200)),
          compileTimeOverride: 'http://compile.example.com:8000',
        ).widget,
      );
      await tester.pumpAndSettle();

      expect(
        find.textContaining('Pinned by --dart-define=API_BASE_URL='),
        findsOneWidget,
      );
      expect(find.textContaining('Save is disabled'), findsOneWidget);
      // The field is pre-filled with the pinned URL it can probe but not save.
      final field = tester.widget<TextField>(find.byType(TextField));
      expect(field.controller?.text, 'http://compile.example.com:8000');

      final saveButton = tester.widget<FilledButton>(_buttonWithLabel('Save'));
      expect(saveButton.onPressed, isNull);
    });

    testWidgets('Reset to Default clears the stored URL while pinned', (
      WidgetTester tester,
    ) async {
      SharedPreferences.setMockInitialValues({
        Config.apiBaseUrlPrefsKey: 'http://stored.example.com:1234',
      });

      await tester.pumpWidget(
        _buildSettingsScreen(
          MockClient((request) async => http.Response('{}', 200)),
          compileTimeOverride: 'http://compile.example.com:8000',
        ).widget,
      );
      await tester.pumpAndSettle();

      // Reset stays enabled while pinned: removing a stored override is
      // always safe, because it can never take effect in this build.
      await tester.tap(_buttonWithLabel('Reset to Default'));
      await tester.pumpAndSettle();

      expect(
        find.textContaining(
          'Cleared the stored URL; --dart-define=API_BASE_URL=',
        ),
        findsOneWidget,
      );
      expect(
        find.textContaining('http://compile.example.com:8000'),
        findsWidgets,
      );

      // The stored override is gone: the field shows the pinned URL and the
      // platform default resolves again once the pin is removed.
      final field = tester.widget<TextField>(find.byType(TextField));
      expect(field.controller?.text, 'http://compile.example.com:8000');
      final prefs = await SharedPreferences.getInstance();
      expect(prefs.getString(Config.apiBaseUrlPrefsKey), isNull);
      expect(
        await Config.resolveApiBaseUrl(),
        Config.platformDefaultApiBaseUrl,
      );
    });

    testWidgets(
      'Save is disabled while pinned and stores nothing when tapped',
      (WidgetTester tester) async {
        await tester.pumpWidget(
          _buildSettingsScreen(
            MockClient((request) async => http.Response('{}', 200)),
            compileTimeOverride: 'http://compile.example.com:8000',
          ).widget,
        );
        await tester.pumpAndSettle();

        await tester.enterText(
          find.byType(TextField),
          'http://saved.example.com:9000',
        );
        await tester.pump();

        // Observable contract of the pinned build: the Save button is
        // disabled, so tapping it stores nothing and reports no success.
        final saveButton = tester.widget<FilledButton>(
          _buttonWithLabel('Save'),
        );
        expect(saveButton.onPressed, isNull);

        await tester.tap(_buttonWithLabel('Save'));
        await tester.pumpAndSettle();

        expect(find.textContaining('Saved. Using'), findsNothing);
        final prefs = await SharedPreferences.getInstance();
        expect(prefs.getString(Config.apiBaseUrlPrefsKey), isNull);
      },
    );

    testWidgets('Save shows an error banner when storage fails', (
      WidgetTester tester,
    ) async {
      SharedPreferencesStorePlatform.instance = ThrowingPrefsStore();
      SharedPreferences.resetStatic();

      await tester.pumpWidget(
        _buildSettingsScreen(
          MockClient((request) async => http.Response('{}', 200)),
        ).widget,
      );
      await tester.pumpAndSettle();

      await tester.enterText(
        find.byType(TextField),
        'http://saved.example.com:9000',
      );
      await tester.pump();

      await tester.tap(_buttonWithLabel('Save'));
      await tester.pumpAndSettle();

      expect(find.textContaining('Could not save'), findsOneWidget);
    });

    testWidgets(
      'Reset to Default clears the override and restores the default',
      (WidgetTester tester) async {
        SharedPreferences.setMockInitialValues({
          Config.apiBaseUrlPrefsKey: 'http://stored.example.com:1234',
        });

        await tester.pumpWidget(
          _buildSettingsScreen(
            MockClient((request) async => http.Response('{}', 200)),
          ).widget,
        );
        await tester.pumpAndSettle();

        await tester.tap(_buttonWithLabel('Reset to Default'));
        await tester.pumpAndSettle();

        final field = tester.widget<TextField>(find.byType(TextField));
        expect(field.controller?.text, Config.platformDefaultApiBaseUrl);
        expect(
          await Config.resolveApiBaseUrl(),
          Config.platformDefaultApiBaseUrl,
        );
        expect(find.textContaining('Reset to default'), findsOneWidget);
      },
    );

    testWidgets('Reset to Default reports failure when storage fails', (
      WidgetTester tester,
    ) async {
      SharedPreferencesStorePlatform.instance = ThrowingPrefsStore();
      SharedPreferences.resetStatic();

      await tester.pumpWidget(
        _buildSettingsScreen(
          MockClient((request) async => http.Response('{}', 200)),
        ).widget,
      );
      await tester.pumpAndSettle();

      await tester.tap(_buttonWithLabel('Reset to Default'));
      await tester.pumpAndSettle();

      expect(
        find.textContaining('Could not remove the stored URL'),
        findsOneWidget,
      );
      expect(find.textContaining('still in use'), findsNothing);
      // The field keeps showing the resolved default; nothing was cleared.
      final field = tester.widget<TextField>(find.byType(TextField));
      expect(field.controller?.text, Config.platformDefaultApiBaseUrl);
    });

    testWidgets(
      'Test Connection shows a failure status for an unreachable URL',
      (WidgetTester tester) async {
        final mockClient = MockClient((request) async {
          throw SocketException('Connection refused');
        });

        await tester.pumpWidget(_buildSettingsScreen(mockClient).widget);
        await tester.pumpAndSettle();

        await tester.tap(_buttonWithLabel('Test Connection'));
        await tester.pump();
        await tester.pumpAndSettle();

        expect(
          find.textContaining(
            'Could not reach ${Config.platformDefaultApiBaseUrl}',
          ),
          findsOneWidget,
        );
      },
    );

    testWidgets('Test Connection shows a success status for a reachable URL', (
      WidgetTester tester,
    ) async {
      final requestedUrls = <String>[];
      final mockClient = MockClient((request) async {
        requestedUrls.add(request.url.toString());
        return http.Response('{"status": "ok"}', 200);
      });

      await tester.pumpWidget(_buildSettingsScreen(mockClient).widget);
      await tester.pumpAndSettle();

      await tester.tap(_buttonWithLabel('Test Connection'));
      await tester.pump();
      await tester.pumpAndSettle();

      expect(
        find.textContaining('Connected to ${Config.platformDefaultApiBaseUrl}'),
        findsOneWidget,
      );
      expect(
        requestedUrls,
        contains('${Config.platformDefaultApiBaseUrl}/health'),
      );

      // Typing a new value clears the banner about the previous one.
      await tester.enterText(
        find.byType(TextField),
        'http://changed.example.com:9000',
      );
      await tester.pump();

      expect(find.textContaining('Connected to'), findsNothing);
    });

    testWidgets('Test Connection probes the URL as typed', (
      WidgetTester tester,
    ) async {
      final requestedUrls = <String>[];
      final mockClient = MockClient((request) async {
        requestedUrls.add(request.url.toString());
        return http.Response('{"status": "ok"}', 200);
      });

      await tester.pumpWidget(_buildSettingsScreen(mockClient).widget);
      await tester.pumpAndSettle();

      await tester.enterText(
        find.byType(TextField),
        'http://typed.example.com:9000',
      );
      await tester.pump();

      await tester.tap(_buttonWithLabel('Test Connection'));
      await tester.pump();
      await tester.pumpAndSettle();

      expect(requestedUrls, contains('http://typed.example.com:9000/health'));
      expect(
        find.textContaining('Connected to http://typed.example.com:9000'),
        findsOneWidget,
      );
    });

    testWidgets('Test Connection reports a reachable but unhealthy backend', (
      WidgetTester tester,
    ) async {
      final mockClient = MockClient(
        (request) async => http.Response('{"status": "degraded"}', 500),
      );

      await tester.pumpWidget(_buildSettingsScreen(mockClient).widget);
      await tester.pumpAndSettle();

      await tester.tap(_buttonWithLabel('Test Connection'));
      await tester.pump();
      await tester.pumpAndSettle();

      expect(find.textContaining('Reached'), findsOneWidget);
      expect(find.textContaining('HTTP 500'), findsOneWidget);
    });

    testWidgets(
      'Test Connection reports an unreachable backend for a client exception',
      (WidgetTester tester) async {
        final mockClient = MockClient((request) async {
          throw http.ClientException('connection closed');
        });

        await tester.pumpWidget(_buildSettingsScreen(mockClient).widget);
        await tester.pumpAndSettle();

        await tester.tap(_buttonWithLabel('Test Connection'));
        await tester.pump();
        await tester.pumpAndSettle();

        // A ClientException is swallowed by healthStatusCode (it reports no
        // status code), so the probe reports the backend as unreachable.
        expect(
          find.textContaining(
            'Could not reach ${Config.platformDefaultApiBaseUrl}',
          ),
          findsOneWidget,
        );
        // The probe finished: the button is re-enabled for another attempt.
        final probeButton = tester.widget<FilledButton>(
          _buttonWithLabel('Test Connection'),
        );
        expect(probeButton.onPressed, isNotNull);
      },
    );

    testWidgets(
      'Test Connection reports unexpected probe failures without crashing',
      (WidgetTester tester) async {
        final mockClient = MockClient((request) async {
          throw Exception('boom');
        });

        await tester.pumpWidget(_buildSettingsScreen(mockClient).widget);
        await tester.pumpAndSettle();

        await tester.tap(_buttonWithLabel('Test Connection'));
        await tester.pump();
        await tester.pumpAndSettle();

        // The probe failed without an answer, so the banner reports that the
        // backend could not be reached. The text of the exception itself stays
        // out of the banner: it can carry internal details.
        expect(
          find.textContaining(
            'Could not reach ${Config.platformDefaultApiBaseUrl}',
          ),
          findsOneWidget,
        );
        expect(find.textContaining('boom'), findsNothing);
        // The probe finished: the button is re-enabled for another attempt.
        final probeButton = tester.widget<FilledButton>(
          _buttonWithLabel('Test Connection'),
        );
        expect(probeButton.onPressed, isNotNull);
      },
    );

    testWidgets('a long address is shortened in the banner', (
      WidgetTester tester,
    ) async {
      final mockClient = MockClient((request) async {
        throw Exception('boom');
      });
      // A valid address can still be far longer than a banner can show.
      final longUrl = 'http://${'a' * 200}.example.com:8000';

      await tester.pumpWidget(_buildSettingsScreen(mockClient).widget);
      await tester.pumpAndSettle();

      await tester.enterText(find.byType(TextField), longUrl);
      await tester.tap(_buttonWithLabel('Test Connection'));
      await tester.pump();
      await tester.pumpAndSettle();

      // The banner stays readable: the address is shortened. The field
      // itself still holds what the user typed, so the assertion looks at
      // the banner only.
      final banner = tester.widget<Text>(
        find.textContaining('Could not reach'),
      );
      expect(banner.data, contains('…'));
      expect(banner.data, isNot(contains(longUrl)));
    });

    testWidgets('a control character in the field is rejected', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        _buildSettingsScreen(
          MockClient((request) async => http.Response('{}', 200)),
        ).widget,
      );
      await tester.pumpAndSettle();

      await tester.enterText(
        find.byType(TextField),
        'http://exa\x01mple.com:8000',
      );
      await tester.pumpAndSettle();

      // The address is refused, so the request is never made with it.
      expect(
        find.text('Not a valid URL — use http:// or https:// with a host'),
        findsOneWidget,
      );
    });

    testWidgets('a control character in the pinned address is not shown', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        _buildSettingsScreen(
          http.Client(),
          compileTimeOverride: 'http://exa\x01mple.com:8000',
        ).widget,
      );
      await tester.pumpAndSettle();

      // The banner carries the address without the control character.
      final banner = tester.widget<Text>(
        find.textContaining('Pinned by --dart-define'),
      );
      expect(banner.data, isNot(contains('\x01')));
      expect(banner.data, contains('http://example.com:8000'));
    });

    testWidgets('a credential in the pinned address is not shown', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        _buildSettingsScreen(
          http.Client(),
          compileTimeOverride: 'http://user:pass@pinned.example.com:8000',
        ).widget,
      );
      await tester.pumpAndSettle();

      final banner = tester.widget<Text>(
        find.textContaining('Pinned by --dart-define'),
      );
      expect(banner.data, isNot(contains('user:pass')));
      expect(banner.data, contains('http://pinned.example.com:8000'));
    });

    testWidgets('a credential variant in the pinned address is not shown', (
      WidgetTester tester,
    ) async {
      // A password that carries an at sign, a scheme in capitals and a
      // leading space all hide a credential from a simple prefix match.
      const variants = <String>[
        'http://user:p@ss@host:8000',
        'HTTP://user:pass@host:8000',
        ' http://user:pass@host:8000',
      ];

      for (final variant in variants) {
        await tester.pumpWidget(
          _buildSettingsScreen(
            http.Client(),
            compileTimeOverride: variant,
          ).widget,
        );
        await tester.pumpAndSettle();

        final banner = tester.widget<Text>(
          find.textContaining('Pinned by --dart-define'),
        );
        expect(banner.data, isNot(contains('user')));
        expect(banner.data, isNot(contains('pass')));
        expect(banner.data, isNot(contains('ss')));
        expect(banner.data, contains('host:8000'));
      }
    });

    testWidgets('a client failure never puts its message in the banner', (
      WidgetTester tester,
    ) async {
      final mockClient = MockClient((request) async {
        throw StateError('secret internal detail');
      });

      await tester.pumpWidget(_buildSettingsScreen(mockClient).widget);
      await tester.pumpAndSettle();

      await tester.tap(_buttonWithLabel('Test Connection'));
      await tester.pump();
      await tester.pumpAndSettle();

      // The text of an unexpected failure stays out of the banner.
      expect(find.textContaining('secret internal detail'), findsNothing);
      expect(find.textContaining('Bad state:'), findsNothing);
    });

    group('unmounted while a request is in flight', () {
      testWidgets(
        'a probe that finishes after the screen is closed is ignored',
        (WidgetTester tester) async {
          final gate = Completer<void>();
          final gatedClient = MockClient((request) async {
            await gate.future;
            return http.Response('{}', 200);
          });

          await tester.pumpWidget(_buildSettingsScreen(gatedClient).widget);
          await tester.pumpAndSettle();

          await tester.tap(_buttonWithLabel('Test Connection'));
          await tester.pump();

          // The screen is closed while the probe is still running.
          await tester.pumpWidget(const SizedBox.shrink());

          gate.complete();
          await tester.pump();
          await tester.pump();

          // The late result must not touch the closed screen.
          expect(tester.takeException(), isNull);
        },
      );

      testWidgets('a probe that fails after the screen is closed is ignored', (
        WidgetTester tester,
      ) async {
        final gate = Completer<void>();
        final gatedClient = MockClient((request) async {
          await gate.future;
          throw Exception('boom');
        });

        await tester.pumpWidget(_buildSettingsScreen(gatedClient).widget);
        await tester.pumpAndSettle();

        await tester.tap(_buttonWithLabel('Test Connection'));
        await tester.pump();

        // The screen is closed while the probe is still running.
        await tester.pumpWidget(const SizedBox.shrink());

        gate.complete();
        await tester.pump();
        await tester.pump();

        // The late failure must not touch the closed screen.
        expect(tester.takeException(), isNull);
      });

      testWidgets(
        'a save that finishes after the screen is closed is ignored',
        (WidgetTester tester) async {
          final store = DelayedMutationPrefsStore({});
          SharedPreferencesStorePlatform.instance = store;
          SharedPreferences.resetStatic();

          await tester.pumpWidget(
            _buildSettingsScreen(
              MockClient((request) async => http.Response('{}', 200)),
            ).widget,
          );
          await tester.pumpAndSettle();

          await tester.enterText(
            find.byType(TextField),
            'http://saved.example.com:9000',
          );
          await tester.pump();
          await tester.tap(_buttonWithLabel('Save'));
          await tester.pump();

          // The screen is closed while the write is still running.
          await tester.pumpWidget(const SizedBox.shrink());

          store.releaseWrite();
          await tester.pump();
          await tester.pump();

          // The late result must not touch the closed screen.
          expect(tester.takeException(), isNull);
        },
      );

      testWidgets(
        'a reset that finishes after the screen is closed is ignored',
        (WidgetTester tester) async {
          final store = DelayedMutationPrefsStore({
            'flutter.${Config.apiBaseUrlPrefsKey}':
                'http://stored.example.com:1234',
          });
          SharedPreferencesStorePlatform.instance = store;
          SharedPreferences.resetStatic();

          await tester.pumpWidget(
            _buildSettingsScreen(
              MockClient((request) async => http.Response('{}', 200)),
            ).widget,
          );
          await tester.pumpAndSettle();

          await tester.tap(_buttonWithLabel('Reset to Default'));
          await tester.pump();

          // The screen is closed while the removal is still running.
          await tester.pumpWidget(const SizedBox.shrink());

          store.releaseRemoval();
          await tester.pumpAndSettle();

          // The late result must not touch the closed screen.
          expect(tester.takeException(), isNull);
        },
      );
    });
  });
}
