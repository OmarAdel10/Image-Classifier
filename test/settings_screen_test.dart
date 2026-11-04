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

import 'support/delayed_prefs_store.dart';
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
        find.textContaining('Could not clear the stored URL'),
        findsOneWidget,
      );
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

        expect(
          find.textContaining(
            'Could not test ${Config.platformDefaultApiBaseUrl}',
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
  });
}
