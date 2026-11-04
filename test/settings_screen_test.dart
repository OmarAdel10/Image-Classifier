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

    testWidgets('warns that a compile-time override wins over the saved URL', (
      WidgetTester tester,
    ) async {
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

      await tester.tap(_buttonWithLabel('Save'));
      await tester.pumpAndSettle();

      expect(find.textContaining('overrides it'), findsOneWidget);
      expect(find.textContaining('compile.example.com:8000'), findsOneWidget);
    });

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
  });
}
