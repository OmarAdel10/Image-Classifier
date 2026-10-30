import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:image_classifier/main.dart';

void main() {
  group('MyApp', () {
    testWidgets('configures Material 3 theme correctly', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MyApp());

      final MaterialApp app = tester.widget(find.byType(MaterialApp));

      // Verify Material 3 is enabled via theme
      expect(app.theme!.useMaterial3, isTrue);

      // Verify light theme
      expect(app.theme, isNotNull);
      final ColorScheme lightScheme = app.theme!.colorScheme;
      expect(lightScheme.brightness, Brightness.light);
      // deepPurple is the seed color
      expect(lightScheme.primary, isNotNull);

      // Verify dark theme
      expect(app.darkTheme, isNotNull);
      expect(app.darkTheme!.useMaterial3, isTrue);
      final ColorScheme darkScheme = app.darkTheme!.colorScheme;
      expect(darkScheme.brightness, Brightness.dark);
      expect(darkScheme.primary, isNotNull);

      // Verify theme mode follows system
      expect(app.themeMode, ThemeMode.system);
    });

    testWidgets('renders HomeScreen as home', (WidgetTester tester) async {
      await tester.pumpWidget(const MyApp());

      // Verify HomeScreen is the home widget
      expect(find.byType(HomeScreen), findsOneWidget);
    });
  });
}
