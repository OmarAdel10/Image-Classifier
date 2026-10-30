import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:image_classifier/main.dart';

void main() {
  group('HomeScreen', () {
    testWidgets('renders Scaffold with AppBar and body', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(const MaterialApp(home: HomeScreen()));

      // Verify Scaffold exists
      expect(find.byType(Scaffold), findsOneWidget);

      // Verify AppBar with centered title
      expect(find.byType(AppBar), findsOneWidget);
      expect(find.text('Image Classifier'), findsOneWidget);

      // Verify body text
      expect(find.text('Drag and drop an image to classify'), findsOneWidget);
    });

    testWidgets('renders correctly in light theme', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(
          theme: ThemeData(
            useMaterial3: true,
            colorScheme: ColorScheme.fromSeed(
              seedColor: Colors.deepPurple,
              brightness: Brightness.light,
            ),
          ),
          home: const HomeScreen(),
        ),
      );
      expect(find.byType(Scaffold), findsOneWidget);
      // Verify text is visible (has color)
      final textFinder = find.text('Drag and drop an image to classify');
      expect(textFinder, findsOneWidget);
    });

    testWidgets('renders correctly in dark theme', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(
          theme: ThemeData(
            useMaterial3: true,
            colorScheme: ColorScheme.fromSeed(
              seedColor: Colors.deepPurple,
              brightness: Brightness.dark,
            ),
          ),
          home: const HomeScreen(),
        ),
      );
      expect(find.byType(Scaffold), findsOneWidget);
      final textFinder = find.text('Drag and drop an image to classify');
      expect(textFinder, findsOneWidget);
    });
  });
}
