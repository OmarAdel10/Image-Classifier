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
        MaterialApp(theme: ThemeData.light(), home: const HomeScreen()),
      );
      expect(find.byType(Scaffold), findsOneWidget);
    });

    testWidgets('renders correctly in dark theme', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(theme: ThemeData.dark(), home: const HomeScreen()),
      );
      expect(find.byType(Scaffold), findsOneWidget);
    });
  });
}
