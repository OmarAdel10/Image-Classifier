import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:image_classifier/providers/prediction_provider.dart';
import 'package:image_classifier/screens/home_screen.dart';
import 'package:image_classifier/services/api_client.dart';
import 'package:image_classifier/services/image_service.dart';
import 'package:http/http.dart' as http;

void main() {
  group('HomeScreen', () {
    testWidgets('renders Scaffold with AppBar and body', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MultiProvider(
          providers: [
            Provider<ImageService>(create: (_) => ImageService()),
            Provider<ApiClient>(
              create: (_) => ApiClient(client: http.Client()),
              dispose: (_, client) => client.client.close(),
            ),
            ChangeNotifierProvider<PredictionProvider>(
              create: (ctx) =>
                  PredictionProvider(apiClient: ctx.read<ApiClient>()),
            ),
          ],
          child: const MaterialApp(home: HomeScreen()),
        ),
      );

      // Verify Scaffold exists
      expect(find.byType(Scaffold), findsOneWidget);

      // Verify AppBar with centered title
      expect(find.byType(AppBar), findsOneWidget);
      expect(find.text('Image Classifier'), findsOneWidget);
    });

    testWidgets('renders correctly in light theme', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MultiProvider(
          providers: [
            Provider<ImageService>(create: (_) => ImageService()),
            Provider<ApiClient>(
              create: (_) => ApiClient(client: http.Client()),
              dispose: (_, client) => client.client.close(),
            ),
            ChangeNotifierProvider<PredictionProvider>(
              create: (ctx) =>
                  PredictionProvider(apiClient: ctx.read<ApiClient>()),
            ),
          ],
          child: MaterialApp(
            theme: ThemeData(
              useMaterial3: true,
              colorScheme: ColorScheme.fromSeed(
                seedColor: Colors.deepPurple,
                brightness: Brightness.light,
              ),
            ),
            home: const HomeScreen(),
          ),
        ),
      );
      expect(find.byType(Scaffold), findsOneWidget);
    });

    testWidgets('renders correctly in dark theme', (WidgetTester tester) async {
      await tester.pumpWidget(
        MultiProvider(
          providers: [
            Provider<ImageService>(create: (_) => ImageService()),
            Provider<ApiClient>(
              create: (_) => ApiClient(client: http.Client()),
              dispose: (_, client) => client.client.close(),
            ),
            ChangeNotifierProvider<PredictionProvider>(
              create: (ctx) =>
                  PredictionProvider(apiClient: ctx.read<ApiClient>()),
            ),
          ],
          child: MaterialApp(
            theme: ThemeData(
              useMaterial3: true,
              colorScheme: ColorScheme.fromSeed(
                seedColor: Colors.deepPurple,
                brightness: Brightness.dark,
              ),
            ),
            home: const HomeScreen(),
          ),
        ),
      );
      expect(find.byType(Scaffold), findsOneWidget);
    });
  });
}
