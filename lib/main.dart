import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:provider/provider.dart';

import 'package:image_classifier/config.dart';
import 'package:image_classifier/providers/prediction_provider.dart';
import 'package:image_classifier/screens/home_screen.dart';
import 'package:image_classifier/services/api_client.dart';
import 'package:image_classifier/services/image_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Fail fast on a malformed API base URL (IC-60): a bad --dart-define
  // must not silently point requests at an attacker-controlled URL.
  if (!Config.isValidApiBaseUrl(Config.apiBaseUrl)) {
    throw StateError(
      'Invalid API_BASE_URL "${Config.apiBaseUrl}": '
      'must be a well-formed http/https URL with a host.',
    );
  }
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]);
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        Provider<ImageService>(create: (_) => ImageService()),
        Provider<ApiClient>(
          create: (_) => ApiClient(client: http.Client()),
          dispose: (_, client) => client.client.close(),
        ),
        ChangeNotifierProvider<PredictionProvider>(
          create: (ctx) => PredictionProvider(apiClient: ctx.read<ApiClient>()),
        ),
      ],
      child: MaterialApp(
        title: 'Image Classifier',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          useMaterial3: true,
          colorScheme: ColorScheme.fromSeed(
            seedColor: Colors.deepPurple,
            brightness: Brightness.light,
          ),
        ),
        darkTheme: ThemeData(
          useMaterial3: true,
          colorScheme: ColorScheme.fromSeed(
            seedColor: Colors.deepPurple,
            brightness: Brightness.dark,
          ),
        ),
        themeMode: ThemeMode.system,
        home: const HomeScreen(),
      ),
    );
  }
}
