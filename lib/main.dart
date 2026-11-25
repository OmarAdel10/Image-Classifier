import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:provider/provider.dart';

import 'package:image_classifier/config.dart';
import 'package:image_classifier/providers/backend_controller.dart';
import 'package:image_classifier/providers/prediction_provider.dart';
import 'package:image_classifier/screens/home_screen.dart';
import 'package:image_classifier/screens/startup_screen.dart';
import 'package:image_classifier/services/api_client.dart';
import 'package:image_classifier/services/image_service.dart';

/// Creates the shared HTTP client for the app lifecycle.
///
/// This ensures the client is disposed when the app closes.
final http.Client _sharedHttpClient = http.Client();

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  // Fail fast on a malformed API base URL (IC-60): a bad --dart-define
  // must not silently point requests at an attacker-controlled URL.
  Config.validateStartupApiBaseUrl();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]);
  runApp(const MyApp());
}

/// Wrapper that disposes the shared HTTP client when the app is removed.
class _HttpClientDisposer extends StatefulWidget {
  final Widget child;

  const _HttpClientDisposer({required this.child});

  @override
  State<_HttpClientDisposer> createState() => _HttpClientDisposerState();
}

class _HttpClientDisposerState extends State<_HttpClientDisposer> {
  @override
  void dispose() {
    _sharedHttpClient.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => widget.child;
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        Provider<ImageService>(create: (_) => ImageService()),
        Provider<ApiClient>(
          create: (_) => ApiClient(client: _sharedHttpClient),
        ),
        ChangeNotifierProvider<BackendController>(
          create: (ctx) => BackendController(apiClient: ctx.read<ApiClient>()),
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
        home: _HttpClientDisposer(child: const _AppInitializer()),
      ),
    );
  }
}

/// Initializes the backend and shows the startup screen, then transitions to HomeScreen.
class _AppInitializer extends StatefulWidget {
  const _AppInitializer();

  @override
  State<_AppInitializer> createState() => _AppInitializerState();
}

class _AppInitializerState extends State<_AppInitializer> {
  String? _initError;
  bool _initialized = false;

  @override
  void initState() {
    super.initState();
    _initializeBackend();
  }

  Future<void> _initializeBackend() async {
    try {
      final controller = context.read<BackendController>();
      await controller.initialize();
      if (mounted) {
        setState(() => _initialized = true);
      }
    } catch (e) {
      if (mounted) {
        setState(() => _initError = e.toString());
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    if (_initialized) {
      return const HomeScreen();
    }
    return StartupScreen(
      onReady: () => setState(() => _initialized = true),
      error: _initError,
    );
  }
}
