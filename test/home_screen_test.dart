import 'dart:io';
import 'dart:typed_data';

import 'package:cross_file/cross_file.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image_classifier/providers/prediction_provider.dart';
import 'package:image_classifier/screens/home_screen.dart';
import 'package:image_classifier/screens/settings_screen.dart';
import 'package:image_classifier/services/api_client.dart';
import 'package:image_classifier/services/image_service.dart';
import 'package:mockito/mockito.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'widget_test.mocks.dart';

File _createTempImageFile() {
  final tempDir = Directory.systemTemp.createTempSync('home_screen_test_');
  final file = File('${tempDir.path}/test_image.png');
  // Minimal valid 1x1 PNG.
  file.writeAsBytesSync([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
    0x00, 0x00, 0x00, 0x0D, // IHDR chunk length
    0x49, 0x48, 0x44, 0x52, // IHDR
    0x00, 0x00, 0x00, 0x01, // width: 1
    0x00, 0x00, 0x00, 0x01, // height: 1
    0x08,
    0x02,
    0x00,
    0x00,
    0x00, // bit depth, color type, compression, filter, interlace
    0x90, 0x77, 0x53, 0xDE, // CRC
    0x00, 0x00, 0x00, 0x0C, // IDAT chunk length
    0x49, 0x44, 0x41, 0x54, // IDAT
    0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x03, 0x01, 0x01, 0x00,
    0x18, 0xDD, 0x8D, 0xB4, // compressed data + CRC
    0x00, 0x00, 0x00, 0x00, // IEND chunk length
    0x49, 0x45, 0x4E, 0x44, // IEND
    0xAE, 0x42, 0x60, 0x82, // CRC
  ]);
  return file;
}

Widget _buildHomeScreen({
  required http.Client httpClient,
  required ImageService imageService,
  ThemeData? theme,
}) {
  final apiClient = ApiClient(
    client: httpClient,
    // A generous budget: a short timeout races the mocked failures under
    // load, which made the error-path tests flaky.
    timeout: const Duration(seconds: 5),
  );
  return MultiProvider(
    providers: [
      Provider<ImageService>.value(value: imageService),
      Provider<ApiClient>.value(value: apiClient),
      ChangeNotifierProvider<PredictionProvider>(
        create: (_) => PredictionProvider(apiClient: apiClient),
      ),
    ],
    child: MaterialApp(theme: theme, home: const HomeScreen()),
  );
}

MockClient _offlineClient() {
  return MockClient((request) async {
    throw SocketException('Connection refused');
  });
}

/// An [XFile] whose bytes are available without touching the file system, so
/// the drop-zone preview does not leave a pending real-I/O future behind.
class _InMemoryXFile extends XFile {
  _InMemoryXFile(super.path, this._bytes);

  final Uint8List _bytes;

  @override
  Future<Uint8List> readAsBytes() async => _bytes;
}

/// Taps the drop zone and lets the image flow complete.
Future<void> _tapDropZoneAndSettle(WidgetTester tester) async {
  await tester.tap(find.byKey(const Key('drop_zone')).first);
  await tester.pumpAndSettle();
}

void main() {
  group('HomeScreen', () {
    setUp(() {
      // SettingsScreen resolves the runtime override when it is opened.
      SharedPreferences.setMockInitialValues({});
    });

    testWidgets('renders Scaffold with AppBar and body', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        _buildHomeScreen(
          httpClient: MockClient((request) async => http.Response('{}', 200)),
          imageService: MockImageService(),
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
        _buildHomeScreen(
          httpClient: MockClient((request) async => http.Response('{}', 200)),
          imageService: MockImageService(),
          theme: ThemeData(
            useMaterial3: true,
            colorScheme: ColorScheme.fromSeed(
              seedColor: Colors.deepPurple,
              brightness: Brightness.light,
            ),
          ),
        ),
      );
      expect(find.byType(Scaffold), findsOneWidget);
    });

    testWidgets('renders correctly in dark theme', (WidgetTester tester) async {
      await tester.pumpWidget(
        _buildHomeScreen(
          httpClient: MockClient((request) async => http.Response('{}', 200)),
          imageService: MockImageService(),
          theme: ThemeData(
            useMaterial3: true,
            colorScheme: ColorScheme.fromSeed(
              seedColor: Colors.deepPurple,
              brightness: Brightness.dark,
            ),
          ),
        ),
      );
      expect(find.byType(Scaffold), findsOneWidget);
    });

    testWidgets('opens the backend settings screen from the gear icon', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        _buildHomeScreen(
          httpClient: MockClient((request) async => http.Response('{}', 200)),
          imageService: MockImageService(),
        ),
      );
      await tester.pumpAndSettle();

      await tester.tap(find.byTooltip('Backend settings'));
      await tester.pumpAndSettle();

      expect(find.byType(SettingsScreen), findsOneWidget);
    });

    testWidgets('shows a snackbar when model info cannot be loaded', (
      WidgetTester tester,
    ) async {
      final imageFile = _createTempImageFile();
      final pickedFile = _InMemoryXFile(
        imageFile.path,
        imageFile.readAsBytesSync(),
      );
      final imageService = MockImageService();
      when(imageService.pickImage()).thenAnswer((_) async => pickedFile);
      when(imageService.validateAndCompress(any))
          .thenAnswer((_) async => imageFile);

      final mockClient = MockClient((request) async {
        if (request.url.path == '/model-info') {
          throw SocketException('Connection refused');
        }
        return http.Response('{}', 200);
      });

      await tester.pumpWidget(
        _buildHomeScreen(httpClient: mockClient, imageService: imageService),
      );
      await tester.pumpAndSettle();

      await _tapDropZoneAndSettle(tester);

      expect(find.textContaining('Could not load model info'), findsOneWidget);
      expect(find.textContaining('Cannot reach'), findsOneWidget);
    });

    testWidgets('shows an image error snackbar without a retry action', (
      WidgetTester tester,
    ) async {
      final imageFile = _createTempImageFile();
      final pickedFile = _InMemoryXFile(
        imageFile.path,
        imageFile.readAsBytesSync(),
      );
      final imageService = MockImageService();
      when(imageService.pickImage()).thenAnswer((_) async => pickedFile);
      when(imageService.validateAndCompress(any))
          .thenThrow(StateError('bad image'));

      await tester.pumpWidget(
        _buildHomeScreen(
          httpClient: MockClient((request) async => http.Response('{}', 200)),
          imageService: imageService,
        ),
      );
      await tester.pumpAndSettle();

      await _tapDropZoneAndSettle(tester);

      expect(find.textContaining('Invalid image:'), findsOneWidget);
      expect(find.text('Retry'), findsNothing);
      final snackBar = tester.widget<SnackBar>(find.byType(SnackBar));
      expect(snackBar.action, isNull);
    });

    testWidgets('does not leak the raw text of an image failure', (
      WidgetTester tester,
    ) async {
      final imageFile = _createTempImageFile();
      final pickedFile = _InMemoryXFile(
        imageFile.path,
        imageFile.readAsBytesSync(),
      );
      final imageService = MockImageService();
      when(imageService.pickImage()).thenAnswer((_) async => pickedFile);
      when(imageService.validateAndCompress(any))
          .thenThrow(StateError('bad image'));

      await tester.pumpWidget(
        _buildHomeScreen(
          httpClient: MockClient((request) async => http.Response('{}', 200)),
          imageService: imageService,
        ),
      );
      await tester.pumpAndSettle();

      await _tapDropZoneAndSettle(tester);

      // The fallback text is pinned exactly: a wrong sentence would still
      // satisfy the negative assertions below.
      expect(
        find.text('Invalid image: the selected file could not be read.'),
        findsOneWidget,
      );
      expect(find.textContaining('Bad state:'), findsNothing);
      expect(find.textContaining('bad image'), findsNothing);
      expect(find.textContaining('StateError'), findsNothing);
    });

    testWidgets('names the same reason for any other image failure', (
      WidgetTester tester,
    ) async {
      final imageFile = _createTempImageFile();
      final pickedFile = _InMemoryXFile(
        imageFile.path,
        imageFile.readAsBytesSync(),
      );
      final imageService = MockImageService();
      when(imageService.pickImage()).thenAnswer((_) async => pickedFile);
      when(imageService.validateAndCompress(any)).thenThrow(Exception('boom'));

      await tester.pumpWidget(
        _buildHomeScreen(
          httpClient: MockClient((request) async => http.Response('{}', 200)),
          imageService: imageService,
        ),
      );
      await tester.pumpAndSettle();

      await _tapDropZoneAndSettle(tester);

      expect(
        find.text('Invalid image: the selected file could not be read.'),
        findsOneWidget,
      );
      expect(find.textContaining('boom'), findsNothing);
      expect(find.textContaining('Exception'), findsNothing);
    });

    testWidgets('shows the reason why an image was rejected', (
      WidgetTester tester,
    ) async {
      final imageFile = _createTempImageFile();
      final pickedFile = _InMemoryXFile(
        imageFile.path,
        imageFile.readAsBytesSync(),
      );
      final imageService = MockImageService();
      when(imageService.pickImage()).thenAnswer((_) async => pickedFile);
      when(imageService.validateAndCompress(any)).thenThrow(
        const ImageValidationException(
          'only PNG and JPEG images are supported',
        ),
      );

      await tester.pumpWidget(
        _buildHomeScreen(
          httpClient: MockClient((request) async => http.Response('{}', 200)),
          imageService: imageService,
        ),
      );
      await tester.pumpAndSettle();

      await _tapDropZoneAndSettle(tester);

      expect(
        find.textContaining('only PNG and JPEG images are supported'),
        findsOneWidget,
      );
      expect(find.textContaining('ImageValidationException'), findsNothing);
    });

    testWidgets('does not leak the raw text of a model info failure', (
      WidgetTester tester,
    ) async {
      final imageFile = _createTempImageFile();
      final pickedFile = _InMemoryXFile(
        imageFile.path,
        imageFile.readAsBytesSync(),
      );
      final imageService = MockImageService();
      when(imageService.pickImage()).thenAnswer((_) async => pickedFile);
      when(imageService.validateAndCompress(any))
          .thenAnswer((_) async => imageFile);

      final mockClient = MockClient((request) async {
        if (request.url.path == '/model-info') {
          throw StateError('backend exploded');
        }
        return http.Response('{}', 200);
      });

      await tester.pumpWidget(
        _buildHomeScreen(httpClient: mockClient, imageService: imageService),
      );
      await tester.pumpAndSettle();

      await _tapDropZoneAndSettle(tester);

      expect(find.textContaining('Could not load model info'), findsOneWidget);
      // The client names the failure class with a fixed message, so the raw
      // text of the failure never reaches the screen.
      expect(find.textContaining('Cannot reach'), findsOneWidget);
      expect(find.textContaining('Bad state:'), findsNothing);
      expect(find.textContaining('backend exploded'), findsNothing);
      expect(find.textContaining('StateError'), findsNothing);
    });

    testWidgets(
      'replaces a previous image error snackbar instead of queueing',
      (WidgetTester tester) async {
        final imageFile = _createTempImageFile();
        final pickedFile = _InMemoryXFile(
          imageFile.path,
          imageFile.readAsBytesSync(),
        );
        var validations = 0;
        final imageService = MockImageService();
        when(imageService.pickImage()).thenAnswer((_) async => pickedFile);
        when(imageService.validateAndCompress(any)).thenAnswer((_) async {
          validations++;
          // A rejection with a reason the user may read: the two messages
          // have to differ, so the second snackbar can be told apart from the
          // first one.
          throw ImageValidationException(
            validations == 1 ? 'first image failure' : 'second image failure',
          );
        });

        await tester.pumpWidget(
          _buildHomeScreen(
            httpClient: MockClient((request) async => http.Response('{}', 200)),
            imageService: imageService,
          ),
        );
        await tester.pumpAndSettle();

        await _tapDropZoneAndSettle(tester);
        await _tapDropZoneAndSettle(tester);

        // The second image error must REPLACE the first snackbar
        // (hideCurrentSnackBar), not queue behind it.
        expect(find.textContaining('second image failure'), findsOneWidget);
        expect(find.textContaining('first image failure'), findsNothing);
        expect(find.byType(SnackBar), findsOneWidget);
      },
    );

    testWidgets('prediction errors offer a Settings action that navigates', (
      WidgetTester tester,
    ) async {
      final imageFile = _createTempImageFile();

      await tester.pumpWidget(
        _buildHomeScreen(
          httpClient: _offlineClient(),
          imageService: MockImageService(),
        ),
      );
      await tester.pumpAndSettle();

      final provider = tester
          .element(find.byType(HomeScreen))
          .read<PredictionProvider>();
      // The multipart upload reads the temp file with real I/O, so run it
      // outside the fake-async zone.
      await tester.runAsync(() async {
        await provider.predict(XFile(imageFile.path));
      });
      await tester.pumpAndSettle();

      expect(find.textContaining('Cannot reach'), findsOneWidget);
      final snackBar = tester.widget<SnackBar>(find.byType(SnackBar));
      expect(snackBar.action?.label, 'Settings');

      await tester.tap(find.text('Settings'));
      await tester.pumpAndSettle();

      expect(find.byType(SettingsScreen), findsOneWidget);
    });

    testWidgets('deduplicates the same prediction error across rebuilds', (
      WidgetTester tester,
    ) async {
      final imageFile = _createTempImageFile();

      await tester.pumpWidget(
        _buildHomeScreen(
          httpClient: _offlineClient(),
          imageService: MockImageService(),
        ),
      );
      await tester.pumpAndSettle();

      final provider = tester
          .element(find.byType(HomeScreen))
          .read<PredictionProvider>();
      final image = XFile(imageFile.path);

      // Trigger the same failing prediction twice before pumping so both
      // errors carry the same message.
      await tester.runAsync(() async {
        await Future.wait([provider.predict(image), provider.predict(image)]);
      });
      await tester.pumpAndSettle();

      expect(find.byType(SnackBar), findsOneWidget);

      // A rebuild while the same error is still active must not queue a
      // duplicate snackbar.
      provider.notifyListeners();
      await tester.pump();
      expect(find.byType(SnackBar), findsOneWidget);

      tester
          .state<ScaffoldMessengerState>(find.byType(ScaffoldMessenger))
          .hideCurrentSnackBar();
      await tester.pumpAndSettle();

      expect(find.byType(SnackBar), findsNothing);
    });

    testWidgets('replaces a different prediction error instead of queueing', (
      WidgetTester tester,
    ) async {
      final imageFile = _createTempImageFile();
      var predictCalls = 0;
      final mockClient = MockClient((request) async {
        if (request.url.path == '/model-info') {
          return http.Response(
            '{"model": "cnn_scratch", "classes": ["cat", "dog"]}',
            200,
          );
        }
        if (predictCalls == 0) {
          predictCalls++;
          throw const SocketException('first failure');
        }
        // The backend answered with an error status, so the banner reports
        // that status instead of blaming the connection.
        return http.Response('internal failure', 500);
      });

      await tester.pumpWidget(
        _buildHomeScreen(
          httpClient: mockClient,
          imageService: MockImageService(),
        ),
      );
      await tester.pumpAndSettle();

      final provider = tester
          .element(find.byType(HomeScreen))
          .read<PredictionProvider>();
      final image = XFile(imageFile.path);

      await tester.runAsync(() async {
        await provider.predict(image);
      });
      await tester.pumpAndSettle();

      expect(find.textContaining('Cannot reach'), findsOneWidget);
      expect(find.byType(SnackBar), findsOneWidget);

      // A second, different error must REPLACE the first snackbar
      // (hideCurrentSnackBar), not queue behind it.
      await tester.runAsync(() async {
        await provider.predict(image);
      });
      await tester.pumpAndSettle();

      expect(
        find.textContaining('Request failed with status 500'),
        findsOneWidget,
      );
      expect(find.textContaining('Cannot reach'), findsNothing);
      expect(find.byType(SnackBar), findsOneWidget);
    });
  });
}
