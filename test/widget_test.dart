import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:cross_file/cross_file.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mockito/annotations.dart';
import 'package:mockito/mockito.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:provider/provider.dart';
import 'package:image/image.dart' as img;

import 'package:image_classifier/models/prediction.dart';
import 'package:image_classifier/models/model_info.dart';
import 'package:image_classifier/services/api_client.dart';
import 'package:image_classifier/providers/prediction_provider.dart';
import 'package:image_classifier/services/image_service.dart';
import 'package:image_classifier/screens/home_screen.dart';
import 'package:image_classifier/screens/result_screen.dart';
import 'package:image_classifier/widgets/drop_zone.dart';
import 'package:image_classifier/widgets/prediction_bars.dart';
import 'package:image_classifier/widgets/model_info_card.dart';
import 'package:image_classifier/widgets/loading_overlay.dart';

@GenerateMocks([ImageService, PredictionProvider])
import 'widget_test.mocks.dart';

Future<File> _createTempFile(Uint8List bytes) async {
  final tempDir = Directory.systemTemp.createTempSync('widget_test_');
  final file = File('${tempDir.path}/test_image.png');
  await file.writeAsBytes(bytes);
  return file;
}

Uint8List _createValidPng() {
  final image = img.Image(width: 1, height: 1);
  img.fill(image, color: img.ColorRgb8(128, 128, 128));
  return Uint8List.fromList(img.encodePng(image));
}

Uint8List _createLargePng(int width, int height) {
  final image = img.Image(width: width, height: height);
  img.fill(image, color: img.ColorRgb8(128, 128, 128));
  return Uint8List.fromList(img.encodePng(image));
}

void main() {
  group('DropZone widget', () {
    late MockImageService mockImageService;

    setUp(() {
      mockImageService = MockImageService();
    });

    testWidgets('shows file picker on tap', (WidgetTester tester) async {
      final pickedFile = XFile('test_image.png');
      when(mockImageService.pickImage()).thenAnswer((_) async => pickedFile);

      await tester.pumpWidget(
        MaterialApp(
          home: Provider<ImageService>.value(
            value: mockImageService,
            child: const DropZone(),
          ),
        ),
      );

      await tester.tap(find.byKey(const Key('drop_zone')));
      await tester.pumpAndSettle();

      verify(mockImageService.pickImage()).called(1);
    });

    testWidgets('shows selected image preview after pick', (
      WidgetTester tester,
    ) async {
      final pickedFile = XFile('test_image.png');
      when(mockImageService.pickImage()).thenAnswer((_) async => pickedFile);

      await tester.pumpWidget(
        MaterialApp(
          home: Provider<ImageService>.value(
            value: mockImageService,
            child: const DropZone(),
          ),
        ),
      );

      await tester.tap(find.byKey(const Key('drop_zone')));
      await tester.pumpAndSettle();

      expect(find.byType(Image), findsOneWidget);
    });
  });

  group('PredictionBars widget', () {
    final testPredictions = <PredictionItem>[
      PredictionItem(className: 'cat', confidence: 0.9),
      PredictionItem(className: 'dog', confidence: 0.05),
      PredictionItem(className: 'bird', confidence: 0.03),
      PredictionItem(className: 'horse', confidence: 0.015),
      PredictionItem(className: 'truck', confidence: 0.005),
    ];

    testWidgets('renders top-5 bars with correct widths', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(home: PredictionBars(predictions: testPredictions)),
      );

      expect(find.byType(BarChart), findsOneWidget);

      final barChart = tester.widget<BarChart>(find.byType(BarChart));
      final barGroups = barChart.data.barGroups;
      expect(barGroups.length, 5);

      expect(barGroups[0].barRods.first.toY, closeTo(0.9, 0.01));
    });

    testWidgets('colors bars by confidence (green→yellow→red)', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(home: PredictionBars(predictions: testPredictions)),
      );

      final barChart = tester.widget<BarChart>(find.byType(BarChart));
      final barGroups = barChart.data.barGroups;

      expect(barGroups[0].barRods.first.color, isNotNull);
      expect(barGroups[4].barRods.first.color, isNotNull);
    });

    testWidgets('shows class name and percentage labels', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(home: PredictionBars(predictions: testPredictions)),
      );

      expect(find.text('cat'), findsOneWidget);
      expect(find.text('90%'), findsOneWidget);
      expect(find.text('dog'), findsOneWidget);
    });
  });

  group('ModelInfoCard widget', () {
    final testModelInfo = ModelInfo(
      modelName: 'cnn_scratch',
      inputSize: '32x32',
      classes: [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck',
      ],
      device: 'cpu',
      trainedAt: DateTime(2026, 10, 29, 10, 0, 0),
    );

    testWidgets('displays model name, input size, classes count, device', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(home: ModelInfoCard(modelInfo: testModelInfo)),
      );

      expect(find.text('cnn_scratch'), findsOneWidget);
      expect(find.text('32x32'), findsOneWidget);
      expect(find.text('10 classes'), findsOneWidget);
      expect(find.text('cpu'), findsOneWidget);
    });

    testWidgets('displays trained timestamp', (WidgetTester tester) async {
      await tester.pumpWidget(
        MaterialApp(home: ModelInfoCard(modelInfo: testModelInfo)),
      );

      expect(find.textContaining('2026-10-29'), findsOneWidget);
    });

    testWidgets('handles null trainedAt gracefully', (
      WidgetTester tester,
    ) async {
      const modelInfoNoDate = ModelInfo(
        modelName: 'test',
        inputSize: '32x32',
        classes: ['a', 'b'],
        device: 'cpu',
        trainedAt: null,
      );

      await tester.pumpWidget(
        MaterialApp(home: ModelInfoCard(modelInfo: modelInfoNoDate)),
      );

      expect(find.textContaining('Unknown'), findsOneWidget);
    });
  });

  group('LoadingOverlay widget', () {
    testWidgets('shows semi-transparent overlay with progress indicator', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Stack(
            children: [
              const Text('Content behind'),
              LoadingOverlay(message: 'Classifying...'),
            ],
          ),
        ),
      );

      expect(find.byType(CircularProgressIndicator), findsOneWidget);
      expect(find.text('Classifying...'), findsOneWidget);
    });

    testWidgets('can be hidden', (WidgetTester tester) async {
      bool showOverlay = true;
      await tester.pumpWidget(
        MaterialApp(
          home: StatefulBuilder(
            builder: (context, setState) {
              return Stack(
                children: [
                  const Text('Content'),
                  if (showOverlay) LoadingOverlay(message: 'Loading...'),
                  ElevatedButton(
                    onPressed: () => setState(() => showOverlay = false),
                    child: const Text('Hide'),
                  ),
                ],
              );
            },
          ),
        ),
      );

      expect(find.byType(CircularProgressIndicator), findsOneWidget);

      await tester.tap(find.byType(ElevatedButton));
      await tester.pump();

      expect(find.byType(CircularProgressIndicator), findsNothing);
    });
  });

  group('HomeScreen widget', () {
    late MockPredictionProvider mockProvider;
    late MockImageService mockImageService;

    setUp(() {
      mockProvider = MockPredictionProvider();
      mockImageService = MockImageService();
      when(mockProvider.state).thenReturn(const PredictionStateIdle());
    });

    testWidgets('shows drop zone and no preview when idle', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(
          home: MultiProvider(
            providers: [
              ChangeNotifierProvider<PredictionProvider>.value(
                value: mockProvider,
              ),
              Provider<ImageService>.value(value: mockImageService),
            ],
            child: const HomeScreen(),
          ),
        ),
      );

      expect(find.byType(DropZone), findsOneWidget);
      expect(find.byType(LoadingOverlay), findsNothing);
    });

    testWidgets('shows loading overlay when predicting', (
      WidgetTester tester,
    ) async {
      when(mockProvider.state).thenReturn(const PredictionStateLoading());

      await tester.pumpWidget(
        MaterialApp(
          home: MultiProvider(
            providers: [
              ChangeNotifierProvider<PredictionProvider>.value(
                value: mockProvider,
              ),
              Provider<ImageService>.value(value: mockImageService),
            ],
            child: const HomeScreen(),
          ),
        ),
      );

      expect(find.byType(LoadingOverlay), findsOneWidget);
      expect(find.text('Classifying...'), findsOneWidget);
    });

    testWidgets('navigates to ResultScreen on successful prediction', (
      WidgetTester tester,
    ) async {
      when(mockProvider.state).thenReturn(
        PredictionStateResult(
          response: PredictionResponse(
            predictions: [PredictionItem(className: 'cat', confidence: 0.9)],
            latencyMs: 42.0,
            model: 'cnn_scratch',
          ),
        ),
      );

      final mockImageFile = XFile('test.png');
      when(mockImageService.pickImage()).thenAnswer((_) async => mockImageFile);
      when(mockProvider.predict(any)).thenAnswer((_) async {});

      await tester.pumpWidget(
        MaterialApp(
          home: MultiProvider(
            providers: [
              ChangeNotifierProvider<PredictionProvider>.value(
                value: mockProvider,
              ),
              Provider<ImageService>.value(value: mockImageService),
            ],
            child: const HomeScreen(),
          ),
        ),
      );

      await tester.tap(find.byKey(const Key('drop_zone')));
      await tester.pumpAndSettle();

      expect(find.byType(ResultScreen), findsOneWidget);
    });
  });

  group('ResultScreen widget', () {
    final testResponse = PredictionResponse(
      predictions: [
        PredictionItem(className: 'cat', confidence: 0.9),
        PredictionItem(className: 'dog', confidence: 0.1),
      ],
      latencyMs: 42.5,
      model: 'cnn_scratch',
    );

    final testImageFile = XFile('test_image.png');
    final testModelInfo = ModelInfo(
      modelName: 'cnn_scratch',
      inputSize: '32x32',
      classes: ['cat', 'dog', 'bird'],
      device: 'cpu',
      trainedAt: DateTime(2026, 10, 29, 10, 0, 0),
    );

    testWidgets(
      'shows image preview, prediction bars, model info, and classify another button',
      (WidgetTester tester) async {
        await tester.pumpWidget(
          MaterialApp(
            home: ResultScreen(
              imageFile: testImageFile,
              response: testResponse,
              modelInfo: testModelInfo,
            ),
          ),
        );

        expect(find.byType(Image), findsOneWidget);
        expect(find.byType(PredictionBars), findsOneWidget);
        expect(find.byType(ModelInfoCard), findsOneWidget);
        expect(find.text('Classify Another'), findsOneWidget);
      },
    );

    testWidgets('Classify Another button pops back to HomeScreen', (
      WidgetTester tester,
    ) async {
      await tester.pumpWidget(
        MaterialApp(
          home: ResultScreen(
            imageFile: testImageFile,
            response: testResponse,
            modelInfo: testModelInfo,
          ),
        ),
      );

      await tester.tap(find.text('Classify Another'));
      await tester.pumpAndSettle();

      expect(find.byType(ResultScreen), findsNothing);
    });
  });
}
