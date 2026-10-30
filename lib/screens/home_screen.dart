import 'package:cross_file/cross_file.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:image_classifier/models/model_info.dart';
import 'package:image_classifier/models/prediction.dart';
import 'package:image_classifier/providers/prediction_provider.dart';
import 'package:image_classifier/screens/result_screen.dart';
import 'package:image_classifier/widgets/drop_zone.dart';
import 'package:image_classifier/widgets/loading_overlay.dart';

/// Home screen with drop zone for image classification.
class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<PredictionProvider>(
      builder: (context, provider, _) {
        return Scaffold(
          appBar: AppBar(
            title: const Text('Image Classifier'),
            centerTitle: true,
            actions: [
              if (provider.state is PredictionStateResult)
                IconButton(
                  icon: const Icon(Icons.clear),
                  onPressed: () => provider.clear(),
                  tooltip: 'Clear prediction',
                ),
            ],
          ),
          body: Stack(
            children: [
              SingleChildScrollView(
                padding: const EdgeInsets.all(24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    const DropZone(),
                    const SizedBox(height: 24),
                    if (provider.state is PredictionStateResult)
                      _ResultPreview(
                        response:
                            (provider.state as PredictionStateResult).response,
                      ),
                  ],
                ),
              ),
              if (provider.state is PredictionStateLoading)
                const LoadingOverlay(message: 'Classifying...'),
            ],
          ),
        );
      },
    );
  }
}

class _PredictionTile extends StatelessWidget {
  const _PredictionTile({required this.prediction});

  final PredictionItem prediction;

  @override
  Widget build(BuildContext context) {
    final confidence = prediction.confidence;
    Color confidenceColor;
    if (confidence >= 0.7) {
      confidenceColor = Colors.green.shade600;
    } else if (confidence >= 0.4) {
      confidenceColor = Colors.orange.shade600;
    } else {
      confidenceColor = Colors.red.shade500;
    }

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Expanded(
            child: Text(
              prediction.className,
              style: const TextStyle(fontWeight: FontWeight.w500),
            ),
          ),
          Container(
            width: 80,
            height: 8,
            decoration: BoxDecoration(
              color: Colors.grey.shade200,
              borderRadius: BorderRadius.circular(4),
            ),
            child: FractionallySizedBox(
              alignment: Alignment.centerLeft,
              widthFactor: prediction.confidence.clamp(0.0, 1.0),
              child: Container(
                decoration: BoxDecoration(
                  color: confidenceColor,
                  borderRadius: BorderRadius.circular(4),
                ),
              ),
            ),
          ),
          const SizedBox(width: 12),
          Text(
            '${(prediction.confidence * 100).toStringAsFixed(1)}%',
            style: TextStyle(
              fontWeight: FontWeight.w600,
              color: confidenceColor,
            ),
          ),
        ],
      ),
    );
  }
}

class _ResultPreview extends StatelessWidget {
  const _ResultPreview({required this.response});

  final PredictionResponse response;

  @override
  Widget build(BuildContext context) {
    final topPrediction = response.predictions.first;

    return Card(
      elevation: 4,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(
                  Icons.check_circle,
                  color: Colors.green.shade600,
                  size: 28,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Classification Complete',
                        style: Theme.of(context).textTheme.titleMedium
                            ?.copyWith(
                              fontWeight: FontWeight.bold,
                              color: Colors.green.shade700,
                            ),
                      ),
                      Text(
                        'Top: ${topPrediction.className} (${(topPrediction.confidence * 100).toStringAsFixed(1)}%)',
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                          color: Theme.of(context).colorScheme.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ),
                ),
                const Spacer(),
                FilledButton.icon(
                  onPressed: () => context.read<PredictionProvider>().clear(),
                  icon: const Icon(Icons.refresh),
                  label: const Text('Classify Another'),
                ),
              ],
            ),
            const SizedBox(height: 16),
            const Text(
              'Top Predictions',
              style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16),
            ),
            const SizedBox(height: 8),
            ...response.predictions
                .take(3)
                .map((pred) => _PredictionTile(prediction: pred)),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: () {
                Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (_) => ResultScreen(
                      imageFile: XFile(''), // placeholder
                      response: response,
                      modelInfo: const ModelInfo(
                        modelName: '',
                        inputSize: '',
                        classes: [],
                        device: '',
                        trainedAt: null,
                      ),
                    ),
                  ),
                );
              },
              icon: const Icon(Icons.analytics),
              label: const Text('View Details'),
            ),
          ],
        ),
      ),
    );
  }
}
