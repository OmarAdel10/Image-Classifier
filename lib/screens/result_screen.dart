import 'dart:io';

import 'package:flutter/material.dart';
import 'package:cross_file/cross_file.dart';

import 'package:image_classifier/models/prediction.dart';
import 'package:image_classifier/models/model_info.dart';
import 'package:image_classifier/widgets/prediction_bars.dart';
import 'package:image_classifier/widgets/model_info_card.dart';

/// Result screen showing detailed classification results.
class ResultScreen extends StatelessWidget {
  const ResultScreen({
    super.key,
    required this.imageFile,
    required this.response,
    required this.modelInfo,
  });

  final XFile imageFile;
  final PredictionResponse response;
  final ModelInfo modelInfo;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Classification Result'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image preview
            _ImagePreview(imageFile: imageFile),
            const SizedBox(height: 24),

            // Prediction bars
            const Text(
              'Confidence Distribution',
              style: TextStyle(fontWeight: FontWeight.w600, fontSize: 18),
            ),
            const SizedBox(height: 12),
            PredictionBars(predictions: response.predictions, maxBars: 10),
            const SizedBox(height: 24),

            // Latency info
            Card(
              color: Theme.of(context).colorScheme.primaryContainer,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    _StatItem(
                      label: 'Latency',
                      value: '${response.latencyMs.toStringAsFixed(1)} ms',
                      icon: Icons.speed,
                    ),
                    _StatItem(
                      label: 'Model',
                      value: response.model,
                      icon: Icons.memory,
                    ),
                    _StatItem(
                      label: 'Classes',
                      value: '${response.predictions.length}',
                      icon: Icons.format_list_numbered,
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Model info
            ModelInfoCard(modelInfo: modelInfo),
            const SizedBox(height: 24),

            // Classify another button
            FilledButton.icon(
              onPressed: () => Navigator.of(context).pop(),
              icon: const Icon(Icons.add_photo_alternate),
              label: const Text('Classify Another Image'),
              style: FilledButton.styleFrom(
                padding: const EdgeInsets.symmetric(vertical: 16),
                textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ImagePreview extends StatelessWidget {
  const _ImagePreview({required this.imageFile});

  final XFile imageFile;

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 4,
      clipBehavior: Clip.antiAlias,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: AspectRatio(
        aspectRatio: 1,
        child: Image.file(
          File(imageFile.path),
          fit: BoxFit.cover,
        ),
      ),
    );
  }
}

class _StatItem extends StatelessWidget {
  const _StatItem({
    required this.label,
    required this.value,
    required this.icon,
  });

  final String label;
  final String value;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Icon(icon, size: 28, color: Theme.of(context).colorScheme.primary),
        const SizedBox(height: 8),
        Text(
          value,
          style: Theme.of(context).textTheme.titleLarge?.copyWith(
                fontWeight: FontWeight.bold,
              ),
        ),
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
        ),
      ],
    );
  }
}