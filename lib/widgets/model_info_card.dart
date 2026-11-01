import 'package:flutter/material.dart';

import 'package:image_classifier/models/model_info.dart';

/// Card displaying model metadata.
class ModelInfoCard extends StatelessWidget {
  const ModelInfoCard({super.key, required this.modelInfo});

  final ModelInfo modelInfo;

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Model Information',
              style: Theme.of(context).textTheme.titleLarge
                  ?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 16),
            _InfoRow(label: 'Model', value: modelInfo.modelName),
            _InfoRow(label: 'Input Size', value: modelInfo.inputSize),
            _InfoRow(
              label: 'Classes',
              value: '${modelInfo.classes.length} classes',
            ),
            _InfoRow(label: 'Device', value: modelInfo.device.toUpperCase()),
            _InfoRow(
              label: 'Last Trained',
              value: modelInfo.trainedAt != null
                  ? _formatDateTime(modelInfo.trainedAt!)
                  : 'Unknown',
            ),
          ],
        ),
      ),
    );
  }

  String _formatDateTime(DateTime dateTime) {
    // Storage is UTC (IC-58); display converts to the device's local time
    // so "Last Trained" reads as the user's wall-clock, not UTC.
    final local = dateTime.toLocal();
    return '${local.year}-${local.month.toString().padLeft(2, '0')}-${local.day.toString().padLeft(2, '0')} '
        '${local.hour.toString().padLeft(2, '0')}:${local.minute.toString().padLeft(2, '0')}';
  }
}

class _InfoRow extends StatelessWidget {
  const _InfoRow({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 100,
            child: Text(
              label,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: Theme.of(context).textTheme.bodyMedium
                  ?.copyWith(fontWeight: FontWeight.w600),
            ),
          ),
        ],
      ),
    );
  }
}
