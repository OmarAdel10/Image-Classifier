import 'dart:io';

import 'package:cross_file/cross_file.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:image_classifier/models/model_info.dart';
import 'package:image_classifier/models/prediction.dart';
import 'package:image_classifier/providers/prediction_provider.dart';
import 'package:image_classifier/screens/result_screen.dart';
import 'package:image_classifier/services/api_client.dart';
import 'package:image_classifier/services/image_service.dart';
import 'package:image_classifier/widgets/drop_zone.dart';
import 'package:image_classifier/widgets/loading_overlay.dart';

/// Home screen with drop zone for image classification.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  XFile? _selectedImage;
  File? _compressedImage;
  ModelInfo? _modelInfo;
  bool _navigated = false;
  bool _isFetchingModelInfo = false;

  @override
  Widget build(BuildContext context) {
    return Consumer<PredictionProvider>(
      builder: (context, provider, _) {
        // Handle state transitions
        _handleProviderState(provider);

        return Scaffold(
          appBar: AppBar(
            title: const Text('Image Classifier'),
            centerTitle: true,
            actions: [
              if (provider.state is PredictionStateResult)
                IconButton(
                  icon: const Icon(Icons.clear),
                  onPressed: _clearPrediction,
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
                    DropZone(
                      key: const Key('drop_zone'),
                      onImageSelected: _onImageSelected,
                      onClassify: _onClassify,
                      isBusy: provider.state is PredictionStateLoading,
                    ),
                    const SizedBox(height: 24),
                    if (provider.state is PredictionStateResult)
                      _ResultPreview(
                        response:
                            (provider.state as PredictionStateResult).response,
                        onClear: _clearPrediction,
                        onViewDetails: _onViewDetails,
                      ),
                    if (_isFetchingModelInfo)
                      const Center(
                        child: Padding(
                          padding: EdgeInsets.all(16),
                          child: CircularProgressIndicator(),
                        ),
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

  void _handleProviderState(PredictionProvider provider) {
    final state = provider.state;

    if (state is PredictionStateResult && !_navigated) {
      _navigated = true;
      _pushResultScreen(state.response);
    } else if (state is PredictionStateError) {
      _showErrorSnackBar(state.message);
    } else if (state is PredictionStateIdle) {
      _navigated = false;
    }
  }

  Future<void> _onImageSelected(XFile imageFile) async {
    setState(() {
      _selectedImage = imageFile;
      _compressedImage = null;
      _modelInfo = null;
      _isFetchingModelInfo = true;
    });

    // Validate and compress the image
    try {
      final imageService = context.read<ImageService>();
      final compressed = await imageService.validateAndCompress(
        File(imageFile.path),
      );
      if (!mounted) return;
      setState(() {
        _compressedImage = compressed;
      });
    } catch (e) {
      if (!mounted) return;
      _showErrorSnackBar('Invalid image: $e');
      setState(() {
        _selectedImage = null;
        _isFetchingModelInfo = false;
      });
      return;
    }

    // Fetch model info
    try {
      final apiClient = context.read<ApiClient>();
      final modelInfo = await apiClient.getModelInfo();
      if (!mounted) return;
      setState(() {
        _modelInfo = modelInfo;
        _isFetchingModelInfo = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _modelInfo = const ModelInfo(
          modelName: 'Unknown',
          inputSize: 'Unknown',
          classes: [],
          device: 'Unknown',
          trainedAt: null,
        );
        _isFetchingModelInfo = false;
      });
    }
  }

  void _onClassify() {
    if (_compressedImage == null) return;

    final provider = context.read<PredictionProvider>();
    // Convert File to XFile for the provider
    final xFile = XFile(_compressedImage!.path);
    provider.predict(xFile);
  }

  void _clearPrediction() {
    setState(() {
      _selectedImage = null;
      _compressedImage = null;
      _navigated = false;
    });
    context.read<PredictionProvider>().clear();
  }

  void _onViewDetails(PredictionResponse response) {
    if (_selectedImage == null || _modelInfo == null) return;

    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => ResultScreen(
          imageFile: _selectedImage!,
          response: response,
          modelInfo: _modelInfo!,
        ),
      ),
    );
  }

  void _pushResultScreen(PredictionResponse response) {
    if (_selectedImage == null || _modelInfo == null) return;

    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => ResultScreen(
          imageFile: _selectedImage!,
          response: response,
          modelInfo: _modelInfo!,
        ),
      ),
    );
  }

  void _showErrorSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Theme.of(context).colorScheme.error,
        action: SnackBarAction(
          label: 'Retry',
          textColor: Theme.of(context).colorScheme.onError,
          onPressed: () {
            if (_selectedImage != null) {
              _onClassify();
            }
          },
        ),
      ),
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
  const _ResultPreview({
    required this.response,
    required this.onClear,
    required this.onViewDetails,
  });

  final PredictionResponse response;
  final VoidCallback onClear;
  final void Function(PredictionResponse) onViewDetails;

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
            // Use Wrap to avoid Row overflow on narrow widths
            Wrap(
              alignment: WrapAlignment.end,
              spacing: 12,
              runSpacing: 12,
              children: [
                OutlinedButton.icon(
                  onPressed: () => onClear(),
                  icon: const Icon(Icons.refresh),
                  label: const Text('Classify Another'),
                ),
                OutlinedButton.icon(
                  onPressed: () => onViewDetails(response),
                  icon: const Icon(Icons.analytics),
                  label: const Text('View Details'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
