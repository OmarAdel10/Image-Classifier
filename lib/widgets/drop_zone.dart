import 'dart:typed_data';

import 'package:cross_file/cross_file.dart';
import 'package:desktop_drop/desktop_drop.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:image_classifier/services/image_service.dart';

/// A drop zone widget that allows users to pick an image via tap or native drag-and-drop.
class DropZone extends StatefulWidget {
  /// Creates a [DropZone] widget.
  ///
  /// [onImageSelected] is called when an image is selected (via tap or drag-and-drop).
  /// [onClassify] is called when the user taps the Classify button.
  /// [isBusy] indicates if a prediction is in progress (disables Classify button).
  const DropZone({
    super.key,
    required this.onImageSelected,
    this.onClassify,
    this.isBusy = false,
  });

  final ValueChanged<XFile> onImageSelected;
  final VoidCallback? onClassify;
  final bool isBusy;

  @override
  State<DropZone> createState() => _DropZoneState();
}

class _DropZoneState extends State<DropZone> {
  XFile? _selectedImage;
  bool _isDragOver = false;

  Future<void> _pickImage() async {
    final imageService = context.read<ImageService>();
    final pickedFile = await imageService.pickImage();
    if (pickedFile != null && mounted) {
      _handleFiles([pickedFile]);
    }
  }

  /// Handles a list of files from drag-and-drop or picker.
  /// Accepts the first valid image file and rejects empty lists.
  void _handleFiles(List<XFile> files) {
    if (files.isEmpty) return;

    // Find the first file that appears to be an image
    final imageFile = files.firstWhere(
      (file) => _isImageFile(file),
      orElse: () => files.first,
    );

    setState(() {
      _selectedImage = imageFile;
      _isDragOver = false;
    });

    widget.onImageSelected(imageFile);
  }

  /// Checks if an XFile appears to be an image based on extension.
  bool _isImageFile(XFile file) {
    final name = file.name.toLowerCase();
    return name.endsWith('.png') ||
        name.endsWith('.jpg') ||
        name.endsWith('.jpeg') ||
        name.endsWith('.webp') ||
        name.endsWith('.bmp');
  }

  @override
  Widget build(BuildContext context) {
    return DropTarget(
      onDragEntered: (details) {
        setState(() => _isDragOver = true);
      },
      onDragExited: (details) {
        setState(() => _isDragOver = false);
      },
      onDragDone: (details) {
        final files = details.files;
        _handleFiles(files);
      },
      child: GestureDetector(
        key: const Key('drop_zone'),
        onTap: _pickImage,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeInOut,
          decoration: BoxDecoration(
            border: Border.all(
              color: _isDragOver
                  ? Theme.of(context).colorScheme.primary
                  : Theme.of(context).colorScheme.outline,
              width: _isDragOver ? 3 : 2,
            ),
            borderRadius: BorderRadius.circular(16),
            color: _isDragOver
                ? Theme.of(context).colorScheme.primaryContainer
                      .withValues(alpha: 0.3)
                : Theme.of(context).colorScheme.surfaceContainerHighest,
          ),
          child: _selectedImage != null ? _buildPreview() : _buildPlaceholder(),
        ),
      ),
    );
  }

  Widget _buildPreview() {
    return SizedBox(
      height: 240,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(14),
        child: Stack(
          fit: StackFit.expand,
          children: [
            FutureBuilder<Uint8List>(
              future: _selectedImage!.readAsBytes(),
              builder: (context, snapshot) {
                if (snapshot.hasData) {
                  return Image.memory(
                    snapshot.data!,
                    fit: BoxFit.cover,
                    errorBuilder: (context, error, stackTrace) =>
                        _buildErrorPlaceholder(),
                  );
                }
                return const Center(child: CircularProgressIndicator());
              },
            ),
            Positioned(
              top: 8,
              right: 8,
              child: Material(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(16),
                child: InkWell(
                  onTap: () {
                    setState(() => _selectedImage = null);
                  },
                  borderRadius: BorderRadius.circular(16),
                  child: const Padding(
                    padding: EdgeInsets.all(8),
                    child: Icon(Icons.close, color: Colors.white, size: 20),
                  ),
                ),
              ),
            ),
            // Classify button overlay at bottom
            if (widget.onClassify != null)
              Positioned(
                bottom: 16,
                left: 16,
                right: 16,
                child: FilledButton.icon(
                  onPressed: widget.isBusy ? null : widget.onClassify,
                  icon: widget.isBusy
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : const Icon(Icons.search),
                  label: Text(widget.isBusy ? 'Classifying...' : 'Classify'),
                  style: FilledButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    textStyle: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildErrorPlaceholder() {
    return Container(
      color: Theme.of(context).colorScheme.errorContainer,
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.broken_image,
              size: 48,
              color: Theme.of(context).colorScheme.onErrorContainer,
            ),
            const SizedBox(height: 8),
            Text(
              'Preview unavailable',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                color: Theme.of(context).colorScheme.onErrorContainer,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPlaceholder() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.cloud_upload_outlined,
              size: 64,
              color: Theme.of(context).colorScheme.onSurfaceVariant,
            ),
            const SizedBox(height: 16),
            Text(
              'Drag & drop an image here\nor tap to select',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'PNG, JPEG • Max 10 MB',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant
                    .withValues(alpha: 0.7),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
