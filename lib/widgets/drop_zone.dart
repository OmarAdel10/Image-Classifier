import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:image_classifier/services/image_service.dart';

/// A drop zone widget that allows users to pick an image via tap or drag-and-drop.
class DropZone extends StatefulWidget {
  const DropZone({super.key});

  @override
  State<DropZone> createState() => _DropZoneState();
}

class _DropZoneState extends State<DropZone> {
  File? _previewImage;
  bool _isDragOver = false;

  Future<void> _pickImage() async {
    final imageService = context.read<ImageService>();
    final pickedFile = await imageService.pickImage();
    if (pickedFile != null && mounted) {
      setState(() {
        _previewImage = File(pickedFile.path);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      key: const Key('drop_zone'),
      onTap: _pickImage,
      child: DragTarget<File>(
        onWillAcceptWithDetails: (details) => true,
        onAcceptWithDetails: (details) {
          final file = details.data;
          setState(() {
            _previewImage = file;
            _isDragOver = false;
          });
        },
        onLeave: (_) {
          setState(() => _isDragOver = false);
        },
        builder: (context, candidateFiles, rejectedFiles) {
          final isDragOver = candidateFiles.isNotEmpty;
          if (isDragOver != _isDragOver) {
            WidgetsBinding.instance.addPostFrameCallback((_) {
              if (mounted) setState(() => _isDragOver = isDragOver);
            });
          }

          return AnimatedContainer(
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
            child: _previewImage != null
                ? _buildPreview()
                : _buildPlaceholder(),
          );
        },
      ),
    );
  }

  Widget _buildPreview() {
    return ClipRRect(
      borderRadius: BorderRadius.circular(14),
      child: Stack(
        fit: StackFit.expand,
        children: [
          Image.file(_previewImage!, fit: BoxFit.cover),
          Positioned(
            top: 8,
            right: 8,
            child: Material(
              color: Colors.black54,
              borderRadius: BorderRadius.circular(16),
              child: InkWell(
                onTap: () => setState(() => _previewImage = null),
                borderRadius: BorderRadius.circular(16),
                child: const Padding(
                  padding: EdgeInsets.all(8),
                  child: Icon(Icons.close, color: Colors.white, size: 20),
                ),
              ),
            ),
          ),
        ],
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
