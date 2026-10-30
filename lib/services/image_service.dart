import 'dart:io';
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:image_classifier/config.dart';

/// Service for picking and validating/compressing images.
class ImageService {
  ImageService();

  /// Picks an image from the gallery or file system.
  /// Returns the picked image file, or null if cancelled.
  Future<File?> pickImage() async {
    // TODO: Implement using image_picker and file_picker
    throw UnimplementedError('pickImage not implemented yet');
  }

  /// Validates and compresses an image file for upload.
  ///
  /// Checks:
  /// - File size <= [Config.maxImageSizeBytes]
  /// - Valid image format (PNG, JPEG)
  /// - Resizes if dimensions exceed 1024px
  /// - Re-encodes as JPEG at 85% quality
  ///
  /// Returns the compressed image file.
  /// Throws [StateError] if validation fails.
  Future<File> validateAndCompress(File file) async {
    // Check file size
    final fileSize = await file.length();
    if (fileSize > Config.maxImageSizeBytes) {
      throw StateError(
        'File size ${fileSize}B exceeds maximum ${Config.maxImageSizeBytes}B',
      );
    }

    // Read and decode image
    final bytes = await file.readAsBytes();
    final image = img.decodeImage(bytes);

    if (image == null) {
      throw StateError('Invalid image format or corrupted file');
    }

    // Check format
    final isPng = _isPng(bytes);
    final isJpeg = _isJpeg(bytes);
    if (!isPng && !isJpeg) {
      throw StateError(
        'Unsupported image format. Only PNG and JPEG are allowed.',
      );
    }

    // Resize if needed (max dimension 1024)
    final resized = _resizeIfNeeded(image);

    // Encode as JPEG at 85% quality
    final jpegBytes = img.encodeJpg(resized, quality: 85);

    // Check compressed size
    if (jpegBytes.length > Config.maxImageSizeBytes) {
      throw StateError('Compressed image exceeds maximum size');
    }

    // Write to temp file
    final tempDir = Directory.systemTemp.createTempSync('image_classifier_');
    final outputFile = File('${tempDir.path}/compressed_image.jpg');
    await outputFile.writeAsBytes(jpegBytes);

    return outputFile;
  }

  /// Checks if bytes represent a PNG file.
  bool _isPng(Uint8List bytes) {
    return bytes.length >= 8 &&
        bytes[0] == 0x89 &&
        bytes[1] == 0x50 &&
        bytes[2] == 0x4E &&
        bytes[3] == 0x47 &&
        bytes[4] == 0x0D &&
        bytes[5] == 0x0A &&
        bytes[6] == 0x1A &&
        bytes[7] == 0x0A;
  }

  /// Checks if bytes represent a JPEG file.
  bool _isJpeg(Uint8List bytes) {
    return bytes.length >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8;
  }

  /// Resizes image if either dimension exceeds 1024px.
  img.Image _resizeIfNeeded(img.Image image) {
    const maxDimension = 1024;
    if (image.width <= maxDimension && image.height <= maxDimension) {
      return image;
    }

    final scale =
        maxDimension /
        (image.width > image.height ? image.width : image.height);
    final newWidth = (image.width * scale).round();
    final newHeight = (image.height * scale).round();

    return img.copyResize(
      image,
      width: newWidth,
      height: newHeight,
      interpolation: img.Interpolation.linear,
    );
  }
}
