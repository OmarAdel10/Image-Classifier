import 'dart:io';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:image/image.dart' as img;
import 'package:image_classifier/config.dart';
import 'package:image_picker/image_picker.dart';

/// Type alias for the image picker function to enable testing.
typedef PickFile = Future<XFile?> Function();

/// Thrown when a picked file cannot be used as an image.
///
/// The [message] is written for the user, so it can be shown as is. Any
/// other exception keeps its own text, which must never reach the UI.
class ImageValidationException implements Exception {
  const ImageValidationException(this.message);

  /// The human readable reason why the file was rejected.
  final String message;

  @override
  String toString() => 'ImageValidationException: $message';
}

/// Service for picking and validating/compressing images.
class ImageService {
  /// Creates an [ImageService] with an optional [picker] for testing.
  ///
  /// If [picker] is not provided, uses the platform-appropriate default:
  /// - Desktop (Linux/Windows/macOS): [FilePicker.platform.pickFiles]
  /// - Mobile (Android/iOS): [ImagePicker.pickImage]
  ImageService({PickFile? picker}) : _picker = picker ?? _defaultPicker;

  final PickFile _picker;

  /// Picks an image from the gallery or file system.
  /// Returns the picked image file, or null if cancelled.
  Future<XFile?> pickImage() async {
    return _picker();
  }

  /// Default platform-aware image picker.
  static Future<XFile?> _defaultPicker() async {
    // Desktop platforms use file_picker
    if (Platform.isLinux || Platform.isWindows || Platform.isMacOS) {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.image,
        allowMultiple: false,
      );
      if (result != null && result.files.isNotEmpty) {
        final path = result.files.single.path;
        if (path != null) {
          return XFile(path);
        }
      }
      return null;
    }

    // Mobile platforms use image_picker
    if (Platform.isAndroid || Platform.isIOS) {
      final picker = ImagePicker();
      return picker.pickImage(source: ImageSource.gallery);
    }

    // Fallback for other platforms (web, etc.)
    return null;
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
  /// Throws [ImageValidationException] if validation fails.
  Future<File> validateAndCompress(File file) async {
    // Check file size
    final fileSize = await file.length();
    if (fileSize > Config.maxImageSizeBytes) {
      throw ImageValidationException(
        'the file is larger than the ${Config.maxImageSizeBytes} byte limit',
      );
    }

    // Read and decode image
    final bytes = await file.readAsBytes();
    final image = img.decodeImage(bytes);

    if (image == null) {
      throw ImageValidationException(
        'the file is not a valid PNG or JPEG image',
      );
    }

    // Check format
    final isPng = _isPng(bytes);
    final isJpeg = _isJpeg(bytes);
    if (!isPng && !isJpeg) {
      throw ImageValidationException('only PNG and JPEG images are supported');
    }

    // Resize if needed (max dimension 1024)
    final resized = _resizeIfNeeded(image);

    // Encode as JPEG at 85% quality
    final jpegBytes = img.encodeJpg(resized, quality: 85);

    // Check compressed size
    if (jpegBytes.length > Config.maxImageSizeBytes) {
      throw ImageValidationException(
        'the compressed image is larger than the size limit',
      );
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
