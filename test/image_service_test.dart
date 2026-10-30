import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:image_classifier/config.dart';
import 'package:image_classifier/services/image_service.dart';

void main() {
  group('ImageService', () {
    late ImageService imageService;

    setUp(() {
      imageService = ImageService();
    });

    test(
      'validateAndCompress rejects file larger than maxImageSizeBytes',
      () async {
        final largeBytes = Uint8List(Config.maxImageSizeBytes + 1000);
        final file = await _createTempFile(largeBytes);

        expect(
          () => imageService.validateAndCompress(file),
          throwsA(isA<StateError>()),
        );
      },
    );

    test('validateAndCompress rejects non-image MIME type', () async {
      final file = await _createTempFile(
        Uint8List.fromList([0x25, 0x50, 0x44, 0x46]),
      ); // PDF magic bytes

      expect(
        () => imageService.validateAndCompress(file),
        throwsA(isA<StateError>()),
      );
    });

    test('validateAndCompress accepts valid PNG under size limit', () async {
      final pngBytes = _createValidPng();
      final file = await _createTempFile(pngBytes);

      final result = await imageService.validateAndCompress(file);

      expect(result, isA<File>());
      expect(
        await result.length(),
        lessThanOrEqualTo(Config.maxImageSizeBytes),
      );
      expect(await result.exists(), isTrue);
    });

    test(
      'validateAndCompress resizes image if dimensions exceed 1024',
      () async {
        final largePng = _createLargePng(2048, 2048);
        final file = await _createTempFile(largePng);

        final result = await imageService.validateAndCompress(file);

        // Verify file was compressed (should be much smaller than original)
        expect(await result.length(), lessThan(Config.maxImageSizeBytes));
      },
    );

    test('validateAndCompress re-encodes as JPEG with 85% quality', () async {
      final pngBytes = _createValidPng();
      final file = await _createTempFile(pngBytes);

      final result = await imageService.validateAndCompress(file);

      final header = await result.openRead().take(4).toList();
      // JPEG starts with 0xFF 0xD8 0xFF
      expect(header.first[0], 0xFF);
      expect(header.first[1], 0xD8);
      expect(header.first[2], 0xFF);
    });
  });
}

Future<File> _createTempFile(Uint8List bytes) async {
  final tempDir = Directory.systemTemp.createTempSync('image_service_test_');
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
