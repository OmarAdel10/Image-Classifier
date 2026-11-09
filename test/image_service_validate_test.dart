import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:image_classifier/config.dart';
import 'package:image_classifier/services/image_service.dart';

/// Writes [bytes] into a fresh temporary file and returns it.
File _writeTempFile(String name, List<int> bytes) {
  final directory = Directory.systemTemp.createTempSync('image_service_test_');
  final file = File('${directory.path}/$name');
  file.writeAsBytesSync(bytes);
  return file;
}

/// Returns the bytes of a PNG image with the given dimensions.
List<int> _pngBytes(int width, int height) {
  return img.encodePng(img.Image(width: width, height: height));
}

/// Returns the bytes of a JPEG image with the given dimensions.
List<int> _jpegBytes(int width, int height) {
  return img.encodeJpg(img.Image(width: width, height: height));
}

void main() {
  group('ImageService.validateAndCompress', () {
    late ImageService service;

    setUp(() {
      service = ImageService(picker: () async => null);
    });

    test('turns a PNG into a JPEG file', () async {
      final file = _writeTempFile('image.png', _pngBytes(4, 4));

      final result = await service.validateAndCompress(file);

      expect(result.existsSync(), isTrue);
      expect(result.lengthSync(), greaterThan(0));
      final bytes = result.readAsBytesSync();
      // The result is a JPEG, whatever the input was.
      expect(bytes[0], 0xFF);
      expect(bytes[1], 0xD8);
    });

    test('accepts a JPEG file', () async {
      final file = _writeTempFile('image.jpg', _jpegBytes(4, 4));

      final result = await service.validateAndCompress(file);

      expect(result.existsSync(), isTrue);
      final bytes = result.readAsBytesSync();
      expect(bytes[0], 0xFF);
      expect(bytes[1], 0xD8);
    });

    test('rejects a file that is larger than the limit', () async {
      final file = _writeTempFile(
        'large.png',
        List<int>.filled(Config.maxImageSizeBytes + 1, 0),
      );

      await expectLater(
        service.validateAndCompress(file),
        throwsA(
          isA<ImageValidationException>().having(
            (e) => e.message,
            'message',
            'the file is larger than the ${Config.maxImageSizeBytes} byte limit',
          ),
        ),
      );
    });

    test('accepts a file of exactly the limit size', () async {
      // The file holds the limit in bytes but is not an image, so the
      // rejection has to name the format and not the size.
      final file = _writeTempFile(
        'limit.png',
        List<int>.filled(Config.maxImageSizeBytes, 0),
      );

      await expectLater(
        service.validateAndCompress(file),
        throwsA(
          isA<ImageValidationException>().having(
            (e) => e.message,
            'message',
            'the file is not a valid PNG or JPEG image',
          ),
        ),
      );
    });

    test('rejects a file that is not an image', () async {
      final file = _writeTempFile('broken.png', [1, 2, 3, 4, 5, 6, 7, 8]);

      await expectLater(
        service.validateAndCompress(file),
        throwsA(
          isA<ImageValidationException>().having(
            (e) => e.message,
            'message',
            'the file is not a valid PNG or JPEG image',
          ),
        ),
      );
    });

    test('rejects an image that is neither PNG nor JPEG', () async {
      final file = _writeTempFile(
        'animation.gif',
        img.encodeGif(img.Image(width: 2, height: 2)),
      );

      await expectLater(
        service.validateAndCompress(file),
        throwsA(
          isA<ImageValidationException>().having(
            (e) => e.message,
            'message',
            'only PNG and JPEG images are supported',
          ),
        ),
      );
    });

    test('shrinks an image that is wider than the limit', () async {
      final file = _writeTempFile('wide.png', _pngBytes(2000, 10));

      final result = await service.validateAndCompress(file);

      final image = img.decodeImage(result.readAsBytesSync());
      expect(image, isNotNull);
      expect(image!.width, lessThanOrEqualTo(1024));
      expect(image.height, lessThanOrEqualTo(1024));
    });

    test('keeps the size of an image that fits', () async {
      final file = _writeTempFile('small.png', _pngBytes(8, 6));

      final result = await service.validateAndCompress(file);

      final image = img.decodeImage(result.readAsBytesSync());
      expect(image, isNotNull);
      expect(image!.width, 8);
      expect(image.height, 6);
    });

    test('rejects a compressed image that is larger than the limit', () async {
      // A gradient is used because its JPEG is larger than its PNG (about
      // 51 KB against 22 KB), so the limit can sit between the two sizes:
      // the input passes the check, the compressed result does not.
      final gradient = img.Image(width: 1024, height: 1024);
      for (final pixel in gradient) {
        pixel.r = (pixel.x * 255 / 1024).round();
        pixel.g = (pixel.y * 255 / 1024).round();
        pixel.b = ((pixel.x + pixel.y) * 255 / 2048).round();
      }
      final file = _writeTempFile('gradient.png', img.encodePng(gradient));
      final limit = file.lengthSync();

      await expectLater(
        service.validateAndCompress(file, maxBytes: limit),
        throwsA(
          isA<ImageValidationException>().having(
            (e) => e.message,
            'message',
            'the compressed image is larger than the size limit',
          ),
        ),
      );
    });

    test('the custom limit drives the input size check', () async {
      final file = _writeTempFile('small.png', _pngBytes(4, 4));

      await expectLater(
        service.validateAndCompress(file, maxBytes: 8),
        throwsA(
          isA<ImageValidationException>().having(
            (e) => e.message,
            'message',
            'the file is larger than the 8 byte limit',
          ),
        ),
      );
    });

    test(
      'a limit equal to the file size does not trip the size check',
      () async {
        // The bytes are not an image, so the rejection has to come from the
        // format check and not from the size check.
        final file = _writeTempFile('broken.png', [1, 2, 3, 4, 5, 6, 7, 8]);

        await expectLater(
          service.validateAndCompress(file, maxBytes: file.lengthSync()),
          throwsA(
            isA<ImageValidationException>().having(
              (e) => e.message,
              'message',
              'the file is not a valid PNG or JPEG image',
            ),
          ),
        );
      },
    );

    test('a limit of zero rejects every file', () async {
      final file = _writeTempFile('small.png', _pngBytes(4, 4));

      await expectLater(
        service.validateAndCompress(file, maxBytes: 0),
        throwsA(
          isA<ImageValidationException>().having(
            (e) => e.message,
            'message',
            'the file is larger than the 0 byte limit',
          ),
        ),
      );
    });

    test('writes the result into a temporary JPEG file', () async {
      final file = _writeTempFile('image.png', _pngBytes(4, 4));

      final result = await service.validateAndCompress(file);

      expect(result.path, contains('image_classifier_'));
      expect(result.path, endsWith('.jpg'));
      expect(result.path, startsWith(Directory.systemTemp.path));
    });

    test(
      'every rejection names a reason that may be shown to the user',
      () async {
        final files = <File>[
          _writeTempFile(
            'large.png',
            List<int>.filled(Config.maxImageSizeBytes + 1, 0),
          ),
          _writeTempFile('broken.png', [1, 2, 3, 4, 5, 6, 7, 8]),
          _writeTempFile(
            'animation.gif',
            img.encodeGif(img.Image(width: 2, height: 2)),
          ),
        ];

        for (final file in files) {
          final message = await service
              .validateAndCompress(file)
              .then<ImageValidationException>(
                (_) => fail('expected a rejection'),
              )
              .catchError((Object error) => error as ImageValidationException);

          expect(message.message, isNotEmpty);
          // The message reaches the screen as it is, so it may not carry the
          // name of an exception type or the path of the file.
          expect(message.message, isNot(contains('Exception')));
          expect(message.message, isNot(contains('StateError')));
          expect(message.message, isNot(contains(file.path)));
        }
      },
    );
  });

  group('ImageValidationException', () {
    test('names the type and the reason', () {
      const exception = ImageValidationException('only PNG and JPEG');

      expect(
        exception.toString(),
        'ImageValidationException: only PNG and JPEG',
      );
    });
  });
}
