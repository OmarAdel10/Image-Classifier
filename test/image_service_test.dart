import 'package:cross_file/cross_file.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mockito/annotations.dart';

import 'package:image_classifier/services/image_service.dart';

/// Mock for the picker function.
class MockPicker {
  MockPicker(this.pickFile);
  final Future<XFile?> Function() pickFile;
}

@GenerateNiceMocks([MockSpec<MockPicker>()])
void main() {
  group('ImageService', () {
    group('pickImage', () {
      test('returns null when picker returns null (cancelled)', () async {
        final mockPicker = MockPicker(() async => null);
        final service = ImageService(picker: mockPicker.pickFile);

        final result = await service.pickImage();

        expect(result, isNull);
      });

      test('returns XFile when picker returns a file', () async {
        final testFile = XFile('/test/path/image.jpg');
        final mockPicker = MockPicker(() async => testFile);
        final service = ImageService(picker: mockPicker.pickFile);

        final result = await service.pickImage();

        expect(result, equals(testFile));
      });

      test('rethrows exception from picker', () async {
        final mockPicker = MockPicker(
          () async => throw StateError('picker failed'),
        );
        final service = ImageService(picker: mockPicker.pickFile);

        expect(() => service.pickImage(), throwsA(isA<StateError>()));
      });
    });
  });
}
