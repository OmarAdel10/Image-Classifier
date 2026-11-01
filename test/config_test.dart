import 'package:flutter_test/flutter_test.dart';
import 'package:image_classifier/config.dart';

void main() {
  group('Config.isValidApiBaseUrl', () {
    test('accepts http URL with host', () {
      expect(Config.isValidApiBaseUrl('http://localhost:8000'), isTrue);
    });

    test('accepts https URL with host', () {
      expect(Config.isValidApiBaseUrl('https://api.example.com'), isTrue);
    });

    test('accepts URL with port and path prefix', () {
      expect(Config.isValidApiBaseUrl('https://example.com/api/v1'), isTrue);
    });

    test('rejects non-http scheme (URL injection guard, IC-60)', () {
      expect(Config.isValidApiBaseUrl('ftp://example.com'), isFalse);
      expect(Config.isValidApiBaseUrl('file:///etc/passwd'), isFalse);
      expect(Config.isValidApiBaseUrl('javascript:alert(1)'), isFalse);
    });

    test('rejects URL without host', () {
      expect(Config.isValidApiBaseUrl('http://'), isFalse);
      expect(Config.isValidApiBaseUrl('/api/v1'), isFalse);
    });

    test('rejects malformed and empty strings', () {
      expect(Config.isValidApiBaseUrl('not a url :://'), isFalse);
      expect(Config.isValidApiBaseUrl(''), isFalse);
    });

    test('default apiBaseUrl is valid', () {
      // Guards the shipped default so main() never throws out of the box.
      expect(Config.isValidApiBaseUrl(Config.apiBaseUrl), isTrue);
    });
  });
}
