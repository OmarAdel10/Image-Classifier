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
      // The ports are explicit so the scheme check is the only reason these
      // are rejected: without a port the URL is already invalid because the
      // port is out of range, which would hide a regression of the scheme
      // allowlist.
      expect(Config.isValidApiBaseUrl('ftp://example.com:21'), isFalse);
      expect(Config.isValidApiBaseUrl('ws://example.com:80'), isFalse);
      expect(Config.isValidApiBaseUrl('javascript://example.com:80'), isFalse);
      expect(Config.isValidApiBaseUrl('file:///etc/passwd'), isFalse);
      expect(Config.isValidApiBaseUrl('javascript:alert(1)'), isFalse);
    });

    test('accepts an https URL with an explicit default port', () {
      // Control for the scheme allowlist: the same shape as the rejected
      // URLs above, but with an allowed scheme.
      expect(Config.isValidApiBaseUrl('https://example.com:443'), isTrue);
    });

    test('rejects whitespace anywhere in the URL', () {
      expect(Config.isValidApiBaseUrl('http://exa mple.com'), isFalse);
      // A space in the path is percent-encoded by Uri.tryParse, so the
      // host-only check would not catch it.
      expect(Config.isValidApiBaseUrl('http://example.com/a b'), isFalse);
    });

    test('rejects percent-encoded characters anywhere in the URL', () {
      expect(Config.isValidApiBaseUrl('http://exa%zzmple.com'), isFalse);
      expect(Config.isValidApiBaseUrl('http://example.com/a%20b'), isFalse);
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
      expect(
        Config.isValidApiBaseUrl(Config.platformDefaultApiBaseUrl),
        isTrue,
      );
    });
  });

  group('Config.ensureValidApiBaseUrl (IC-60 enforcement)', () {
    test('returns without throwing for a valid URL', () {
      expect(
        () => Config.ensureValidApiBaseUrl('http://localhost:8000'),
        returnsNormally,
      );
    });

    test('throws StateError for a non-http scheme', () {
      expect(
        () => Config.ensureValidApiBaseUrl('ftp://example.com'),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('Invalid API_BASE_URL'),
          ),
        ),
      );
    });

    test('throws StateError for a URL without a host', () {
      expect(
        () => Config.ensureValidApiBaseUrl('http://'),
        throwsA(isA<StateError>()),
      );
    });

    test('the error message never echoes a credential-bearing URL', () {
      expect(
        () => Config.ensureValidApiBaseUrl('http://user:pass@example.com'),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            isNot(contains('user:pass')),
          ),
        ),
      );
    });
  });
}
