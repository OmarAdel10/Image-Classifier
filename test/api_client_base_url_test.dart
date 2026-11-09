import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image_classifier/config.dart';
import 'package:image_classifier/services/api_client.dart';
import 'package:shared_preferences/shared_preferences.dart';

File _createTempImageFile() {
  final tempDir = Directory.systemTemp.createTempSync('api_client_base_url_');
  final file = File('${tempDir.path}/test_image.png');
  // Minimal valid 1x1 PNG.
  file.writeAsBytesSync([
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
    0x00, 0x00, 0x00, 0x0D, // IHDR chunk length
    0x49, 0x48, 0x44, 0x52, // IHDR
    0x00, 0x00, 0x00, 0x01, // width: 1
    0x00, 0x00, 0x00, 0x01, // height: 1
    0x08,
    0x02,
    0x00,
    0x00,
    0x00, // bit depth, color type, compression, filter, interlace
    0x90, 0x77, 0x53, 0xDE, // CRC
    0x00, 0x00, 0x00, 0x0C, // IDAT chunk length
    0x49, 0x44, 0x41, 0x54, // IDAT
    0x08, 0xD7, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x03, 0x01, 0x01, 0x00,
    0x18, 0xDD, 0x8D, 0xB4, // compressed data + CRC
    0x00, 0x00, 0x00, 0x00, // IEND chunk length
    0x49, 0x45, 0x4E, 0x44, // IEND
    0xAE, 0x42, 0x60, 0x82, // CRC
  ]);
  return file;
}

void main() {
  group('ApiClient base URL', () {
    setUp(() {
      // Start every test with an empty mock preference store.
      SharedPreferences.setMockInitialValues({});
    });

    group('explicit base URL', () {
      test('is used for requests', () async {
        final requestedUrls = <String>[];
        final mockClient = MockClient((request) async {
          requestedUrls.add(request.url.toString());
          return http.Response('{"status": "ok"}', 200);
        });
        final apiClient = ApiClient(
          client: mockClient,
          baseUrl: 'http://explicit.example.com:9000',
        );

        expect(await apiClient.healthCheck(), isTrue);
        expect(requestedUrls, hasLength(1));
        expect(requestedUrls.single, 'http://explicit.example.com:9000/health');
      });

      test('takes precedence over the stored runtime override', () async {
        SharedPreferences.setMockInitialValues({
          Config.apiBaseUrlPrefsKey: 'http://stored.example.com:1234',
        });
        final requestedUrls = <String>[];
        final mockClient = MockClient((request) async {
          requestedUrls.add(request.url.toString());
          return http.Response('{"status": "ok"}', 200);
        });
        final apiClient = ApiClient(
          client: mockClient,
          baseUrl: 'http://explicit.example.com:9000',
        );

        expect(await apiClient.healthCheck(), isTrue);
        expect(requestedUrls.single, 'http://explicit.example.com:9000/health');
      });

      test('is exposed by the baseUrl getter', () async {
        final apiClient = ApiClient(
          client: MockClient((request) async => http.Response('{}', 200)),
          baseUrl: 'http://explicit.example.com:9000',
        );

        expect(await apiClient.baseUrl, 'http://explicit.example.com:9000');
      });

      test('trailing slashes are stripped before paths are appended', () async {
        final requestedUrls = <String>[];
        final mockClient = MockClient((request) async {
          requestedUrls.add(request.url.toString());
          return http.Response('{"status": "ok"}', 200);
        });
        final apiClient = ApiClient(
          client: mockClient,
          baseUrl: 'http://host:8000/',
        );

        expect(await apiClient.baseUrl, 'http://host:8000');
        expect(await apiClient.healthCheck(), isTrue);
        expect(requestedUrls.single, 'http://host:8000/health');
      });
    });

    group('resolved base URL', () {
      test('uses Config.resolveApiBaseUrl when no URL is given', () async {
        SharedPreferences.setMockInitialValues({
          Config.apiBaseUrlPrefsKey: 'http://stored.example.com:1234',
        });
        final requestedUrls = <String>[];
        final mockClient = MockClient((request) async {
          requestedUrls.add(request.url.toString());
          return http.Response('{"status": "ok"}', 200);
        });
        final apiClient = ApiClient(client: mockClient);

        expect(await apiClient.healthCheck(), isTrue);
        expect(requestedUrls.single, 'http://stored.example.com:1234/health');
      });

      test(
        'falls back to the platform default when no override is stored',
        () async {
          final requestedUrls = <String>[];
          final mockClient = MockClient((request) async {
            requestedUrls.add(request.url.toString());
            return http.Response('{"status": "ok"}', 200);
          });
          final apiClient = ApiClient(client: mockClient);

          expect(await apiClient.healthCheck(), isTrue);
          expect(
            requestedUrls.single,
            '${Config.platformDefaultApiBaseUrl}/health',
          );
        },
      );

      test('is cached: a later preference change is not picked up', () async {
        SharedPreferences.setMockInitialValues({
          Config.apiBaseUrlPrefsKey: 'http://first.example.com:1111',
        });
        final requestedUrls = <String>[];
        final mockClient = MockClient((request) async {
          requestedUrls.add(request.url.toString());
          return http.Response('{"status": "ok"}', 200);
        });
        final apiClient = ApiClient(client: mockClient);

        expect(await apiClient.healthCheck(), isTrue);

        // Change the stored override after the client resolved its URL.
        final prefs = await SharedPreferences.getInstance();
        await prefs.setString(
          Config.apiBaseUrlPrefsKey,
          'http://second.example.com:2222',
        );

        expect(await apiClient.healthCheck(), isTrue);

        expect(requestedUrls, hasLength(2));
        expect(requestedUrls.first, 'http://first.example.com:1111/health');
        expect(requestedUrls.last, 'http://first.example.com:1111/health');
        // Sanity check: the preference really changed under the client.
        expect(
          prefs.getString(Config.apiBaseUrlPrefsKey),
          'http://second.example.com:2222',
        );
      });

      test(
        'invalidateBaseUrlCache makes the next request use the new URL',
        () async {
          SharedPreferences.setMockInitialValues({
            Config.apiBaseUrlPrefsKey: 'http://first.example.com:1111',
          });
          final requestedUrls = <String>[];
          final mockClient = MockClient((request) async {
            requestedUrls.add(request.url.toString());
            return http.Response('{"status": "ok"}', 200);
          });
          final apiClient = ApiClient(client: mockClient);

          expect(await apiClient.healthCheck(), isTrue);

          final prefs = await SharedPreferences.getInstance();
          await prefs.setString(
            Config.apiBaseUrlPrefsKey,
            'http://second.example.com:2222',
          );
          apiClient.invalidateBaseUrlCache();

          expect(await apiClient.healthCheck(), isTrue);

          expect(requestedUrls, hasLength(2));
          expect(requestedUrls.first, 'http://first.example.com:1111/health');
          expect(requestedUrls.last, 'http://second.example.com:2222/health');
        },
      );
    });

    group('healthStatusCode', () {
      test('returns 200 for a healthy backend', () async {
        final apiClient = ApiClient(
          client: MockClient((request) async {
            expect(request.url.toString(), endsWith('/health'));
            return http.Response('{"status": "ok"}', 200);
          }),
          baseUrl: 'http://healthy.example.com:8000',
        );

        expect(await apiClient.healthStatusCode(), 200);
      });

      test('returns the status code when reachable but unhealthy', () async {
        final apiClient = ApiClient(
          client: MockClient((request) async => http.Response('{}', 500)),
          baseUrl: 'http://unhealthy.example.com:8000',
        );

        expect(await apiClient.healthStatusCode(), 500);
      });

      test('returns null when the socket fails', () async {
        final apiClient = ApiClient(
          client: MockClient((request) async {
            throw SocketException('Connection refused');
          }),
          baseUrl: 'http://unreachable.example.com:8000',
        );

        expect(await apiClient.healthStatusCode(), isNull);
      });

      test('returns null when the request times out', () async {
        final apiClient = ApiClient(
          client: MockClient((request) async {
            await Future.delayed(const Duration(milliseconds: 200));
            return http.Response('{}', 200);
          }),
          baseUrl: 'http://slow.example.com:8000',
          timeout: const Duration(milliseconds: 50),
        );

        expect(await apiClient.healthStatusCode(), isNull);
      });

      test('returns null when the client throws a ClientException', () async {
        final apiClient = ApiClient(
          client: MockClient((request) async {
            throw http.ClientException('connection closed');
          }),
          baseUrl: 'http://client-error.example.com:8000',
        );

        expect(await apiClient.healthStatusCode(), isNull);
      });

      test('returns null when the request raises an ArgumentError', () async {
        final apiClient = ApiClient(
          client: MockClient((request) async {
            throw ArgumentError('bad request URL');
          }),
          baseUrl: 'http://arg-error.example.com:8000',
        );

        expect(await apiClient.healthStatusCode(), isNull);
      });

      test('returns null when a filesystem-level IOException occurs', () async {
        final apiClient = ApiClient(
          client: MockClient((request) async {
            throw const FileSystemException('disk error', '/tmp/x');
          }),
          baseUrl: 'http://io-error.example.com:8000',
        );

        expect(await apiClient.healthStatusCode(), isNull);
      });

      test('returns null when the response raises an HttpException', () async {
        final apiClient = ApiClient(
          client: MockClient((request) async {
            throw HttpException('malformed response');
          }),
          baseUrl: 'http://http-error.example.com:8000',
        );

        expect(await apiClient.healthStatusCode(), isNull);
      });

      test('returns null when the base URL cannot be parsed', () async {
        final apiClient = ApiClient(
          client: MockClient((request) async => http.Response('{}', 200)),
          // An unterminated IPv6 host makes Uri.parse throw.
          baseUrl: 'http://[bad',
        );

        expect(await apiClient.healthStatusCode(), isNull);
      });
    });

    group('error messages include the base URL', () {
      test(
        'a SocketException produces an ApiException naming the URL',
        () async {
          final mockClient = MockClient((request) async {
            throw SocketException('Connection refused');
          });
          final apiClient = ApiClient(
            client: mockClient,
            baseUrl: 'http://socket.example.com:8000',
          );

          await expectLater(
            apiClient.getModelInfo(),
            throwsA(
              isA<ApiException>()
                  .having(
                    (e) => e.message,
                    'message',
                    contains('http://socket.example.com:8000'),
                  )
                  .having((e) => e.statusCode, 'statusCode', 503),
            ),
          );
        },
      );

      test(
        'an HttpException produces a transport ApiException with status 503',
        () async {
          final mockClient = MockClient((request) async {
            throw const HttpException('connection closed');
          });
          final apiClient = ApiClient(
            client: mockClient,
            baseUrl: 'http://http-error.example.com:8000',
          );

          await expectLater(
            apiClient.getModelInfo(),
            throwsA(
              isA<ApiException>()
                  .having(
                    (e) => e.message,
                    'message',
                    allOf(
                      contains('http://http-error.example.com:8000'),
                      // The raw exception text stays out of the message.
                      isNot(contains('connection closed')),
                    ),
                  )
                  .having((e) => e.statusCode, 'statusCode', 503),
            ),
          );
        },
      );

      test(
        'a TimeoutException produces an ApiException naming the URL',
        () async {
          // Never completes, so the request can only end via the timeout.
          final never = Completer<void>();
          final mockClient = MockClient((request) async {
            await never.future;
            return http.Response('{}', 200);
          });
          final apiClient = ApiClient(
            client: mockClient,
            baseUrl: 'http://slow.example.com:8000',
            timeout: const Duration(milliseconds: 100),
          );

          await expectLater(
            apiClient.getModelInfo(),
            throwsA(
              isA<ApiException>()
                  .having(
                    (e) => e.message,
                    'message',
                    allOf(
                      contains('http://slow.example.com:8000'),
                      contains('timed out'),
                    ),
                  )
                  .having((e) => e.statusCode, 'statusCode', 408),
            ),
          );
        },
      );

      testWidgets('predict names the URL when the socket fails', (
        WidgetTester tester,
      ) async {
        final file = _createTempImageFile();
        final mockClient = MockClient((request) async {
          throw SocketException('Connection refused');
        });
        final apiClient = ApiClient(
          client: mockClient,
          baseUrl: 'http://predict.example.com:8000',
          // The mock fails immediately, so the timeout only decides the
          // outcome when the machine is heavily loaded. A short timeout made
          // this test flaky (it raced the 408 path).
          timeout: const Duration(seconds: 5),
        );

        // The multipart body is read from the real file system, which the
        // fake-async zone of a widget test would never complete.
        await tester.runAsync(() async {
          await expectLater(
            apiClient.predict(file),
            throwsA(
              isA<ApiException>()
                  .having(
                    (e) => e.message,
                    'message',
                    contains('http://predict.example.com:8000'),
                  )
                  .having((e) => e.statusCode, 'statusCode', 503),
            ),
          );
        });
      });

      test(
        'a ClientException produces a transport ApiException with status 503',
        () async {
          final mockClient = MockClient((request) async {
            throw http.ClientException('connection closed');
          });
          final apiClient = ApiClient(
            client: mockClient,
            baseUrl: 'http://client-error.example.com:8000',
          );

          await expectLater(
            apiClient.getModelInfo(),
            throwsA(
              isA<ApiException>()
                  .having(
                    (e) => e.message,
                    'message',
                    allOf(
                      contains('http://client-error.example.com:8000'),
                      // The raw exception text stays out of the message.
                      isNot(contains('connection closed')),
                    ),
                  )
                  .having((e) => e.statusCode, 'statusCode', 503),
            ),
          );
        },
      );

      test('an IOException produces an ApiException with status 503', () async {
        final mockClient = MockClient((request) async {
          throw const FileSystemException('disk error', '/tmp/x');
        });
        final apiClient = ApiClient(
          client: mockClient,
          baseUrl: 'http://io-error.example.com:8000',
        );

        await expectLater(
          apiClient.getModelInfo(),
          throwsA(
            isA<ApiException>()
                .having(
                  (e) => e.message,
                  'message',
                  allOf(
                    contains('Cannot reach http://io-error.example.com:8000'),
                    // The raw exception text stays out of the message.
                    isNot(contains('disk error')),
                  ),
                )
                .having((e) => e.statusCode, 'statusCode', 503),
          ),
        );
      });

      test(
        'an ArgumentError produces an ApiException with status 502',
        () async {
          final mockClient = MockClient((request) async {
            throw ArgumentError('bad address');
          });
          final apiClient = ApiClient(
            client: mockClient,
            baseUrl: 'http://arg-error.example.com:8000',
          );

          await expectLater(
            apiClient.getModelInfo(),
            throwsA(
              isA<ApiException>()
                  .having(
                    (e) => e.message,
                    'message',
                    allOf(
                      startsWith(
                        'Invalid request to http://arg-error.example.com:8000:',
                      ),
                      contains('bad address'),
                      // The address is validated before the request, so the
                      // message must not blame the backend address.
                      isNot(contains('Invalid backend address')),
                      isNot(contains('null')),
                      // The request is at fault, not the transport.
                      isNot(contains('Cannot reach')),
                      isNot(contains('Connection to')),
                    ),
                  )
                  .having((e) => e.statusCode, 'statusCode', 502),
            ),
          );
        },
      );

      test('the same wording is used when a request cannot be built', () async {
        final mockClient = MockClient((request) async {
          throw ArgumentError('bad address');
        });
        final apiClient = ApiClient(
          client: mockClient,
          baseUrl: 'http://arg-error.example.com:8000',
        );

        await expectLater(
          apiClient.predictBytes(Uint8List.fromList([0x00]), 'test.png'),
          throwsA(
            isA<ApiException>()
                .having(
                  (e) => e.message,
                  'message',
                  startsWith(
                    'Invalid request to http://arg-error.example.com:8000:',
                  ),
                )
                .having((e) => e.statusCode, 'statusCode', 502),
          ),
        );
      });

      test(
        'an ArgumentError without a message is reported without null',
        () async {
          final mockClient = MockClient((request) async {
            throw ArgumentError();
          });
          final apiClient = ApiClient(
            client: mockClient,
            baseUrl: 'http://arg-error.example.com:8000',
          );

          await expectLater(
            apiClient.getModelInfo(),
            throwsA(
              isA<ApiException>()
                  .having(
                    (e) => e.message,
                    'message',
                    allOf(
                      contains('invalid argument'),
                      isNot(contains('null')),
                    ),
                  )
                  .having((e) => e.statusCode, 'statusCode', 502),
            ),
          );
        },
      );

      test('a missing image file is reported as a file problem', () async {
        final mockClient = MockClient((request) async {
          return http.Response('{}', 200);
        });
        final apiClient = ApiClient(
          client: mockClient,
          baseUrl: 'http://predict.example.com:8000',
        );

        await expectLater(
          apiClient.predict(
            File('${Directory.systemTemp.path}/missing-image-file.png'),
          ),
          throwsA(
            isA<ApiException>()
                .having(
                  (e) => e.message,
                  'message',
                  allOf(
                    contains('Could not read the image file'),
                    // A missing file is not a connection problem.
                    isNot(contains('Connection to')),
                  ),
                )
                .having((e) => e.statusCode, 'statusCode', 400),
          ),
        );
      });

      test(
        'predictBytes names the URL when the client throws a ClientException',
        () async {
          final mockClient = MockClient((request) async {
            throw http.ClientException('connection closed');
          });
          final apiClient = ApiClient(
            client: mockClient,
            baseUrl: 'http://bytes.example.com:8000',
          );

          await expectLater(
            apiClient.predictBytes(Uint8List.fromList([1, 2, 3]), 'image.png'),
            throwsA(
              isA<ApiException>()
                  .having(
                    (e) => e.message,
                    'message',
                    allOf(
                      contains('http://bytes.example.com:8000'),
                      // The raw exception text stays out of the message.
                      isNot(contains('connection closed')),
                    ),
                  )
                  .having((e) => e.statusCode, 'statusCode', 503),
            ),
          );
        },
      );
    });

    group('malformed JSON responses', () {
      Future<void> expectInvalidJsonResponse(
        String body, {
        String? messageContains,
      }) async {
        final mockClient = MockClient(
          (request) async => http.Response(body, 200),
        );
        final apiClient = ApiClient(
          client: mockClient,
          baseUrl: 'http://bad-json.example.com:8000',
        );

        final messageParts = <Matcher>[
          contains('Invalid response from'),
          contains('http://bad-json.example.com:8000'),
          if (messageContains != null) contains(messageContains),
        ];

        await expectLater(
          apiClient.getModelInfo(),
          throwsA(
            isA<ApiException>()
                .having((e) => e.message, 'message', allOf(messageParts))
                .having((e) => e.statusCode, 'statusCode', 502),
          ),
        );
      }

      test(
        'a non-JSON body produces an ApiException with status 502',
        () async {
          await expectInvalidJsonResponse('not json');
        },
      );

      test(
        'a JSON array body produces an ApiException with status 502',
        () async {
          // The body is valid JSON but not a map, so the response-structure
          // check reports it. This distinguishes it from the TypeError path,
          // which reports an unexpected response shape.
          await expectInvalidJsonResponse(
            '[1,2]',
            messageContains: 'could not be parsed as the expected JSON',
          );
        },
      );

      test(
        'a 200 whose classes field is not a list produces an ApiException 502',
        () async {
          // Dedicated (non-helper) assertion: pins the TypeError-specific
          // message so this test fails if the `on TypeError` catch is
          // removed and the FormatException path silently takes over.
          final mockClient = MockClient(
            (request) async => http.Response('{"classes": "not-a-list"}', 200),
          );
          final apiClient = ApiClient(
            client: mockClient,
            baseUrl: 'http://bad-json.example.com:8000',
          );

          await expectLater(
            apiClient.getModelInfo(),
            throwsA(
              isA<ApiException>()
                  .having(
                    (e) => e.message,
                    'message',
                    allOf(
                      contains('Invalid response from'),
                      contains('unexpected response shape'),
                      contains('http://bad-json.example.com:8000'),
                    ),
                  )
                  .having((e) => e.statusCode, 'statusCode', 502),
            ),
          );
        },
      );
    });
  });
}
