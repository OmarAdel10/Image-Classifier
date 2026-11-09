import 'dart:async';
import 'dart:io';
import 'dart:convert';
import 'dart:typed_data';

import 'package:cross_file/cross_file.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image_classifier/models/prediction.dart';
import 'package:image_classifier/models/model_info.dart';
import 'package:image_classifier/services/api_client.dart';

File _createTempImageFile() {
  final tempDir = Directory.systemTemp.createTempSync('api_client_test_');
  final file = File('${tempDir.path}/test_image.png');
  // Create a minimal valid PNG (1x1 pixel)
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

/// Deletes the image file before the request body is streamed.
///
/// This reproduces the race between the size check that runs before the
/// request and the upload that reads the file while the request is sent.
class _DeletingClient extends http.BaseClient {
  _DeletingClient(this.file);

  final File file;

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) async {
    file.deleteSync();
    // The file has to be gone before the body is read, so that the read
    // fails on every platform instead of reading an open handle.
    expect(file.existsSync(), isFalse);
    await request.finalize().toBytes();
    return http.StreamedResponse(const Stream<List<int>>.empty(), 200);
  }
}

/// Returns a response whose body cannot be read.
///
/// The connection succeeds and the status code is delivered, but reading the
/// body fails. That is a transfer problem, not a connection problem.
class _BrokenBodyClient extends http.BaseClient {
  _BrokenBodyClient(this.fault);

  /// The failure the body stream reports.
  ///
  /// A truncated body is reported by the real transport as a
  /// [http.ClientException]; a socket error arrives as a [SocketException].
  final Object fault;

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) async {
    return http.StreamedResponse(Stream<List<int>>.error(fault), 200);
  }
}

/// Returns a response whose body never ends.
///
/// The status code arrives, but the body stream stays open forever. Without a
/// bound on the body read the request would never finish.
class _StalledBodyClient extends http.BaseClient {
  _StalledBodyClient(this.body);

  final StreamController<List<int>> body;

  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) async {
    return http.StreamedResponse(body.stream, 200, contentLength: 128);
  }
}

/// A client whose send fails with a file system fault.
///
/// A custom client can read the request body from the file system, so the
/// fault can happen while the request is sent.
class _FileFaultOnSendClient extends http.BaseClient {
  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) async {
    throw FileSystemException(
      'the upload file could not be read',
      '/tmp/image_classifier_upload',
    );
  }
}

void main() {
  group('ApiClient', () {
    late MockClient mockClient;
    late ApiClient apiClient;

    setUp(() {
      mockClient = MockClient((request) async {
        return http.Response('{"error": "not implemented"}', 501);
      });
      apiClient = ApiClient(client: mockClient);
    });

    test('healthCheck returns true on 200', () async {
      mockClient = MockClient((request) async {
        expect(request.url.toString(), endsWith('/health'));
        expect(request.method, 'GET');
        return http.Response('{"status": "ok"}', 200);
      });
      apiClient = ApiClient(client: mockClient);

      final result = await apiClient.healthCheck();
      expect(result, isTrue);
    });

    test('healthCheck returns false on non-200', () async {
      mockClient = MockClient((request) async {
        return http.Response('{"status": "degraded"}', 503);
      });
      apiClient = ApiClient(client: mockClient);

      final result = await apiClient.healthCheck();
      expect(result, isFalse);
    });

    test('healthCheck returns false on exception', () async {
      mockClient = MockClient((request) async {
        throw SocketException('Connection refused');
      });
      apiClient = ApiClient(client: mockClient);

      final result = await apiClient.healthCheck();
      expect(result, isFalse);
    });

    test('getModelInfo returns parsed ModelInfo on 200', () async {
      mockClient = MockClient((request) async {
        expect(request.url.toString(), endsWith('/model-info'));
        expect(request.method, 'GET');
        return http.Response(
          jsonEncode({
            'model': 'cnn_scratch',
            'input_size': '32x32',
            'classes': [
              'airplane',
              'automobile',
              'bird',
              'cat',
              'deer',
              'dog',
              'frog',
              'horse',
              'ship',
              'truck',
            ],
            'device': 'cpu',
            'trained_at': '2026-10-29T10:00:00.000Z',
          }),
          200,
        );
      });
      apiClient = ApiClient(client: mockClient);

      final result = await apiClient.getModelInfo();

      expect(result, isA<ModelInfo>());
      expect(result.modelName, 'cnn_scratch');
      expect(result.inputSize, '32x32');
      expect(result.classes, hasLength(10));
      expect(result.classes.first, 'airplane');
      expect(result.device, 'cpu');
      expect(result.trainedAt, isNotNull);
      expect(result.trainedAt!.year, 2026);
    });

    test('getModelInfo throws ApiException on non-200', () async {
      mockClient = MockClient((request) async {
        return http.Response('{"detail": "model unavailable"}', 503);
      });
      apiClient = ApiClient(client: mockClient);

      expect(() => apiClient.getModelInfo(), throwsA(isA<ApiException>()));
    });

    test('getModelInfo sanitizes control characters in error detail', () async {
      mockClient = MockClient((request) async {
        return http.Response(
          jsonEncode({'detail': 'bad\u0000de\u0007tail\u001b\nok\t'}),
          503,
        );
      });
      apiClient = ApiClient(client: mockClient);

      await expectLater(
        apiClient.getModelInfo(),
        throwsA(
          isA<ApiException>().having(
            (e) => e.message,
            'message',
            'baddetail\nok\t',
          ),
        ),
      );
    });

    test('getModelInfo truncates overlong error detail', () async {
      mockClient = MockClient((request) async {
        return http.Response(jsonEncode({'detail': 'x' * 600}), 503);
      });
      apiClient = ApiClient(client: mockClient);

      await expectLater(
        apiClient.getModelInfo(),
        throwsA(
          isA<ApiException>().having(
            (e) => e.message,
            'message',
            matches(RegExp(r'^x{500}\.\.\. \[truncated\]$')),
          ),
        ),
      );
    });

    test('getModelInfo keeps detail at exactly the 500-char cap', () async {
      mockClient = MockClient((request) async {
        return http.Response(jsonEncode({'detail': 'x' * 500}), 503);
      });
      apiClient = ApiClient(client: mockClient);

      await expectLater(
        apiClient.getModelInfo(),
        throwsA(
          isA<ApiException>().having((e) => e.message, 'message', 'x' * 500),
        ),
      );
    });

    test(
      'getModelInfo falls back to sanitized default on non-JSON body',
      () async {
        mockClient = MockClient((request) async {
          return http.Response('oops\u0000html not json', 502);
        });
        apiClient = ApiClient(client: mockClient);

        await expectLater(
          apiClient.getModelInfo(),
          throwsA(
            isA<ApiException>().having(
              (e) => e.message,
              'message',
              'Request failed with status 502',
            ),
          ),
        );
      },
    );

    test('getModelInfo falls back on non-String detail', () async {
      final body = jsonEncode({
        'detail': [
          {
            'loc': ['file'],
            'msg': 'm',
          },
        ],
      });
      mockClient = MockClient((request) async {
        return http.Response(body, 422);
      });
      apiClient = ApiClient(client: mockClient);

      await expectLater(
        apiClient.getModelInfo(),
        throwsA(
          isA<ApiException>().having(
            (e) => e.message,
            'message',
            'Request failed with status 422',
          ),
        ),
      );
    });

    test('predict returns parsed PredictionResponse on 200', () async {
      final xFile = XFile(_createTempImageFile().path);
      mockClient = MockClient((request) async {
        expect(request.url.toString(), endsWith('/predict'));
        expect(request.method, 'POST');
        expect(
          request.headers['content-type'],
          contains('multipart/form-data'),
        );
        return http.Response(
          jsonEncode({
            'predictions': [
              {'class_name': 'cat', 'confidence': 0.9},
              {'class_name': 'dog', 'confidence': 0.1},
            ],
            'latency_ms': 42.5,
            'model': 'cnn_scratch',
          }),
          200,
        );
      });
      apiClient = ApiClient(client: mockClient);

      final result = await apiClient.predictXFile(xFile);

      expect(result, isA<PredictionResponse>());
      expect(result.predictions, hasLength(2));
      expect(result.predictions.first.className, 'cat');
      expect(result.predictions.first.confidence, 0.9);
      expect(result.predictions.last.className, 'dog');
      expect(result.predictions.last.confidence, 0.1);
      expect(result.latencyMs, 42.5);
      expect(result.model, 'cnn_scratch');
    });

    test('predict throws ApiException on 400', () async {
      final xFile = XFile(_createTempImageFile().path);
      mockClient = MockClient((request) async {
        return http.Response('{"detail": "file must be an image"}', 400);
      });
      apiClient = ApiClient(client: mockClient);

      await expectLater(
        apiClient.predictXFile(xFile),
        throwsA(
          isA<ApiException>()
              .having((e) => e.statusCode, 'statusCode', 400)
              .having(
                (e) => e.message,
                'message',
                contains('file must be an image'),
              ),
        ),
      );
    });

    test('predict throws ApiException on 413', () async {
      final xFile = XFile(_createTempImageFile().path);
      mockClient = MockClient((request) async {
        return http.Response('{"detail": "image file too large"}', 413);
      });
      apiClient = ApiClient(client: mockClient);

      await expectLater(
        apiClient.predictXFile(xFile),
        throwsA(
          isA<ApiException>()
              .having((e) => e.statusCode, 'statusCode', 413)
              .having(
                (e) => e.message,
                'message',
                contains('image file too large'),
              ),
        ),
      );
    });

    test('predict throws ApiException on 500', () async {
      final xFile = XFile(_createTempImageFile().path);
      mockClient = MockClient((request) async {
        return http.Response('{"detail": "model inference failed"}', 500);
      });
      apiClient = ApiClient(client: mockClient);

      await expectLater(
        apiClient.predictXFile(xFile),
        throwsA(
          isA<ApiException>()
              .having((e) => e.statusCode, 'statusCode', 500)
              .having(
                (e) => e.message,
                'message',
                contains('model inference failed'),
              ),
        ),
      );
    });

    test('predict throws ApiException on timeout', () async {
      final xFile = XFile(_createTempImageFile().path);
      mockClient = MockClient((request) async {
        await Future.delayed(const Duration(seconds: 1));
        return http.Response('{}', 200);
      });
      // A short timeout keeps the test fast and stops the pending response
      // from deciding the outcome of a later test.
      apiClient = ApiClient(
        client: mockClient,
        timeout: const Duration(milliseconds: 50),
      );

      await expectLater(
        apiClient.predictXFile(xFile),
        throwsA(
          isA<ApiException>()
              .having((e) => e.statusCode, 'statusCode', 408)
              .having((e) => e.message, 'message', contains('timed out')),
        ),
      );
    });

    test('predict throws ApiException on network error', () async {
      final xFile = XFile(_createTempImageFile().path);
      mockClient = MockClient((request) async {
        throw SocketException('Connection refused');
      });
      apiClient = ApiClient(client: mockClient);

      await expectLater(
        apiClient.predictXFile(xFile),
        throwsA(
          isA<ApiException>()
              .having((e) => e.statusCode, 'statusCode', 503)
              .having(
                (e) => e.message,
                'message',
                allOf(
                  contains('Cannot reach'),
                  isNot(contains('Connection to')),
                ),
              ),
        ),
      );
    });

    group('predictBytes (web)', () {
      test('predictBytes returns PredictionResponse on 200', () async {
        final bytes = Uint8List.fromList([
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
          0x08,
          0xD7,
          0x63,
          0xF8,
          0xCF,
          0xC0,
          0x00,
          0x00,
          0x03,
          0x01,
          0x01,
          0x00,
          0x18, 0xDD, 0x8D, 0xB4, // compressed data + CRC
          0x00, 0x00, 0x00, 0x00, // IEND chunk length
          0x49, 0x45, 0x4E, 0x44, // IEND
          0xAE, 0x42, 0x60, 0x82, // CRC
        ]);
        mockClient = MockClient((request) async {
          expect(request.url.toString(), endsWith('/predict'));
          expect(request.method, 'POST');
          return http.Response(
            jsonEncode({
              'predictions': [
                {'class_name': 'cat', 'confidence': 0.9},
                {'class_name': 'dog', 'confidence': 0.1},
              ],
              'latency_ms': 42.5,
              'model': 'cnn_scratch',
            }),
            200,
          );
        });
        apiClient = ApiClient(client: mockClient);

        final result = await apiClient.predictBytes(bytes, 'test_image.png');

        expect(result, isA<PredictionResponse>());
        expect(result.predictions, hasLength(2));
        expect(result.predictions.first.className, 'cat');
        expect(result.predictions.first.confidence, 0.9);
        expect(result.predictions.last.className, 'dog');
        expect(result.predictions.last.confidence, 0.1);
        expect(result.latencyMs, 42.5);
        expect(result.model, 'cnn_scratch');
      });

      test('predictBytes throws ApiException on 400', () async {
        final bytes = Uint8List.fromList([0x00]);
        mockClient = MockClient((request) async {
          return http.Response('{"detail": "file must be an image"}', 400);
        });
        apiClient = ApiClient(client: mockClient);

        await expectLater(
          apiClient.predictBytes(bytes, 'test.png'),
          throwsA(
            isA<ApiException>()
                .having((e) => e.statusCode, 'statusCode', 400)
                .having(
                  (e) => e.message,
                  'message',
                  contains('file must be an image'),
                ),
          ),
        );
      });

      test('predictBytes throws ApiException on 413', () async {
        final bytes = Uint8List.fromList([0x00]);
        mockClient = MockClient((request) async {
          return http.Response('{"detail": "image file too large"}', 413);
        });
        apiClient = ApiClient(client: mockClient);

        await expectLater(
          apiClient.predictBytes(bytes, 'test.png'),
          throwsA(
            isA<ApiException>()
                .having((e) => e.statusCode, 'statusCode', 413)
                .having(
                  (e) => e.message,
                  'message',
                  contains('image file too large'),
                ),
          ),
        );
      });

      test('predictBytes throws ApiException on 500', () async {
        final bytes = Uint8List.fromList([0x00]);
        mockClient = MockClient((request) async {
          return http.Response('{"detail": "model inference failed"}', 500);
        });
        apiClient = ApiClient(client: mockClient);

        await expectLater(
          apiClient.predictBytes(bytes, 'test.png'),
          throwsA(
            isA<ApiException>()
                .having((e) => e.statusCode, 'statusCode', 500)
                .having(
                  (e) => e.message,
                  'message',
                  contains('model inference failed'),
                ),
          ),
        );
      });

      test('predictBytes throws ApiException on timeout', () async {
        final bytes = Uint8List.fromList([0x00]);
        mockClient = MockClient((request) async {
          await Future.delayed(const Duration(seconds: 1));
          return http.Response('{}', 200);
        });
        apiClient = ApiClient(
          client: mockClient,
          timeout: const Duration(milliseconds: 50),
        );

        await expectLater(
          apiClient.predictBytes(bytes, 'test.png'),
          throwsA(
            isA<ApiException>()
                .having((e) => e.statusCode, 'statusCode', 408)
                .having((e) => e.message, 'message', contains('timed out')),
          ),
        );
      });

      test('predictBytes throws ApiException on network error', () async {
        final bytes = Uint8List.fromList([0x00]);
        mockClient = MockClient((request) async {
          throw SocketException('Connection refused');
        });
        apiClient = ApiClient(client: mockClient);

        await expectLater(
          apiClient.predictBytes(bytes, 'test.png'),
          throwsA(
            isA<ApiException>()
                .having((e) => e.statusCode, 'statusCode', 503)
                .having(
                  (e) => e.message,
                  'message',
                  allOf(
                    contains('Cannot reach'),
                    isNot(contains('Connection to')),
                  ),
                ),
          ),
        );
      });
    });

    group('image file read failures', () {
      test('an unreadable image file is reported as a file problem', () async {
        // The parent of the path is a regular file, so the path cannot be
        // opened on any platform. A permission bit would not work on a root
        // runner, and the path would then be readable after all.
        final blocker = _createTempImageFile();
        final file = File('${blocker.path}/image.png');

        final mockClient = MockClient((request) async {
          return http.Response('{}', 200);
        });
        final apiClient = ApiClient(client: mockClient);

        await expectLater(
          apiClient.predict(file),
          throwsA(
            isA<ApiException>()
                .having(
                  (e) => e.message,
                  'message',
                  allOf(
                    // The message names the problem, without the local path
                    // and without the raw exception text.
                    'Could not read the image file.',
                    // A file that cannot be read is not a connection problem.
                    isNot(contains('Connection to')),
                    isNot(contains('Cannot reach')),
                    // The local path of the rejected file stays out of the
                    // message that reaches the screen.
                    isNot(contains(file.path)),
                    isNot(contains(blocker.path)),
                  ),
                )
                .having((e) => e.statusCode, 'statusCode', 400),
          ),
        );
      });

      test(
        'an image file removed after the size check is a file problem',
        () async {
          final file = _createTempImageFile();
          final client = _DeletingClient(file);
          final apiClient = ApiClient(client: client);

          await expectLater(
            apiClient.predict(file),
            throwsA(
              isA<ApiException>()
                  .having(
                    (e) => e.message,
                    'message',
                    allOf(
                      'Could not read the image file.',
                      isNot(contains('Connection to')),
                      // The local path of the removed file stays out of the
                      // message that reaches the screen.
                      isNot(contains(file.path)),
                    ),
                  )
                  .having((e) => e.statusCode, 'statusCode', 400),
            ),
          );
        },
      );

      test('an unusable path is reported as a file problem', () async {
        final mockClient = MockClient((request) async {
          return http.Response('{}', 200);
        });
        final apiClient = ApiClient(client: mockClient);

        await expectLater(
          apiClient.predictXFile(XFile('')),
          throwsA(
            isA<ApiException>()
                .having(
                  (e) => e.message,
                  'message',
                  allOf(
                    'Could not read the image file.',
                    isNot(contains('Connection to')),
                  ),
                )
                .having((e) => e.statusCode, 'statusCode', 400),
          ),
        );
      });
    });

    group('response read failures', () {
      test('a truncated response body is not a connection error', () async {
        // The real transport reports a body that was cut short as a client
        // exception, not as a file system failure.
        final file = _createTempImageFile();
        final apiClient = ApiClient(
          client: _BrokenBodyClient(
            http.ClientException('Connection closed while receiving data.'),
          ),
        );

        await expectLater(
          apiClient.predict(file),
          throwsA(
            isA<ApiException>()
                .having(
                  (e) => e.message,
                  'message',
                  allOf(
                    contains('Could not read the response from'),
                    // The backend answered, so the connection was fine.
                    isNot(contains('Connection to')),
                    isNot(contains('Cannot reach')),
                    // The raw transport text must not reach the screen.
                    isNot(contains('Connection closed while receiving data')),
                  ),
                )
                .having((e) => e.statusCode, 'statusCode', 502),
          ),
        );
      });

      test(
        'a socket error while the body is read is not a connection error',
        () async {
          final file = _createTempImageFile();
          final apiClient = ApiClient(
            client: _BrokenBodyClient(
              SocketException('Connection reset by peer'),
            ),
          );

          await expectLater(
            apiClient.predict(file),
            throwsA(
              isA<ApiException>()
                  .having(
                    (e) => e.message,
                    'message',
                    allOf(
                      contains('Could not read the response from'),
                      isNot(contains('Connection to')),
                      isNot(contains('Cannot reach')),
                      isNot(contains('Connection reset by peer')),
                    ),
                  )
                  .having((e) => e.statusCode, 'statusCode', 502),
            ),
          );
        },
      );

      test(
        'a response body that cannot be read is not a connection error',
        () async {
          final file = _createTempImageFile();
          final apiClient = ApiClient(
            client: _BrokenBodyClient(
              FileSystemException(
                'the response body could not be read',
                '/tmp/image_classifier_response',
              ),
            ),
          );

          await expectLater(
            apiClient.predict(file),
            throwsA(
              isA<ApiException>()
                  .having(
                    (e) => e.message,
                    'message',
                    allOf(
                      contains('Could not read the response from'),
                      // The transfer failed, but the connection was
                      // established.
                      isNot(contains('Connection to')),
                      isNot(contains('Cannot reach')),
                    ),
                  )
                  .having((e) => e.statusCode, 'statusCode', 502),
            ),
          );
        },
      );

      test(
        'a response body that cannot be read fails the byte upload too',
        () async {
          final apiClient = ApiClient(
            client: _BrokenBodyClient(
              http.ClientException('Connection closed while receiving data.'),
            ),
          );

          await expectLater(
            apiClient.predictBytes(Uint8List.fromList([1, 2, 3]), 'image.png'),
            throwsA(
              isA<ApiException>()
                  .having(
                    (e) => e.message,
                    'message',
                    allOf(
                      contains('Could not read the response from'),
                      isNot(contains('Connection to')),
                    ),
                  )
                  .having((e) => e.statusCode, 'statusCode', 502),
            ),
          );
        },
      );
      test('a truncated model info body is not a connection error', () async {
        final apiClient = ApiClient(
          client: _BrokenBodyClient(
            http.ClientException('Connection closed while receiving data.'),
          ),
        );

        await expectLater(
          apiClient.getModelInfo(),
          throwsA(
            isA<ApiException>()
                .having(
                  (e) => e.message,
                  'message',
                  allOf(
                    contains('Could not read the response from'),
                    isNot(contains('Cannot reach')),
                    isNot(contains('Connection closed while receiving data')),
                  ),
                )
                .having((e) => e.statusCode, 'statusCode', 502),
          ),
        );
      });

      test(
        'a socket error while the bytes are read is not a connection error',
        () async {
          final apiClient = ApiClient(
            client: _BrokenBodyClient(
              SocketException('Connection reset by peer'),
            ),
          );

          await expectLater(
            apiClient.predictBytes(Uint8List.fromList([1, 2, 3]), 'image.png'),
            throwsA(
              isA<ApiException>()
                  .having(
                    (e) => e.message,
                    'message',
                    allOf(
                      contains('Could not read the response from'),
                      isNot(contains('Cannot reach')),
                      isNot(contains('Connection reset by peer')),
                    ),
                  )
                  .having((e) => e.statusCode, 'statusCode', 502),
            ),
          );
        },
      );

      test('a response body of the byte upload that cannot be read is not a '
          'connection error', () async {
        final apiClient = ApiClient(
          client: _BrokenBodyClient(
            FileSystemException(
              'the response body could not be read',
              '/tmp/image_classifier_response',
            ),
          ),
        );

        await expectLater(
          apiClient.predictBytes(Uint8List.fromList([1, 2, 3]), 'image.png'),
          throwsA(
            isA<ApiException>()
                .having(
                  (e) => e.message,
                  'message',
                  allOf(
                    contains('Could not read the response from'),
                    isNot(contains('Cannot reach')),
                  ),
                )
                .having((e) => e.statusCode, 'statusCode', 502),
          ),
        );
      });
    });

    group('a response that never ends', () {
      test('a stalled response body is reported as a timeout', () async {
        final body = StreamController<List<int>>();
        addTearDown(body.close);
        final file = _createTempImageFile();
        final apiClient = ApiClient(
          client: _StalledBodyClient(body),
          timeout: const Duration(milliseconds: 50),
        );

        await expectLater(
          apiClient.predict(file),
          throwsA(
            isA<ApiException>()
                .having((e) => e.statusCode, 'statusCode', 408)
                .having(
                  (e) => e.message,
                  'message',
                  allOf(
                    contains('timed out'),
                    isNot(contains('Connection to')),
                  ),
                ),
          ),
        );

        expect(
          body.hasListener,
          isFalse,
          reason:
              'the body subscription must be released when the read times out',
        );
      });

      test('a stalled response body is reported as a timeout for bytes too', () async {
        final body = StreamController<List<int>>();
        addTearDown(body.close);
        final apiClient = ApiClient(
          client: _StalledBodyClient(body),
          timeout: const Duration(milliseconds: 50),
        );

        await expectLater(
          apiClient.predictBytes(Uint8List.fromList([1, 2, 3]), 'image.png'),
          throwsA(
            isA<ApiException>()
                .having((e) => e.statusCode, 'statusCode', 408)
                .having(
                  (e) => e.message,
                  'message',
                  allOf(
                    contains('timed out'),
                    isNot(contains('Connection to')),
                  ),
                ),
          ),
        );

        expect(
          body.hasListener,
          isFalse,
          reason:
              'the body subscription must be released when the read times out',
        );
      });

      test('a stalled model info body is reported as a timeout', () async {
        final body = StreamController<List<int>>();
        addTearDown(body.close);
        final apiClient = ApiClient(
          client: _StalledBodyClient(body),
          timeout: const Duration(milliseconds: 50),
        );

        await expectLater(
          apiClient.getModelInfo(),
          throwsA(
            isA<ApiException>()
                .having((e) => e.statusCode, 'statusCode', 408)
                .having((e) => e.message, 'message', contains('timed out')),
          ),
        );

        expect(
          body.hasListener,
          isFalse,
          reason:
              'the body subscription must be released when the read times out',
        );
      });
    });

    group('a file fault while the request is sent', () {
      test(
        'a file fault while the bytes are sent is reported as a file problem',
        () async {
          final apiClient = ApiClient(client: _FileFaultOnSendClient());

          await expectLater(
            apiClient.predictBytes(Uint8List.fromList([1, 2, 3]), 'image.png'),
            throwsA(
              isA<ApiException>()
                  .having((e) => e.statusCode, 'statusCode', 400)
                  .having(
                    (e) => e.message,
                    'message',
                    'Could not read the image file.',
                  ),
            ),
          );
        },
      );
    });

    group('health status edge cases', () {
      test(
        'a malformed response is reported as an unreachable backend',
        () async {
          final malformedClient = MockClient((request) async {
            throw const FormatException('malformed response');
          });
          final apiClient = ApiClient(client: malformedClient);

          expect(await apiClient.healthStatusCode(), isNull);
        },
      );
    });

    test('dispose leaves a caller owned client usable', () async {
      mockClient = MockClient((request) async {
        return http.Response('{"status": "ok"}', 200);
      });
      apiClient = ApiClient(client: mockClient);

      await apiClient.dispose();

      // The caller owns the client, so disposing the ApiClient must not close
      // it.
      expect(await apiClient.healthCheck(), isTrue);
    });
  });

  group('PredictionItem', () {
    test('fromJson creates correct object', () {
      final json = {'class_name': 'cat', 'confidence': 0.95};
      final item = PredictionItem.fromJson(json);

      expect(item.className, 'cat');
      expect(item.confidence, 0.95);
    });

    test('toJson returns correct map', () {
      const item = PredictionItem(className: 'dog', confidence: 0.8);
      final json = item.toJson();

      expect(json, {'class_name': 'dog', 'confidence': 0.8});
    });
  });

  group('PredictionResponse', () {
    test('fromJson creates correct object with multiple predictions', () {
      final json = {
        'predictions': [
          {'class_name': 'cat', 'confidence': 0.9},
          {'class_name': 'dog', 'confidence': 0.1},
        ],
        'latency_ms': 42.0,
        'model': 'cnn_scratch',
      };
      final response = PredictionResponse.fromJson(json);

      expect(response.predictions, hasLength(2));
      expect(response.latencyMs, 42.0);
      expect(response.model, 'cnn_scratch');
    });

    test('toJson returns correct map', () {
      const response = PredictionResponse(
        predictions: [
          PredictionItem(className: 'cat', confidence: 0.9),
          PredictionItem(className: 'dog', confidence: 0.1),
        ],
        latencyMs: 42.0,
        model: 'cnn_scratch',
      );
      final json = response.toJson();

      expect(json['latency_ms'], 42.0);
      expect(json['model'], 'cnn_scratch');
      expect(json['predictions'], isA<List>());
    });
  });

  group('ModelInfo', () {
    test('fromJson creates correct object with trained_at', () {
      final json = {
        'model': 'cnn_scratch',
        'input_size': '32x32',
        'classes': ['cat', 'dog'],
        'device': 'cpu',
        'trained_at': '2026-10-29T10:00:00.000Z',
      };
      final info = ModelInfo.fromJson(json);

      expect(info.modelName, 'cnn_scratch');
      expect(info.inputSize, '32x32');
      expect(info.classes, ['cat', 'dog']);
      expect(info.device, 'cpu');
      expect(info.trainedAt, isNotNull);
      expect(info.trainedAt!.isUtc, isTrue);
    });

    test('fromJson handles null trained_at', () {
      final json = {
        'model': 'cnn_scratch',
        'input_size': '32x32',
        'classes': ['cat', 'dog'],
        'device': 'cpu',
        'trained_at': null,
      };
      final info = ModelInfo.fromJson(json);

      expect(info.trainedAt, isNull);
    });

    test('fromJson preserves UTC instant on roundtrip (IC-58)', () {
      final json = {
        'model': 'cnn_scratch',
        'input_size': '32x32',
        'classes': ['cat', 'dog'],
        'device': 'cpu',
        'trained_at': '2026-10-29T10:00:00.000Z',
      };
      final info = ModelInfo.fromJson(json);
      final roundtripped = ModelInfo.fromJson(info.toJson());

      // The serialized instant must be timezone-independent: local
      // conversion shifted it by the machine offset before IC-58.
      expect(info.toJson()['trained_at'], '2026-10-29T10:00:00.000Z');
      expect(roundtripped.trainedAt, info.trainedAt);
      expect(roundtripped.trainedAt!.isUtc, isTrue);
    });

    test('toJson returns correct map', () {
      final info = ModelInfo(
        modelName: 'cnn_scratch',
        inputSize: '32x32',
        classes: ['cat', 'dog'],
        device: 'cpu',
        trainedAt: DateTime.parse('2026-10-29T10:00:00.000Z'),
      );
      final json = info.toJson();

      expect(json['model'], 'cnn_scratch');
      expect(json['input_size'], '32x32');
      expect(json['trained_at'], '2026-10-29T10:00:00.000Z');
    });
  });

  group('ApiException', () {
    test('includes status code and message', () {
      const exception = ApiException(
        message: 'model unavailable',
        statusCode: 503,
      );

      expect(exception.message, 'model unavailable');
      expect(exception.statusCode, 503);
      expect(exception.toString(), contains('503'));
      expect(exception.toString(), contains('model unavailable'));
    });
  });

  group('PredictionItem equality & hashCode & toString', () {
    const item1 = PredictionItem(className: 'cat', confidence: 0.95);
    const item2 = PredictionItem(className: 'cat', confidence: 0.95);
    const item3 = PredictionItem(className: 'dog', confidence: 0.95);
    const item4 = PredictionItem(className: 'cat', confidence: 0.8);

    test('identical instances are equal', () {
      expect(item1 == item2, isTrue);
      expect(item1.hashCode, item2.hashCode);
    });

    test('different className are not equal', () {
      expect(item1 == item3, isFalse);
    });

    test('different confidence are not equal', () {
      expect(item1 == item4, isFalse);
    });

    test('different runtimeType returns false', () {
      expect(item1 == 'not a prediction' as Object, isFalse);
    });

    test('toString contains className and confidence', () {
      const item = PredictionItem(className: 'cat', confidence: 0.95);
      expect(item.toString(), contains('cat'));
      expect(item.toString(), contains('0.95'));
    });
  });

  group('PredictionItem fromJson error handling', () {
    test('throws on missing class_name', () {
      expect(
        () => PredictionItem.fromJson({'confidence': 0.5}),
        throwsA(isA<TypeError>()),
      );
    });

    test('throws on missing confidence', () {
      expect(
        () => PredictionItem.fromJson({'class_name': 'cat'}),
        throwsA(isA<TypeError>()),
      );
    });

    test('throws when confidence is not a number', () {
      expect(
        () => PredictionItem.fromJson({
          'class_name': 'cat',
          'confidence': 'high',
        }),
        throwsA(isA<TypeError>()),
      );
    });

    test('fromJson keeps in-range confidence unchanged', () {
      final item = PredictionItem.fromJson({
        'class_name': 'cat',
        'confidence': 0.42,
      });
      expect(item.confidence, 0.42);
    });

    test('asserts out-of-range confidence in debug mode (IC-59)', () {
      // Debug builds (including flutter test) surface server bugs loudly
      // via the assert; release builds keep the defensive clamp.
      expect(
        () => PredictionItem.fromJson({'class_name': 'cat', 'confidence': 1.5}),
        throwsA(isA<AssertionError>()),
      );
      expect(
        () =>
            PredictionItem.fromJson({'class_name': 'cat', 'confidence': -0.5}),
        throwsA(isA<AssertionError>()),
      );
    });
  });

  group('PredictionResponse equality & hashCode & toString', () {
    const pred1 = PredictionItem(className: 'cat', confidence: 0.9);
    const pred2 = PredictionItem(className: 'dog', confidence: 0.1);
    const resp1 = PredictionResponse(
      predictions: [pred1, pred2],
      latencyMs: 42.0,
      model: 'cnn_scratch',
    );
    const resp2 = PredictionResponse(
      predictions: [pred1, pred2],
      latencyMs: 42.0,
      model: 'cnn_scratch',
    );
    const resp3 = PredictionResponse(
      predictions: [pred2, pred1],
      latencyMs: 42.0,
      model: 'cnn_scratch',
    );
    const resp4 = PredictionResponse(
      predictions: [pred1],
      latencyMs: 42.0,
      model: 'cnn_scratch',
    );
    const resp5 = PredictionResponse(
      predictions: [pred1, pred2],
      latencyMs: 43.0,
      model: 'cnn_scratch',
    );
    const resp6 = PredictionResponse(
      predictions: [pred1, pred2],
      latencyMs: 42.0,
      model: 'resnet18',
    );

    test('identical instances are equal', () {
      expect(resp1 == resp2, isTrue);
      expect(resp1.hashCode, resp2.hashCode);
    });

    test('different predictions order are not equal', () {
      expect(resp1 == resp3, isFalse);
    });

    test('different predictions list are not equal', () {
      expect(resp1 == resp4, isFalse);
    });

    test('different latencyMs are not equal', () {
      expect(resp1 == resp5, isFalse);
    });

    test('different model are not equal', () {
      expect(resp1 == resp6, isFalse);
    });

    test('different runtimeType returns false', () {
      expect(resp1 == 'not a response' as Object, isFalse);
    });

    test('toString contains all fields', () {
      expect(resp1.toString(), contains('cat'));
      expect(resp1.toString(), contains('42.0'));
      expect(resp1.toString(), contains('cnn_scratch'));
    });
  });

  group('PredictionResponse fromJson error handling', () {
    test('handles empty predictions list', () {
      final json = {'predictions': [], 'latency_ms': 10.0, 'model': 'test'};
      final r = PredictionResponse.fromJson(json);
      expect(r.predictions, isEmpty);
    });

    test('throws on invalid prediction item', () {
      final json = {
        'predictions': [
          {'invalid': 'item'},
        ],
        'latency_ms': 10.0,
        'model': 'test',
      };
      expect(
        () => PredictionResponse.fromJson(json),
        throwsA(isA<TypeError>()),
      );
    });

    test('throws on missing latency_ms', () {
      final json = {'predictions': [], 'model': 'test'};
      expect(
        () => PredictionResponse.fromJson(json),
        throwsA(isA<TypeError>()),
      );
    });

    test('throws on missing model', () {
      final json = {'predictions': [], 'latency_ms': 10.0};
      expect(
        () => PredictionResponse.fromJson(json),
        throwsA(isA<TypeError>()),
      );
    });
  });

  group('ModelInfo equality & hashCode & toString', () {
    final info1 = ModelInfo(
      modelName: 'cnn_scratch',
      inputSize: '32x32',
      classes: ['cat', 'dog'],
      device: 'cpu',
      trainedAt: DateTime(2026, 10, 29, 10, 0, 0),
    );
    final info2 = ModelInfo(
      modelName: 'cnn_scratch',
      inputSize: '32x32',
      classes: ['cat', 'dog'],
      device: 'cpu',
      trainedAt: DateTime(2026, 10, 29, 10, 0, 0),
    );
    final info3 = ModelInfo(
      modelName: 'resnet18',
      inputSize: '32x32',
      classes: ['cat', 'dog'],
      device: 'cpu',
      trainedAt: DateTime(2026, 10, 29, 10, 0, 0),
    );
    final info4 = ModelInfo(
      modelName: 'cnn_scratch',
      inputSize: '224x224',
      classes: ['cat', 'dog'],
      device: 'cpu',
      trainedAt: DateTime(2026, 10, 29, 10, 0, 0),
    );
    final info5 = ModelInfo(
      modelName: 'cnn_scratch',
      inputSize: '32x32',
      classes: ['dog', 'cat'],
      device: 'cpu',
      trainedAt: DateTime(2026, 10, 29, 10, 0, 0),
    );
    final info6 = ModelInfo(
      modelName: 'cnn_scratch',
      inputSize: '32x32',
      classes: ['cat', 'dog'],
      device: 'cuda',
      trainedAt: DateTime(2026, 10, 29, 10, 0, 0),
    );
    final info7 = ModelInfo(
      modelName: 'cnn_scratch',
      inputSize: '32x32',
      classes: ['cat', 'dog'],
      device: 'cpu',
      trainedAt: null,
    );

    test('identical instances are equal', () {
      expect(info1 == info2, isTrue);
      expect(info1.hashCode, info2.hashCode);
    });

    test('different modelName are not equal', () {
      expect(info1 == info3, isFalse);
    });

    test('different inputSize are not equal', () {
      expect(info1 == info4, isFalse);
    });

    test('different classes order are not equal', () {
      expect(info1 == info5, isFalse);
    });

    test('different device are not equal', () {
      expect(info1 == info6, isFalse);
    });

    test('null trainedAt vs non-null are not equal', () {
      expect(info1 == info7, isFalse);
    });

    test('different runtimeType returns false', () {
      expect(info1 == 'not a model info' as Object, isFalse);
    });

    test('toString contains all fields', () {
      expect(info1.toString(), contains('cnn_scratch'));
      expect(info1.toString(), contains('32x32'));
      expect(info1.toString(), contains('cpu'));
    });
  });

  group('ModelInfo fromJson error handling', () {
    test('throws on missing required fields', () {
      expect(() => ModelInfo.fromJson({}), throwsA(isA<TypeError>()));
      expect(
        () => ModelInfo.fromJson({'model': 'x'}),
        throwsA(isA<TypeError>()),
      );
    });

    test('throws when classes is not a list', () {
      final json = {
        'model': 'x',
        'input_size': '32x32',
        'classes': 'not-a-list',
        'device': 'cpu',
      };
      expect(() => ModelInfo.fromJson(json), throwsA(isA<TypeError>()));
    });

    test('handles invalid trained_at format gracefully', () {
      final json = {
        'model': 'x',
        'input_size': '32x32',
        'classes': [],
        'device': 'cpu',
        'trained_at': 'not-a-date',
      };
      final info = ModelInfo.fromJson(json);
      expect(info.trainedAt, isNull);
    });

    test('handles missing trained_at gracefully', () {
      final json = {
        'model': 'x',
        'input_size': '32x32',
        'classes': [],
        'device': 'cpu',
      };
      final info = ModelInfo.fromJson(json);
      expect(info.trainedAt, isNull);
    });
  });
}
