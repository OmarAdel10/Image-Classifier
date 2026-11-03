import 'dart:async';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:image_classifier/config.dart';
import 'package:image_classifier/services/api_client.dart';
import 'package:shared_preferences/shared_preferences.dart';

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
    });
  });
}
