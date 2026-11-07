import 'package:shared_preferences_platform_interface/shared_preferences_platform_interface.dart';

/// A preferences store whose reads succeed but whose writes throw.
///
/// Simulates a platform that accepts the initial load but fails a write, so
/// the write-failure path of `Config.setApiBaseUrl` is exercised without
/// touching real platform storage.
class ThrowingWritePrefsStore extends InMemorySharedPreferencesStore {
  ThrowingWritePrefsStore() : super.empty();

  @override
  Future<bool> setValue(String valueType, String key, Object value) async {
    throw StateError('write failed');
  }
}
