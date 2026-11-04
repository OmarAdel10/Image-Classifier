import 'package:shared_preferences_platform_interface/shared_preferences_platform_interface.dart';

/// A preferences store whose every operation fails.
///
/// Simulates a platform where the shared_preferences plugin is unavailable so
/// the storage-failure fallbacks in `Config` and the settings screen can be
/// exercised without touching real platform storage.
class ThrowingPrefsStore extends InMemorySharedPreferencesStore {
  ThrowingPrefsStore() : super.empty();

  @override
  Future<Map<String, Object>> getAll() async {
    throw StateError('storage unavailable');
  }

  @override
  Future<bool> setValue(String valueType, String key, Object value) async {
    throw StateError('storage unavailable');
  }

  @override
  Future<bool> remove(String key) async {
    throw StateError('storage unavailable');
  }
}
