import 'package:shared_preferences_platform_interface/shared_preferences_platform_interface.dart';

/// A preferences store whose reads succeed but whose removal throws.
///
/// Simulates a platform that fails while removing a stored value, so the
/// removal-failure path of `Config.clearApiBaseUrl` is exercised without
/// touching real platform storage.
///
/// Keys must carry the `flutter.` prefix: the shared_preferences legacy API
/// prefixes every key it writes and strips that prefix from every key it
/// reads back.
class ThrowingRemovingPrefsStore extends SharedPreferencesStorePlatform {
  ThrowingRemovingPrefsStore(this._values);

  final Map<String, Object> _values;

  @override
  Future<Map<String, Object>> getAll() async {
    return Map<String, Object>.from(_values);
  }

  @override
  Future<bool> setValue(String valueType, String key, Object value) async {
    _values[key] = value;
    return true;
  }

  @override
  Future<bool> remove(String key) async {
    throw StateError('removal failed');
  }

  @override
  Future<bool> clear() async {
    _values.clear();
    return true;
  }
}
