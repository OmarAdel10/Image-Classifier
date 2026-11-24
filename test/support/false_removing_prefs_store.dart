import 'package:shared_preferences_platform_interface/shared_preferences_platform_interface.dart';

/// A preferences store that holds data but refuses to remove it.
///
/// Simulates a platform that keeps a stored value even when the removal is
/// reported as failed, so the reset path can be exercised without touching
/// real platform storage.
///
/// Keys must carry the `flutter.` prefix: the shared_preferences legacy API
/// prefixes every key it writes and strips that prefix from every key it
/// reads back.
class FalseRemovingPrefsStore extends SharedPreferencesStorePlatform {
  FalseRemovingPrefsStore(this._values);

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
    return false;
  }

  @override
  Future<bool> clear() async {
    _values.clear();
    return true;
  }
}
