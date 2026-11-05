import 'package:shared_preferences_platform_interface/shared_preferences_platform_interface.dart';

/// A preferences store that refuses writes but keeps the stored values.
///
/// Simulates a platform whose write failed while the previously stored
/// values are still intact, so the app has to keep using the backend
/// address that is actually persisted.
///
/// Keys carry the `flutter.` prefix: the shared_preferences legacy API
/// prefixes every key it writes and strips that prefix from every key it
/// reads back.
class RefusingWritePrefsStore extends SharedPreferencesStorePlatform {
  RefusingWritePrefsStore(Map<String, Object> values)
    : _values = Map.of(values);

  final Map<String, Object> _values;

  @override
  Future<Map<String, Object>> getAll() async {
    return Map<String, Object>.of(_values);
  }

  /// The write is refused and the stored value is left untouched.
  @override
  Future<bool> setValue(String valueType, String key, Object value) async {
    return false;
  }

  @override
  Future<bool> remove(String key) async {
    _values.remove(key);
    return true;
  }

  @override
  Future<bool> clear() async {
    _values.clear();
    return true;
  }
}
