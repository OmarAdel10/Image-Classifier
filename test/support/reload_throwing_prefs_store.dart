import 'package:shared_preferences_platform_interface/shared_preferences_platform_interface.dart';

/// A preferences store that becomes unreadable after the first read.
///
/// The first read returns the stored values, so the app starts normally.
/// Every later read throws, which models a platform that cannot be read any
/// more while the app is running - for example when the rollback that
/// restores the persisted state fails to read the store.
///
/// Keys carry the `flutter.` prefix: the shared_preferences legacy API
/// prefixes every key it writes and strips that prefix from every key it
/// reads back.
class ReloadThrowingPrefsStore extends SharedPreferencesStorePlatform {
  ReloadThrowingPrefsStore(Map<String, Object> values)
    : _values = Map.of(values);

  final Map<String, Object> _values;
  int _reads = 0;

  @override
  Future<Map<String, Object>> getAll() async {
    _reads++;
    if (_reads > 1) {
      throw StateError('storage unavailable');
    }
    return Map<String, Object>.of(_values);
  }

  /// The write is refused, like a platform that reports a failed write.
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
