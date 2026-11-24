import 'package:shared_preferences_platform_interface/shared_preferences_platform_interface.dart';

/// A preferences store whose writes always fail and whose reads fail after
/// the first one.
///
/// The first read succeeds, so `SharedPreferences.getInstance()` can load the
/// seeded values. Every later read throws, which is what the `reload()`
/// inside the rollback of a refused write meets when the platform has become
/// unreadable. The writes throw as well, so the write-back that restores the
/// previously stored address fails too.
class ThrowingWriteBackPrefsStore extends SharedPreferencesStorePlatform {
  ThrowingWriteBackPrefsStore(this._values);

  final Map<String, Object> _values;
  bool _read = true;

  @override
  Future<Map<String, Object>> getAll() async {
    if (!_read) {
      throw StateError('storage unavailable');
    }
    _read = false;
    return Map<String, Object>.from(_values);
  }

  @override
  Future<bool> setValue(String valueType, String key, Object value) async {
    throw StateError('storage unavailable');
  }

  @override
  Future<bool> remove(String key) async {
    throw StateError('storage unavailable');
  }

  @override
  Future<bool> clear() async {
    throw StateError('storage unavailable');
  }
}
