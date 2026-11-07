import 'dart:async';

import 'package:shared_preferences_platform_interface/shared_preferences_platform_interface.dart';

/// A preferences store whose first read completes only when [release] is called.
///
/// The settings screen loads the stored API base URL asynchronously while it
/// is being built. This store makes that read controllable, so a test can type
/// into the field *before* the stored URL arrives and observe whether the
/// screen clobbers the typed text.
///
/// Keys must carry the `flutter.` prefix: the shared_preferences legacy API
/// prefixes every key it writes and strips that prefix from every key it
/// reads back.
class DelayedPrefsStore extends SharedPreferencesStorePlatform {
  DelayedPrefsStore(this._values);

  final Map<String, Object> _values;
  final Completer<void> _gate = Completer<void>();
  bool _released = false;

  /// Lets the pending read complete with the stored values.
  void release() {
    if (_released) return;
    _released = true;
    _gate.complete();
  }

  @override
  Future<Map<String, Object>> getAll() async {
    if (!_released) {
      await _gate.future;
    }
    return Map<String, Object>.from(_values);
  }

  @override
  Future<bool> setValue(String valueType, String key, Object value) async {
    _values[key] = value;
    return true;
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
