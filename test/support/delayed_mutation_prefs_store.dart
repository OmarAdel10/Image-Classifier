import 'dart:async';

import 'package:shared_preferences_platform_interface/shared_preferences_platform_interface.dart';

/// A preferences store whose writes and removals complete only when released.
///
/// Simulates a platform that answers a save or a reset long after the screen
/// that asked for it has been closed, so the unmount guards of the settings
/// screen can be exercised without touching real platform storage.
///
/// Keys must carry the `flutter.` prefix: the shared_preferences legacy API
/// prefixes every key it writes and strips that prefix from every key it
/// reads back.
class DelayedMutationPrefsStore extends SharedPreferencesStorePlatform {
  DelayedMutationPrefsStore(this._values);

  final Map<String, Object> _values;
  final Completer<void> _writeGate = Completer<void>();
  final Completer<void> _removalGate = Completer<void>();
  bool _writeReleased = false;
  bool _removalReleased = false;

  /// Lets the pending write complete.
  void releaseWrite() {
    if (_writeReleased) return;
    _writeReleased = true;
    _writeGate.complete();
  }

  /// Lets the pending removal complete.
  void releaseRemoval() {
    if (_removalReleased) return;
    _removalReleased = true;
    _removalGate.complete();
  }

  @override
  Future<Map<String, Object>> getAll() async {
    return Map<String, Object>.from(_values);
  }

  @override
  Future<bool> setValue(String valueType, String key, Object value) async {
    if (!_writeReleased) {
      await _writeGate.future;
    }
    _values[key] = value;
    return true;
  }

  @override
  Future<bool> remove(String key) async {
    if (!_removalReleased) {
      await _removalGate.future;
    }
    _values.remove(key);
    return true;
  }

  @override
  Future<bool> clear() async {
    _values.clear();
    return true;
  }
}
