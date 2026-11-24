import 'package:shared_preferences_platform_interface/shared_preferences_platform_interface.dart';

/// A preferences store that accepts writes but reports failure.
///
/// Simulates a platform plugin whose setValue/remove calls complete without
/// throwing yet return false, exercising the `!stored` guard in
/// `Config.setApiBaseUrl` without touching real platform storage.
class FalseReturningPrefsStore extends InMemorySharedPreferencesStore {
  FalseReturningPrefsStore() : super.empty();

  @override
  Future<bool> setValue(String valueType, String key, Object value) async {
    return false;
  }

  @override
  Future<bool> remove(String key) async {
    return false;
  }
}
