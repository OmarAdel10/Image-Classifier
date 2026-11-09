import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:image_classifier/config.dart';
import 'package:image_classifier/services/api_client.dart';

/// Shortens an address for a banner.
///
/// The field accepts anything the user types, so the address is stripped of
/// control characters and shortened before it reaches a message. A banner
/// stays readable and cannot carry a pasted payload. The credentials of an
/// address are removed as well, so a banner never repeats a secret.
String _displayUrl(String url) {
  final cleaned = url
      .replaceAll(RegExp(r'[\x00-\x1F\x7F]'), '')
      .replaceFirstMapped(
        RegExp(r'^(https?://)[^/@]*@'),
        (match) => match.group(1)!,
      );
  if (cleaned.length <= 120) return cleaned;
  return '${cleaned.substring(0, 119)}…';
}

/// Screen for configuring the backend API base URL at runtime.
///
/// The compile-time `--dart-define=API_BASE_URL` override always wins. When it
/// is present the field is used only for Test Connection and Save is disabled,
/// so a runtime value cannot be stored and later activate in an unpinned build.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({
    super.key,
    this.compileTimeOverride = Config.compileTimeApiBaseUrl,
  });

  /// Compile-time `--dart-define=API_BASE_URL` value in effect for this app.
  ///
  /// Defaults to [Config.compileTimeApiBaseUrl]; tests inject a value to
  /// verify the pinned-build behavior.
  final String compileTimeOverride;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _controller;
  bool _testing = false;
  String? _statusMessage;
  bool _statusIsError = false;

  bool get _pinned => widget.compileTimeOverride.isNotEmpty;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController();
    _loadCurrentUrl();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  Future<void> _loadCurrentUrl() async {
    final url = await Config.resolveApiBaseUrl(
      compileTimeOverride: widget.compileTimeOverride,
    );
    if (!mounted) return;
    // Do not clobber anything the user typed while storage was resolving.
    if (_controller.text.isEmpty) {
      _controller.text = url;
    }
    setState(() {});
  }

  String? _validate(String value) {
    if (value.trim().isEmpty) {
      return 'Enter a URL such as http://localhost:8000';
    }
    if (!Config.isValidApiBaseUrl(value.trim())) {
      return 'Not a valid URL — use http:// or https:// with a host';
    }
    return null;
  }

  /// Renders an error for the user without the raw Dart exception wrapper.
  ///
  /// [StateError] carries a message that is written for humans, but
  /// `toString()` prefixes it with 'Bad state:'. A missing provider is
  /// reported with its own text, because it names the provider the screen
  /// needs. Every other error is reported as an unexpected failure: its own
  /// text can carry internal details that do not belong on the screen.
  static String _describeError(Object error) {
    if (error is StateError) return error.message;
    if (error is ProviderNotFoundException) return error.toString();
    return 'an unexpected error occurred';
  }

  Future<void> _testConnection() async {
    final url = Config.normalizeApiBaseUrl(_controller.text);

    setState(() {
      _testing = true;
      _statusMessage = null;
    });

    try {
      // Read the client inside the try so a missing provider is reported to
      // the user instead of escaping the handler. No BuildContext is used
      // after an await.
      final httpClient = context.read<ApiClient>().client;

      // Probe the URL as typed, without disturbing the stored configuration.
      final probe = ApiClient(client: httpClient, baseUrl: url);
      final statusCode = await probe.healthStatusCode();

      if (!mounted) return;
      setState(() {
        if (statusCode == 200) {
          _statusIsError = false;
          _statusMessage = 'Connected to ${_displayUrl(url)}';
        } else if (statusCode != null) {
          _statusIsError = true;
          _statusMessage =
              'Reached ${_displayUrl(url)}, but /health returned HTTP '
              '$statusCode. '
              'The backend is running but not healthy.';
        } else {
          _statusIsError = true;
          _statusMessage =
              'Could not reach ${_displayUrl(url)} — is the backend running?';
        }
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _statusIsError = true;
        // The probe never raises an ApiException: healthStatusCode() reports
        // an unreachable backend as a null status code instead of throwing.
        // The generic renderer therefore names the actual cause.
        _statusMessage =
            'Could not test ${_displayUrl(url)}: ${_describeError(e)}';
      });
    } finally {
      if (mounted) {
        setState(() => _testing = false);
      }
    }
  }

  Future<void> _save() async {
    final url = Config.normalizeApiBaseUrl(_controller.text);
    if (_validate(url) != null) return;

    final override = widget.compileTimeOverride;

    ApiClient? apiClient;
    try {
      // Read the client inside the try so a missing provider is reported to
      // the user instead of escaping the handler.
      apiClient = context.read<ApiClient>();
      await Config.setApiBaseUrl(url, compileTimeOverride: override);
    } catch (e) {
      // Config restores the persisted URL after a refused write, so the
      // client cache is dropped here as well: the next request has to
      // re-resolve the URL that is actually stored.
      apiClient?.invalidateBaseUrlCache();
      if (!mounted) return;
      setState(() {
        _statusIsError = true;
        _statusMessage =
            'Could not save ${_displayUrl(url)}: ${_describeError(e)}';
      });
      return;
    }

    apiClient.invalidateBaseUrlCache();
    if (!mounted) return;
    setState(() {
      _statusIsError = false;
      _statusMessage = 'Saved. Using ${_displayUrl(url)}';
    });
  }

  Future<void> _resetToDefault() async {
    late final ApiClient apiClient;
    try {
      // Read the client inside the try so a missing provider is reported to
      // the user instead of escaping the handler.
      apiClient = context.read<ApiClient>();
    } catch (e) {
      // The read is synchronous, so the screen is still mounted here and the
      // banner can be written without a guard.
      setState(() {
        _statusIsError = true;
        _statusMessage = 'Could not reset the stored URL: ${_describeError(e)}';
      });
      return;
    }

    final cleared = await Config.clearApiBaseUrl();
    apiClient.invalidateBaseUrlCache();
    if (!mounted) return;
    if (!cleared) {
      setState(() {
        _statusIsError = true;
        // The in-memory cache is cleared even when the platform refused the
        // removal, so the stored URL is not active now - but the platform may
        // still hold it and restore it after a restart.
        _statusMessage =
            'Could not remove the stored URL - it is not active now, '
            'but it may return after a restart.';
      });
      return;
    }
    final url = await Config.resolveApiBaseUrl(
      compileTimeOverride: widget.compileTimeOverride,
    );
    if (!mounted) return;
    setState(() {
      final override = widget.compileTimeOverride;
      _statusIsError = false;
      _statusMessage = override.isNotEmpty
          ? 'Cleared the stored URL; --dart-define=API_BASE_URL='
                '${_displayUrl(override)} still applies.'
          : 'Reset to default (${_displayUrl(url)})';
    });
    _controller.text = url;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final errorText = _validate(_controller.text);

    return Scaffold(
      appBar: AppBar(title: const Text('Backend Settings')),
      body: ListView(
        padding: const EdgeInsets.all(24),
        children: [
          Text('API Base URL', style: theme.textTheme.titleMedium),
          const SizedBox(height: 8),
          Text(
            'Address of the FastAPI backend serving /health, /model-info '
            'and /predict.',
            style: theme.textTheme.bodySmall,
          ),
          if (_pinned) ...[
            const SizedBox(height: 12),
            _PinnedBanner(compileTimeOverride: widget.compileTimeOverride),
          ],
          const SizedBox(height: 16),
          TextField(
            controller: _controller,
            onChanged: (_) => setState(() {
              // The banner described the previous value; do not keep claiming
              // it is connected while the field shows something else.
              _statusMessage = null;
            }),
            keyboardType: TextInputType.url,
            autocorrect: false,
            decoration: InputDecoration(
              border: const OutlineInputBorder(),
              hintText: 'http://localhost:8000',
              errorText: errorText,
            ),
          ),
          const SizedBox(height: 16),
          _StatusBanner(message: _statusMessage, isError: _statusIsError),
          const SizedBox(height: 16),
          Wrap(
            spacing: 12,
            runSpacing: 12,
            children: [
              FilledButton.icon(
                onPressed: (_testing || errorText != null)
                    ? null
                    : _testConnection,
                icon: _testing
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Icon(Icons.wifi_tethering),
                label: const Text('Test Connection'),
              ),
              FilledButton.icon(
                onPressed: (errorText == null && !_pinned) ? _save : null,
                icon: const Icon(Icons.save),
                label: const Text('Save'),
              ),
              OutlinedButton.icon(
                onPressed: _resetToDefault,
                icon: const Icon(Icons.restart_alt),
                label: const Text('Reset to Default'),
              ),
            ],
          ),
          const SizedBox(height: 32),
          _PlatformHint(theme: theme),
        ],
      ),
    );
  }
}

class _StatusBanner extends StatelessWidget {
  const _StatusBanner({required this.message, required this.isError});

  final String? message;
  final bool isError;

  @override
  Widget build(BuildContext context) {
    if (message == null) return const SizedBox.shrink();
    final theme = Theme.of(context);
    final color = isError ? theme.colorScheme.error : theme.colorScheme.primary;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.10),
        border: Border.all(color: color.withValues(alpha: 0.40)),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          Icon(
            isError ? Icons.error_outline : Icons.check_circle_outline,
            color: color,
            size: 20,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              message!,
              style: theme.textTheme.bodyMedium?.copyWith(color: color),
            ),
          ),
        ],
      ),
    );
  }
}

class _PinnedBanner extends StatelessWidget {
  const _PinnedBanner({required this.compileTimeOverride});

  final String compileTimeOverride;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final color = theme.colorScheme.primary;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.10),
        border: Border.all(color: color.withValues(alpha: 0.40)),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          Icon(Icons.lock_outline, color: color, size: 20),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              'Pinned by --dart-define=API_BASE_URL='
              '${_displayUrl(compileTimeOverride)}. '
              'This build always uses that URL; the field below is for Test '
              'Connection only and Save is disabled.',
              style: theme.textTheme.bodyMedium?.copyWith(color: color),
            ),
          ),
        ],
      ),
    );
  }
}

class _PlatformHint extends StatelessWidget {
  const _PlatformHint({required this.theme});

  final ThemeData theme;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Reachability notes', style: theme.textTheme.titleSmall),
        const SizedBox(height: 8),
        Text(
          '• Linux/Windows/macOS desktop: http://localhost:8000 reaches the '
          'backend on this machine.\n'
          '• Android emulator: use http://10.0.2.2:8000 — localhost points '
          'at the emulator itself, not your computer.\n'
          '• Physical phone/tablet: use your computer\'s LAN address, for '
          'example http://192.168.1.10:8000, and start the backend bound to '
          '0.0.0.0.',
          style: theme.textTheme.bodySmall,
        ),
      ],
    );
  }
}
