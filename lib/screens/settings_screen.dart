import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:image_classifier/config.dart';
import 'package:image_classifier/services/api_client.dart';

/// Screen for configuring the backend API base URL at runtime.
///
/// The compile-time `--dart-define=API_BASE_URL` override always wins; the
/// value entered here is used only when no compile-time override is present.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({
    super.key,
    this.compileTimeOverride = Config.compileTimeApiBaseUrl,
  });

  /// Compile-time `--dart-define=API_BASE_URL` value in effect for this app.
  ///
  /// Defaults to [Config.compileTimeApiBaseUrl]; tests inject a value to
  /// verify that the screen warns when the runtime value is overridden.
  final String compileTimeOverride;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _controller;
  bool _testing = false;
  String? _statusMessage;
  bool _statusIsError = false;

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
    _controller.text = url;
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

  Future<void> _testConnection() async {
    final url = _controller.text.trim();

    setState(() {
      _testing = true;
      _statusMessage = null;
    });

    // Capture the client before the async gap so no BuildContext is used
    // after an await.
    final httpClient = context.read<ApiClient>().client;

    // Probe the URL as typed, without disturbing the stored configuration.
    final probe = ApiClient(client: httpClient, baseUrl: url);
    final statusCode = await probe.healthStatusCode();

    if (!mounted) return;
    setState(() {
      _testing = false;
      if (statusCode == 200) {
        _statusIsError = false;
        _statusMessage = 'Connected to $url';
      } else if (statusCode != null) {
        _statusIsError = true;
        _statusMessage =
            'Reached $url, but /health returned HTTP $statusCode. '
            'The backend is running but not healthy.';
      } else {
        _statusIsError = true;
        _statusMessage = 'Could not reach $url — is the backend running?';
      }
    });
  }

  Future<void> _save() async {
    final url = Config.normalizeApiBaseUrl(_controller.text);
    if (_validate(url) != null) return;

    final apiClient = context.read<ApiClient>();

    try {
      await Config.setApiBaseUrl(url);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _statusIsError = true;
        _statusMessage = 'Could not save $url: $e';
      });
      return;
    }

    apiClient.invalidateBaseUrlCache();
    if (!mounted) return;
    setState(() {
      final override = widget.compileTimeOverride;
      if (override.isNotEmpty) {
        _statusIsError = true;
        _statusMessage =
            'Stored $url, but --dart-define=API_BASE_URL=$override '
            'overrides it at runtime.';
      } else {
        _statusIsError = false;
        _statusMessage = 'Saved. Using $url';
      }
    });
  }

  Future<void> _resetToDefault() async {
    final apiClient = context.read<ApiClient>();

    await Config.clearApiBaseUrl();
    apiClient.invalidateBaseUrlCache();
    final url = await Config.resolveApiBaseUrl(
      compileTimeOverride: widget.compileTimeOverride,
    );
    if (!mounted) return;
    setState(() {
      final override = widget.compileTimeOverride;
      _statusIsError = false;
      _statusMessage = override.isNotEmpty
          ? 'Cleared the stored URL; --dart-define=API_BASE_URL=$override '
                'still applies.'
          : 'Reset to default ($url)';
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
          const SizedBox(height: 16),
          TextField(
            controller: _controller,
            onChanged: (_) => setState(() {}),
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
                onPressed: errorText == null ? _save : null,
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
