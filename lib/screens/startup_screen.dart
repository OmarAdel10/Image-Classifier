import 'dart:async';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'package:image_classifier/models/backend_status.dart';
import 'package:image_classifier/providers/backend_controller.dart';

/// Screen shown during backend startup.
class StartupScreen extends StatefulWidget {
  final VoidCallback onReady;
  final String? error;

  const StartupScreen({super.key, required this.onReady, this.error});

  @override
  State<StartupScreen> createState() => _StartupScreenState();
}

class _StartupScreenState extends State<StartupScreen>
    with TickerProviderStateMixin {
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;
  Timer? _retryTimer;

  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      duration: const Duration(milliseconds: 1500),
      vsync: this,
    )..repeat(reverse: true);
    _pulseAnimation = Tween<double>(begin: 0.6, end: 1.0).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _retryTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<BackendController>(
      builder: (context, controller, _) {
        final status = controller.status;
        final isManaged = status.isManaged;
        final managedStatus = status.managedStatus;

        if (isManaged && managedStatus == ManagedBackendStatus.ready) {
          // Ready, transition to main app
          WidgetsBinding.instance.addPostFrameCallback((_) {
            widget.onReady();
          });
        }

        return Scaffold(
          body: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 480),
              child: Padding(
                padding: const EdgeInsets.all(32),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    // App icon / logo
                    _buildLogo(status),

                    const SizedBox(height: 32),

                    // Title
                    Text(
                      _getTitle(status),
                      style: Theme.of(context).textTheme.headlineSmall
                          ?.copyWith(fontWeight: FontWeight.w600),
                      textAlign: TextAlign.center,
                    ),

                    const SizedBox(height: 16),

                    // // Status message
                    // AnimatedSwitcher(
                    //   duration: const Duration(milliseconds: 200),
                    //   child: Text(
                    //     _getMessage(status),
                    //     key: ValueKey(status.managedStatus),
                    //     style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                    //       color: Theme.of(context).colorScheme.onSurfaceVariant,
                    //     ),
                    //     textAlign: TextAlign.center,
                    //   ),
                    // ),

                    // const SizedBox(height: 32),

                    // Progress indicator
                    if (isManaged && managedStatus != null) ...[
                      _buildProgress(status),
                      const SizedBox(height: 24),
                    ],

                    // Error state with retry
                    if (status.isFailed || widget.error != null) ...[
                      _buildError(status, widget.error),
                      const SizedBox(height: 24),
                    ],

                    // Action buttons
                    _buildActions(status),
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildLogo(BackendStatus status) {
    final isFailed = status.isFailed;
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 300),
      child: Container(
        key: ValueKey(isFailed),
        width: 100,
        height: 100,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: isFailed
              ? Theme.of(context).colorScheme.errorContainer
              : Theme.of(context).colorScheme.primaryContainer,
        ),
        child: Center(
          child: AnimatedBuilder(
            animation: _pulseAnimation,
            builder: (context, child) {
              return Transform.scale(
                scale: isFailed ? 1.0 : _pulseAnimation.value,
                child: Icon(
                  isFailed ? Icons.error_outline : Icons.psychology_outlined,
                  size: 50,
                  color: isFailed
                      ? Theme.of(context).colorScheme.onErrorContainer
                      : Theme.of(context).colorScheme.onPrimaryContainer,
                ),
              );
            },
          ),
        ),
      ),
    );
  }

  String _getTitle(BackendStatus status) {
    if (status.isFailed) {
      return 'Backend Failed to Start';
    }
    if (status.managedStatus == ManagedBackendStatus.modelLoading) {
      return 'Loading Model';
    }
    if (status.managedStatus == ManagedBackendStatus.probing) {
      return 'Starting Backend';
    }
    if (status.managedStatus == ManagedBackendStatus.starting) {
      return 'Initializing';
    }
    if (status.isReady) {
      return 'Ready';
    }
    return 'Image Classifier';
  }

  String _getMessage(BackendStatus status) {
    if (status.error != null) {
      return status.error!;
    }
    switch (status.managedStatus) {
      case ManagedBackendStatus.starting:
        return 'Starting the classification backend...';
      case ManagedBackendStatus.probing:
        return 'Waiting for backend to respond...';
      case ManagedBackendStatus.modelLoading:
        return status.modelInfo ?? 'Loading the AI model into memory...';
      case ManagedBackendStatus.ready:
        return 'Backend is ready!';
      case ManagedBackendStatus.failed:
        return 'The backend could not start. See error below.';
      case ManagedBackendStatus.stopped:
      case null:
        if (status.isExternal) {
          return 'Connecting to ${status.baseUrl}...';
        }
        return 'Preparing the classification backend...';
    }
  }

  Widget _buildProgress(BackendStatus status) {
    final managedStatus = status.managedStatus;
    if (managedStatus == ManagedBackendStatus.modelLoading) {
      return Column(
        children: [
          LinearProgressIndicator(
            value: null, // Indeterminate
            backgroundColor: Theme.of(context)
                .colorScheme
                .surfaceContainerHighest,
            valueColor: AlwaysStoppedAnimation(
              Theme.of(context).colorScheme.primary,
            ),
          ),
          const SizedBox(height: 8),
          // Status message
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 200),
            child: Text(
              _getMessage(status),
              key: ValueKey(status.managedStatus),
              style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
              textAlign: TextAlign.center,
            ),
          ),
          // Text(
          //   status.modelInfo ?? 'Loading model...',
          //   style: Theme.of(context).textTheme.bodySmall?.copyWith(
          //     color: Theme.of(context).colorScheme.onSurfaceVariant,
          //   ),
          // ),
        ],
      );
    }

    if (managedStatus == ManagedBackendStatus.probing) {
      return const LinearProgressIndicator();
    }

    if (managedStatus == ManagedBackendStatus.starting) {
      return const LinearProgressIndicator();
    }

    return const SizedBox.shrink();
  }

  Widget _buildError(BackendStatus status, String? externalError) {
    final error = status.error ?? externalError;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.errorContainer,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.error_outline,
                color: Theme.of(context).colorScheme.onErrorContainer,
              ),
              const SizedBox(width: 8),
              Text(
                'Error',
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  color: Theme.of(context).colorScheme.onErrorContainer,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            error ?? 'Unknown error',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              color: Theme.of(context).colorScheme.onErrorContainer,
            ),
          ),
          if (status.isManaged &&
              status.managedStatus == ManagedBackendStatus.failed) ...[
            const SizedBox(height: 16),
            _buildTrainingHint(),
          ],
        ],
      ),
    );
  }

  Widget _buildTrainingHint() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'The AI model has not been trained yet.',
          style: Theme.of(context).textTheme.bodyMedium
              ?.copyWith(fontWeight: FontWeight.w500),
        ),
        const SizedBox(height: 8),
        Text(
          'Run this command once to train the model:',
          style: Theme.of(context).textTheme.bodySmall,
        ),
        const SizedBox(height: 8),
        Container(
          width: double.infinity,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: Theme.of(context).colorScheme.surface,
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
              color: Theme.of(context).colorScheme.outlineVariant,
            ),
          ),
          child: SelectableText(
            './scripts/train_first_model.sh',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
              fontFamily: 'monospace',
              color: Theme.of(context).colorScheme.primary,
            ),
          ),
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            ElevatedButton.icon(
              onPressed: _copyCommand,
              icon: const Icon(Icons.copy, size: 18),
              label: const Text('Copy Command'),
            ),
            const SizedBox(width: 12),
            OutlinedButton.icon(
              onPressed: _openTerminal,
              icon: const Icon(Icons.terminal, size: 18),
              label: const Text('Open Terminal'),
            ),
          ],
        ),
      ],
    );
  }

  void _copyCommand() {
    // In a real app, use clipboard
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Command copied (run in terminal)')),
    );
  }

  void _openTerminal() {
    // Platform-specific terminal opening would go here
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Open a terminal and run the command above'),
      ),
    );
  }

  Widget _buildActions(BackendStatus status) {
    if (status.isReady) {
      return const SizedBox.shrink();
    }

    if (status.isFailed || status.error != null) {
      return FilledButton.icon(
        onPressed: () {
          final controller = context.read<BackendController>();
          if (status.isManaged) {
            controller.restartManagedBackend();
          } else {
            controller.initialize();
          }
        },
        icon: const Icon(Icons.refresh),
        label: const Text('Retry'),
        style: FilledButton.styleFrom(
          padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
        ),
      );
    }

    return const SizedBox.shrink();
  }
}
