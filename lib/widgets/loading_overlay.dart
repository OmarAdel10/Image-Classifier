import 'package:flutter/material.dart';

/// Full-screen loading overlay with progress indicator and message.
class LoadingOverlay extends StatelessWidget {
  const LoadingOverlay({
    super.key,
    this.message = 'Loading...',
    this.backgroundColor = Colors.black54,
  });

  final String message;
  final Color backgroundColor;

  @override
  Widget build(BuildContext context) {
    return Positioned.fill(
      child: IgnorePointer(
        child: Container(
          color: backgroundColor,
          child: Center(
            child: Card(
              elevation: 8,
              child: Padding(
                padding: const EdgeInsets.all(32),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const CircularProgressIndicator(),
                    const SizedBox(height: 24),
                    Text(
                      message,
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.w500,
                          ),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}