import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

import 'package:image_classifier/models/prediction.dart';

/// Horizontal bar chart showing top-5 predictions with confidence bars.
class PredictionBars extends StatelessWidget {
  const PredictionBars({
    super.key,
    required this.predictions,
    this.maxBars = 5,
  });

  final List<PredictionItem> predictions;
  final int maxBars;

  @override
  Widget build(BuildContext context) {
    final sorted = _getSortedPredictions();

    return BarChart(
      BarChartData(
        alignment: BarChartAlignment.spaceAround,
        maxY: 1.0,
        minY: 0,
        barGroups: _buildBarGroups(sorted),
        titlesData: FlTitlesData(
          show: true,
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              getTitlesWidget: (value, meta) {
                if (value.toInt() >= 0 && value.toInt() < maxBars) {
                  final item = sorted[value.toInt()];
                  return Padding(
                    padding: const EdgeInsets.only(right: 8),
                    child: Text(
                      item.className,
                      style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w500),
                      textAlign: TextAlign.right,
                    ),
                  );
                }
                return const Text('');
              },
              reservedSize: 80,
            ),
          ),
          bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              getTitlesWidget: (value, meta) {
                if (value.toInt() >= 0 && value.toInt() < maxBars) {
                  final item = sorted[value.toInt()];
                  return Padding(
                    padding: const EdgeInsets.only(left: 8),
                    child: Text(
                      '${(item.confidence * 100).round()}%',
                      style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600),
                    ),
                  );
                }
                return const Text('');
              },
              reservedSize: 50,
            ),
          ),
        ),
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          horizontalInterval: 0.2,
          getDrawingHorizontalLine: (value) => FlLine(
            color: Colors.grey.withValues(alpha: 0.2),
            strokeWidth: 1,
          ),
        ),
        borderData: FlBorderData(show: false),
        barTouchData: BarTouchData(
          enabled: true,
          touchTooltipData: BarTouchTooltipData(
            getTooltipItem: (group, groupIndex, rod, rodIndex) {
              final item = sorted[group.x];
              return BarTooltipItem(
                '${item.className}: ${(item.confidence * 100).toStringAsFixed(1)}%',
                const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
              );
            },
          ),
        ),
      )
    );
  }

  List<BarChartGroupData> _buildBarGroups(List<PredictionItem> sorted) {
    return List.generate(
      sorted.length.clamp(0, maxBars),
      (index) {
        final item = sorted[index];
        final color = _confidenceToColor(item.confidence);
        return BarChartGroupData(
          x: index,
          barRods: [
            BarChartRodData(
              toY: item.confidence,
              color: color,
              width: 20,
              borderRadius: const BorderRadius.only(
                topRight: Radius.circular(8),
                bottomRight: Radius.circular(8),
              ),
              backDrawRodData: BackgroundBarChartRodData(
                show: true,
                toY: 1.0,
                color: Colors.grey.withValues(alpha: 0.1),
              ),
            ),
          ],
          showingTooltipIndicators: [0],
        );
      },
    );
  }

  List<PredictionItem> _getSortedPredictions() {
    final sorted = List<PredictionItem>.from(predictions)
      ..sort((a, b) => b.confidence.compareTo(a.confidence));
    return sorted;
  }

  Color _confidenceToColor(double confidence) {
    if (confidence >= 0.7) {
      return Colors.green.shade600;
    } else if (confidence >= 0.4) {
      return Colors.orange.shade600;
    } else if (confidence >= 0.2) {
      return Colors.deepOrange.shade500;
    } else {
      return Colors.red.shade500;
    }
  }
}