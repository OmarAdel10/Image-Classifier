"""Evaluation: top-1 accuracy, per-class metrics, confusion matrix.

Metrics are computed manually from the confusion matrix (no sklearn
dependency). ``evaluate`` mirrors the plan contract: it returns a dict
with ``acc``, ``per_class`` (precision/recall/f1/support per class name),
and ``confusion_matrix`` (nested list). ``save_classification_report``
persists the metrics dict as JSON.
"""

import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader


def _confusion_matrix(preds: list, labels: list, num_classes: int) -> list:
    """Build a num_classes x num_classes nested-list matrix (rows=true)."""
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for pred, label in zip(preds, labels):
        matrix[label][pred] += 1
    return matrix


def _per_class_metrics(matrix: list, class_names: list) -> dict:
    """Precision/recall/F1/support per class from a confusion matrix."""
    num_classes = len(class_names)
    per_class = {}
    for idx, name in enumerate(class_names):
        true_pos = matrix[idx][idx]
        predicted = sum(matrix[row][idx] for row in range(num_classes))
        actual = sum(matrix[idx])
        precision = true_pos / predicted if predicted else 0.0
        recall = true_pos / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": actual,
        }
    return per_class


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    class_names: list,
) -> dict:
    """Evaluate a model and return accuracy, per-class metrics, matrix.

    Args:
        model: Model under evaluation (switched to eval mode).
        test_loader: DataLoader yielding ``(inputs, targets)``.
        device: Device to move batches to.
        class_names: Ordered class labels (index -> name).

    Returns:
        Dict with ``acc`` (top-1 in ``[0, 1]``), ``per_class`` (name ->
        precision/recall/f1/support), and ``confusion_matrix`` (nested
        list, rows = true class).
    """
    model.eval()
    preds: list = []
    labels: list = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds.extend(logits.argmax(dim=1).tolist())
            labels.extend(targets.tolist())
    matrix = _confusion_matrix(preds, labels, len(class_names))
    total = len(labels)
    correct = sum(matrix[idx][idx] for idx in range(len(class_names)))
    return {
        "acc": correct / total if total else 0.0,
        "per_class": _per_class_metrics(matrix, class_names),
        "confusion_matrix": matrix,
    }


def save_classification_report(metrics: dict, path: str | Path) -> None:
    """Persist an evaluate() metrics dict as a JSON report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
