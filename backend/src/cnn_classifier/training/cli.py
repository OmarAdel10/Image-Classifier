"""Shared training-run logic for the CLI entry points."""

import json
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

from cnn_classifier.config import Config, get_config
from cnn_classifier.data import get_dataloaders
from cnn_classifier.models import get_model
from cnn_classifier.training.evaluate import evaluate, save_classification_report
from cnn_classifier.training.train import save_checkpoint, train_one_epoch, validate

logger = logging.getLogger(__name__)

# Stop training after this many epochs without val-acc improvement.
_EARLY_STOP_PATIENCE = 10


def run_training(config: Config, model_name: str, epochs: int, lr: float) -> None:
    """Full training run: data, model, cosine LR, early stop, checkpoint.

    Args:
        config: Application config (seed, batch size, device, paths).
        model_name: Registry model name (e.g. ``'cnn_scratch'``).
        epochs: Maximum number of epochs.
        lr: Optimizer learning rate.
    """
    torch.manual_seed(config.seed)
    train_loader, val_loader, test_loader = get_dataloaders(config)
    model = get_model(model_name, config).to(config.device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0
    epochs_since_best = 0
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, config.device)
        val_metrics = validate(model, val_loader, criterion, config.device)
        scheduler.step()
        logger.info(
            "epoch %d: train loss %.4f acc %.4f | val loss %.4f acc %.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["acc"],
            val_metrics["loss"],
            val_metrics["acc"],
        )
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            epochs_since_best = 0
            save_checkpoint(
                config.model_path_resolved,
                model,
                optimizer,
                epoch,
                best_val_acc,
                config,
                config.class_names,
            )
            logger.info("saved best checkpoint (val acc %.4f)", best_val_acc)
        else:
            epochs_since_best += 1
            if epochs_since_best >= _EARLY_STOP_PATIENCE:
                logger.info(
                    "early stopping at epoch %d (no val improvement " "for %d epochs)",
                    epoch,
                    _EARLY_STOP_PATIENCE,
                )
                break
    logger.info("training done: best val acc %.4f at epoch %d", best_val_acc, best_epoch)

    # Spec artifacts: evaluate on the held-out test set and persist the
    # class names alongside the checkpoint (plan lines 357-359).
    test_metrics = evaluate(model, test_loader, config.device, config.class_names)
    save_classification_report(
        test_metrics, config.model_path_resolved.parent / "classification_report.json"
    )
    config.labels_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    config.labels_path_resolved.write_text(json.dumps(config.class_names))
    logger.info("saved labels.json and classification_report.json")


__all__ = ["run_training", "get_config", "DataLoader"]
