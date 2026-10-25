"""Shared training-run logic for the CLI entry points."""

import json
import logging
import random

import torch
from torch import nn

from cnn_classifier.config import Config
from cnn_classifier.data import get_dataloaders
from cnn_classifier.models import get_model
from cnn_classifier.training.atomic import atomic_write
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
    random.seed(config.seed)
    model = get_model(model_name, config).to(config.device)
    train_loader, val_loader, test_loader = get_dataloaders(config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    class_names = config.class_names

    best_val_acc = 0.0
    best_epoch = 0
    checkpoint_saved_this_run = False
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
                model_name,
                config,
                class_names,
            )
            checkpoint_saved_this_run = True
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

    # Report metrics for the saved best checkpoint, not the final weights.
    if checkpoint_saved_this_run:
        best_state = torch.load(config.model_path_resolved, weights_only=True)
        model.load_state_dict(best_state["model_state"])
    elif config.model_path_resolved.exists():
        logger.warning(
            "no val-acc improvement this run; %s left from a previous run "
            "and was not updated - report/labels describe the final weights",
            config.model_path_resolved,
        )
    else:
        logger.warning(
            "no checkpoint saved this run (no val-acc improvement) - "
            "report/labels describe the final weights"
        )

    # Spec artifacts: evaluate on the held-out test set and persist the
    # class names alongside the checkpoint (plan lines 357-359).
    test_metrics = evaluate(model, test_loader, config.device, class_names)
    test_metrics["evaluated_epoch"] = best_epoch if checkpoint_saved_this_run else epochs
    save_classification_report(
        test_metrics, config.model_path_resolved.parent / "classification_report.json"
    )
    atomic_write(config.labels_path_resolved, lambda tmp: tmp.write_text(json.dumps(class_names)))
    logger.info("saved labels.json and classification_report.json")


__all__ = ["run_training"]
