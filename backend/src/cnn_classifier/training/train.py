"""Training loop, validation, and checkpoint utilities.

Consumes ``Config`` (via callers), ``get_dataloaders`` output, and any
``nn.Module`` produced by ``get_model``. The training step enforces the
frozen-backbone contract documented in ``cnn_classifier.models.transfer``:
when a model has a backbone whose parameters are (partially) frozen, the
backbone is switched to ``eval()`` inside ``train_one_epoch`` so BatchNorm
running statistics stay frozen while the head trains.
"""

from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn

from cnn_classifier.config import Config

# Gradient clipping bound applied after loss.backward() (spec: max_norm=1.0).
_GRAD_CLIP_NORM = 1.0


def _enforce_frozen_backbone_eval(model: nn.Module) -> None:
    """Put a (partially) frozen backbone into eval mode.

    Freezing parameters alone does not stop BatchNorm running-stat updates
    in train mode; switching the backbone to eval does. Only applies when
    the model exposes a ``backbone`` attribute with at least one frozen
    parameter (feature-extractor phase).
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return
    if any(not param.requires_grad for param in backbone.parameters()):
        backbone.eval()


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> dict:
    """Run one training epoch and return average loss and accuracy.

    Args:
        model: Training model (any ``nn.Module``).
        loader: DataLoader yielding ``(inputs, targets)`` batches.
        optimizer: Optimizer stepping the parameters.
        criterion: Loss module (e.g. ``nn.CrossEntropyLoss``).
        device: Device to move batches to (e.g. ``'cpu'``).

    Returns:
        Dict with float ``loss`` (average batch loss) and ``acc``
        (top-1 accuracy in ``[0, 1]``).
    """
    model.train()
    _enforce_frozen_backbone_eval(model)
    total_loss = 0.0
    correct = 0
    seen = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=_GRAD_CLIP_NORM)
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += targets.size(0)
    avg_loss = total_loss / len(loader) if len(loader) else 0.0
    return {"loss": avg_loss, "acc": correct / seen if seen else 0.0}


def validate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: str,
) -> dict:
    """Evaluate loss and accuracy without gradient updates.

    Leaves the model in eval mode (callers re-entering training must call
    ``model.train()``; ``train_one_epoch`` does this itself).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    seen = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += targets.size(0)
    avg_loss = total_loss / len(loader) if len(loader) else 0.0
    return {"loss": avg_loss, "acc": correct / seen if seen else 0.0}


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    config: Config,
    class_names: list,
) -> None:
    """Persist a training checkpoint dict to ``path``.

    The dict layout matches the plan contract: ``model_state``,
    ``optimizer_state``, ``epoch``, ``best_val_acc``, ``config``, and
    ``class_names``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "config": {
                "input_size": tuple(config.input_size),
                "seed": config.seed,
                "device": config.device,
                "batch_size": config.batch_size,
            },
            "class_names": list(class_names),
        },
        path,
    )
