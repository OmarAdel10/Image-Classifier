"""Training loop, validation, and checkpoint utilities.

Consumes ``Config`` (via callers), ``get_dataloaders`` output, and any
``nn.Module`` produced by ``get_model``. The training step enforces the
frozen-backbone contract documented in ``cnn_classifier.models.transfer``:
when a model has a backbone, modules whose parameters are all frozen
are switched to ``eval()`` inside ``train_one_epoch`` (param-less modules
like Dropout stay in train mode) so BatchNorm running statistics stay
frozen while the head trains.
"""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn

from cnn_classifier.config import Config
from cnn_classifier.data.dataset import CIFAR10_MEAN, CIFAR10_STD
from cnn_classifier.training.atomic import atomic_write

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
    for module in backbone.modules():
        params = list(module.parameters())
        if params and all(not p.requires_grad for p in params):
            module.eval()


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
        total_loss += loss.item() * targets.size(0)
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += targets.size(0)
    return {"loss": total_loss / seen if seen else 0.0, "acc": correct / seen if seen else 0.0}


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
            total_loss += loss.item() * targets.size(0)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += targets.size(0)
    return {"loss": total_loss / seen if seen else 0.0, "acc": correct / seen if seen else 0.0}


def _save_labels_sidecar(model_path: Path, class_names: list) -> None:
    """Save class names as a sidecar labels.json next to the model checkpoint."""
    labels_path = model_path.with_name(model_path.stem + "_labels.json")
    atomic_write(labels_path, lambda tmp: tmp.write_text(json.dumps(list(class_names))))


def _save_sha256_sidecar(model_path: Path) -> None:
    """Write a SHA256 sidecar file for the checkpoint (IC-32).

    The sidecar is named ``<model_stem>.sha256`` and contains the
    hex-encoded SHA256 hash of the checkpoint file. This enables
    integrity verification at load time to mitigate supply-chain
    and tampering risks (CWE-345).
    """
    sha256_path = model_path.with_suffix(model_path.suffix + ".sha256")
    hasher = hashlib.sha256()
    with model_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    atomic_write(sha256_path, lambda tmp: tmp.write_text(hasher.hexdigest()))


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    model_name: str,
    config: Config,
    class_names: list,
) -> None:
    """Persist a training checkpoint dict to ``path``.

    The dict layout matches the plan contract: ``model_state``,
    ``optimizer_state``, ``epoch``, ``best_val_acc``, ``config``, and
    ``class_names``.

    Also writes a sidecar ``<model_stem>_labels.json`` next to the checkpoint
    for automatic label discovery during inference.
    """
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "config": {
            "model_name": model_name,
            "input_size": tuple(config.input_size),
            "seed": config.seed,
            "device": config.device,
            "batch_size": config.batch_size,
            # Normalization provenance (IC-30): record mean/std used during training
            # so inference can verify it matches the training normalization.
            "normalization": {
                "mean": CIFAR10_MEAN,
                "std": CIFAR10_STD,
            },
        },
        "class_names": list(class_names),
    }
    model_path = Path(path)
    atomic_write(model_path, lambda tmp: torch.save(payload, tmp))
    # Write sidecar labels.json next to the model checkpoint
    _save_labels_sidecar(model_path, class_names)
    # Write SHA256 sidecar for integrity verification at load (IC-32)
    _save_sha256_sidecar(model_path)
