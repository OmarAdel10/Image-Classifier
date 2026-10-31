"""Predictor: load a checkpoint and run single-image inference."""

from collections import namedtuple
from pathlib import Path

import torch

from cnn_classifier.config import Config
from cnn_classifier.inference.checkpoints import load_checkpoint, load_checkpoint_with_sidecar
from cnn_classifier.models import get_model

Prediction = namedtuple("Prediction", ["class_name", "confidence"])


class Predictor:
    """Runs a rebuilt model on preprocessed image tensors."""

    def __init__(
        self,
        model,
        class_names: list,
        device: str,
        input_size: tuple,
        *,
        model_name: str,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.device = device
        self.input_size = tuple(input_size)
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None

    @classmethod
    def from_checkpoint(cls, path: str | Path, config: Config) -> "Predictor":
        """Rebuild the trained model from a checkpoint file.

        The model class comes from the checkpoint's recorded
        ``model_name`` (config provenance), so the correct architecture
        is rebuilt even if the runtime config drifted.

        Args:
            path: Checkpoint file path (explicit; no default locations).
            config: Application config (consumes ``device``).

        Returns:
            A Predictor in eval mode, on ``config.device``.

        Note:
            If a sidecar ``<model_stem>_labels.json`` exists next to the
            checkpoint, its class_names take precedence over the checkpoint's
            embedded class_names for automatic label discovery.
        """
        checkpoint = load_checkpoint_with_sidecar(path)
        ckpt_config = checkpoint["config"]
        if (
            not isinstance(ckpt_config, dict)
            or "model_name" not in ckpt_config
            or "input_size" not in ckpt_config
            or not isinstance(ckpt_config["model_name"], str)
            or not isinstance(ckpt_config["input_size"], (list, tuple))
            or len(ckpt_config["input_size"]) != 2
            or not all(
                isinstance(v, int) and not isinstance(v, bool) for v in ckpt_config["input_size"]
            )
        ):
            raise ValueError(f"malformed checkpoint config: {ckpt_config!r}")
        model_config = Config(
            input_size=tuple(ckpt_config["input_size"]),
            device=config.device,
            model_path=config.model_path,
            labels_path=config.labels_path,
            batch_size=1,
            num_workers=0,
        )
        if not isinstance(checkpoint["model_state"], dict):
            raise ValueError("checkpoint model_state does not match recorded architecture")
        model = get_model(ckpt_config["model_name"], model_config, pretrained=False)
        try:
            model.load_state_dict(checkpoint["model_state"])
        except (RuntimeError, TypeError) as exc:
            raise ValueError("checkpoint model_state does not match recorded architecture") from exc
        model = model.to(config.device)
        model.eval()
        if not isinstance(checkpoint["class_names"], list) or not all(
            isinstance(name, str) for name in checkpoint["class_names"]
        ):
            raise ValueError("malformed checkpoint contents: class_names must be a list of strings")
        class_names = list(checkpoint["class_names"])
        probe_size = tuple(min(dim, 512) for dim in model_config.input_size)
        with torch.no_grad():
            probe = model(torch.ones(1, 3, *probe_size, device=config.device))
        if len(class_names) != probe.shape[1]:
            raise ValueError(f"class count {len(class_names)} != model outputs {probe.shape[1]}")
        if not torch.isfinite(probe).all():
            raise ValueError("checkpoint weights produce non-finite outputs")
        return cls(
            model,
            class_names,
            config.device,
            tuple(ckpt_config["input_size"]),
            model_name=ckpt_config["model_name"],
            checkpoint_path=path,
        )

    def predict(self, tensor: torch.Tensor) -> list[Prediction]:
        """Classify one preprocessed image tensor.

        Args:
            tensor: (1, C, H, W) tensor from ``preprocess_image``.

        Returns:
            Predictions sorted by confidence (descending).

        Raises:
            ValueError: If the tensor is not a single-image batch.
        """
        if tensor.dim() != 4 or tensor.shape[0] != 1:
            raise ValueError(
                f"expected a single-image tensor (1, C, H, W), got shape {tuple(tensor.shape)}"
            )
        if tuple(tensor.shape[1:]) != (3, *self.input_size):
            raise ValueError(
                f"expected tensor (1, 3, {self.input_size[0]}, {self.input_size[1]}) "
                f"matching checkpoint input size {self.input_size}, "
                f"got shape {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.float32:
            raise ValueError(f"expected a float32 tensor, got {tensor.dtype}")
        if not torch.isfinite(tensor).all():
            raise ValueError("non-finite input tensor values")
        tensor = tensor.to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        if not torch.isfinite(probabilities).all():
            raise ValueError("model produced non-finite probabilities")
        ranked = torch.argsort(probabilities, descending=True)
        return [
            Prediction(self.class_names[int(index)], float(probabilities[int(index)]))
            for index in ranked
        ]
