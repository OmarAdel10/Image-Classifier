"""Predictor: load a checkpoint and run single-image inference."""

from collections import namedtuple

import torch

from cnn_classifier.config import Config
from cnn_classifier.inference.checkpoints import load_checkpoint
from cnn_classifier.models import get_model

Prediction = namedtuple("Prediction", ["class_name", "confidence"])


class Predictor:
    """Runs a rebuilt model on preprocessed image tensors."""

    def __init__(self, model, class_names: list, device: str, input_size: tuple) -> None:
        self.model = model
        self.class_names = class_names
        self.device = device
        self.input_size = tuple(input_size)

    @classmethod
    def from_checkpoint(cls, path: str, config: Config) -> "Predictor":
        """Rebuild the trained model from a checkpoint file.

        The model class comes from the checkpoint's recorded
        ``model_name`` (config provenance), so the correct architecture
        is rebuilt even if the runtime config drifted.

        Args:
            path: Checkpoint file path (explicit; no default locations).
            config: Application config (consumes ``device``).

        Returns:
            A Predictor in eval mode, on ``config.device``.
        """
        checkpoint = load_checkpoint(path)
        ckpt_config = checkpoint["config"]
        if (
            not isinstance(ckpt_config, dict)
            or "model_name" not in ckpt_config
            or "input_size" not in ckpt_config
        ):
            raise ValueError(f"malformed checkpoint config: {ckpt_config!r}")
        model_config = Config(input_size=tuple(ckpt_config["input_size"]), device=config.device)
        model = get_model(ckpt_config["model_name"], model_config)
        try:
            model.load_state_dict(checkpoint["model_state"])
        except RuntimeError as exc:
            raise ValueError("checkpoint model_state does not match recorded architecture") from exc
        model = model.to(config.device)
        model.eval()
        class_names = list(checkpoint["class_names"])
        with torch.no_grad():
            probe = model(torch.zeros(1, 3, *model_config.input_size, device=config.device))
        if len(class_names) != probe.shape[1]:
            raise ValueError(f"class count {len(class_names)} != model outputs {probe.shape[1]}")
        return cls(model, class_names, config.device, tuple(ckpt_config["input_size"]))

    def predict(self, tensor: torch.Tensor) -> list:
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
        tensor = tensor.to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        ranked = torch.argsort(probabilities, descending=True)
        return [
            Prediction(self.class_names[int(index)], float(probabilities[int(index)]))
            for index in ranked
        ]
