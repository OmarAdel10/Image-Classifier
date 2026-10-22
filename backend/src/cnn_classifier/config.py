"""Configuration management for CNN Classifier.

Environment variables must use the `CNN_` prefix (e.g., `CNN_PORT=8000`).
See `.env.example` for all available options.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root anchored to this package location (works in Docker/systemd).
# Override with CNN_PROJECT_ROOT env var (e.g., for isolated test environments).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_project_root() -> Path:
    """Return the project root: CNN_PROJECT_ROOT env override or package anchor."""
    env_root = os.environ.get("CNN_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return _PROJECT_ROOT


# Maximum tensor elements per batch to prevent OOM (32*32*3*512 = 1.5M, use 2M margin)
_MAX_TENSOR_ELEMENTS = 2_000_000
# Maximum batch elements (tensor * batch_size) to prevent OOM
_MAX_BATCH_ELEMENTS = 100_000_000


def _parse_input_size(v) -> tuple[int, int, int]:
    """Parse input_size from various formats."""
    if isinstance(v, str):
        # Parse from env string like "32,32,3"
        parts = [int(x.strip()) for x in v.split(",")]
        if len(parts) != 3:
            raise ValueError("input_size must have 3 dimensions (H,W,C)")
        if any(x <= 0 or x > 4096 for x in parts):
            raise ValueError("input_size dimensions must be positive and <= 4096")
        return tuple(parts)
    if isinstance(v, (list, tuple)):
        if len(v) != 3:
            raise ValueError("input_size must have 3 dimensions (H,W,C)")
        if any(x <= 0 or x > 4096 for x in v):
            raise ValueError("input_size dimensions must be positive and <= 4096")
        return tuple(v)
    raise ValueError("input_size must be tuple/list of 3 ints or comma-separated string")


def _detect_device() -> str:
    """Auto-detect best available device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class Config(BaseSettings):
    """Application configuration loaded from environment variables with defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,  # Require exact CNN_ prefix and UPPER_SNAKE names
        extra="ignore",  # Ignore unknown CNN_* vars (explicit prefix required)
        env_prefix="CNN_",
        populate_by_name=True,  # Allow init kwargs by field name alongside aliases
    )

    # Server
    port: int = Field(
        default=8000,
        gt=0,
        le=65535,
        validation_alias="CNN_PORT",
        description="Server port",
    )

    # Model paths
    model_path: str = Field(
        default="models/cifar10_cnn.pth",
        validation_alias="CNN_MODEL_PATH",
        description="Path to trained model weights",
    )
    labels_path: str = Field(
        default="models/labels.json",
        validation_alias="CNN_LABELS_PATH",
        description="Path to class labels JSON",
    )

    # Device
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="auto",
        validation_alias="CNN_DEVICE",
        description="Inference device (auto detects best available)",
    )

    # Model input - stored as string to allow env var parsing, validated in model_validator
    input_size_raw: str = Field(
        default="32,32,3",
        validation_alias="CNN_INPUT_SIZE_RAW",
        description="Model input size (H,W,C) as comma-separated string",
    )

    # Data loading
    batch_size: int = Field(
        default=32,
        gt=0,
        le=512,
        validation_alias="CNN_BATCH_SIZE",
        description="Inference batch size",
    )
    num_workers: int = Field(
        default=4,
        ge=0,
        le=16,
        validation_alias="CNN_NUM_WORKERS",
        description="Data loader workers",
    )

    # Reproducibility
    seed: int = Field(default=42, ge=0, validation_alias="CNN_SEED", description="Random seed")

    @field_validator("device", mode="before")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        """Validate device and auto-detect if 'auto'."""
        if v == "auto":
            return _detect_device()
        return v

    @field_validator("device")
    @classmethod
    def _check_device_availability(cls, v: str) -> str:
        """Check if requested device is available, raise if not."""
        if v == "cpu":
            return v
        try:
            import torch

            if v == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA requested but not available")
            if v == "mps" and not torch.backends.mps.is_available():
                raise ValueError("MPS requested but not available")
        except ImportError:
            raise ValueError("PyTorch not installed, only CPU device available")
        return v

    @model_validator(mode="after")
    def _parse_and_validate(self) -> "Config":
        """Parse input_size, validate paths are safe, check memory limits."""
        # Parse input_size from raw string
        input_size = _parse_input_size(self.input_size_raw)
        self._input_size_parsed = input_size

        # Check combined tensor size limit to prevent OOM
        h, w, c = input_size
        tensor_elements = h * w * c
        if tensor_elements > _MAX_TENSOR_ELEMENTS:
            raise ValueError(
                f"input_size {input_size} exceeds max tensor elements "
                f"({tensor_elements:,} > {_MAX_TENSOR_ELEMENTS:,})"
            )

        # Check combined batch size limit
        batch_elements = tensor_elements * self.batch_size
        if batch_elements > _MAX_BATCH_ELEMENTS:
            raise ValueError(
                f"batch_size {self.batch_size} with input_size {input_size} "
                f"exceeds max batch elements ({batch_elements:,} > {_MAX_BATCH_ELEMENTS:,})"
            )

        # Validate paths are safe (no traversal outside project)
        project_root = _get_project_root()
        for path_str in (self.model_path, self.labels_path):
            path = Path(path_str)
            try:
                resolved = path.resolve()
                # Must be within project root (anchored to package location)
                resolved.relative_to(project_root)
            except ValueError:
                raise ValueError(f"Path '{path_str}' resolves outside project root: {resolved}")

        # Store resolved paths for properties (avoids TOCTOU)
        self._model_path_resolved = Path(self.model_path).resolve()
        self._labels_path_resolved = Path(self.labels_path).resolve()

        return self

    @property
    def input_size(self) -> tuple[int, int, int]:
        """Model input size (H, W, C)."""
        return self._input_size_parsed

    @property
    def class_names(self) -> list[str]:
        """CIFAR-10 class names loaded from labels.json with hardcoded fallback."""
        # Try to load from labels_path if it exists
        if self._labels_path_resolved.exists():
            try:
                with open(self._labels_path_resolved) as f:
                    labels = json.load(f)
                if isinstance(labels, list) and len(labels) == 10:
                    return labels
            except (json.JSONDecodeError, OSError):
                pass
        # Fallback to hardcoded CIFAR-10 classes
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

    @property
    def model_path_resolved(self) -> Path:
        """Resolved absolute path to model file (validated at config load)."""
        return self._model_path_resolved

    @property
    def labels_path_resolved(self) -> Path:
        """Resolved absolute path to labels file (validated at config load)."""
        return self._labels_path_resolved


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get singleton Config instance."""
    return Config()


def clear_config_cache() -> None:
    """Clear the config singleton cache (for testing/hot-reload)."""
    get_config.cache_clear()
