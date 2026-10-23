"""Configuration management for CNN Classifier.

Environment variables must use the `CNN_` prefix (e.g., `CNN_PORT=8000`).
Relative paths are anchored to the project root (not the process working
directory), so the service runs from any CWD. Override the root with
`CNN_PROJECT_ROOT` (e.g., for isolated test environments).
See `.env.example` for all available options.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Project root anchored to this package location (works in Docker/systemd).
# Override with CNN_PROJECT_ROOT env var (e.g., for isolated test environments).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_project_root() -> Path:
    """Return the project root: CNN_PROJECT_ROOT env override or package anchor."""
    env_root = os.environ.get("CNN_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return _PROJECT_ROOT


def _resolve_project_path(path_str: str, project_root: Path) -> Path:
    """Resolve a configured path and verify it stays inside the project root.

    Relative paths are anchored to ``project_root`` (not the process CWD) so
    defaults work regardless of the working directory. Resolution happens once
    here and the result is stored on the instance (avoids TOCTOU re-resolution).
    """
    path = Path(path_str)
    if not path.is_absolute():
        path = project_root / path
    try:
        resolved = path.resolve()
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"Path '{path_str}' resolves outside project root") from exc
    return resolved


# Maximum pixels (H*W) per image to prevent OOM (assumes RGB, C=3 → ~6M elements)
_MAX_IMAGE_PIXELS = 2_000_000
# Maximum pixel-batch (H*W * batch_size) to prevent OOM
_MAX_BATCH_PIXELS = 100_000_000


def _parse_input_size(v) -> tuple[int, int]:
    """Parse input_size from various formats (env string, list, tuple)."""
    if isinstance(v, str):
        # Parse from env string like "32,32"
        parts = [int(x.strip()) for x in v.split(",")]
        if len(parts) != 2:
            raise ValueError("input_size must have 2 dimensions (H,W)")
        if any(x <= 0 or x > 4096 for x in parts):
            raise ValueError("input_size dimensions must be positive and <= 4096")
        return tuple(parts)
    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise ValueError("input_size must have 2 dimensions (H,W)")
        if any(x <= 0 or x > 4096 for x in v):
            raise ValueError("input_size dimensions must be positive and <= 4096")
        return tuple(v)
    raise ValueError("input_size must be tuple/list of 2 ints or comma-separated string")


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
        default="checkpoints/best_model.pt",
        validation_alias="CNN_MODEL_PATH",
        description="Path to trained model weights",
    )
    labels_path: str = Field(
        default="checkpoints/labels.json",
        validation_alias="CNN_LABELS_PATH",
        description="Path to class labels JSON",
    )

    # Device
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="auto",
        validation_alias="CNN_DEVICE",
        description="Inference device (auto detects best available)",
    )

    # Model input size (H, W) for Resize/CenterCrop. NoDecode stops
    # pydantic-settings from JSON-decoding the env string; the before-validator
    # below parses formats like "32,32" (env) and tuple/list kwargs instead.
    input_size: Annotated[tuple[int, int], NoDecode] = Field(
        default=(32, 32),
        validation_alias="CNN_INPUT_SIZE",
        description="Model input size (H, W)",
    )

    @field_validator("input_size", mode="before")
    @classmethod
    def _validate_input_size(cls, v) -> tuple[int, int]:
        """Parse input_size from env string / list / tuple before type coercion."""
        return _parse_input_size(v)

    # Data loading
    batch_size: int = Field(
        default=64,
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
    def _validate_memory_and_paths(self) -> "Config":
        """Enforce memory limits and validate paths are safe."""
        # Check image/batch pixel limits to prevent OOM (RGB, C=3 assumed)
        h, w = self.input_size
        image_pixels = h * w
        if image_pixels > _MAX_IMAGE_PIXELS:
            raise ValueError(
                f"input_size {self.input_size} exceeds max image pixels "
                f"({image_pixels:,} > {_MAX_IMAGE_PIXELS:,})"
            )

        batch_pixels = image_pixels * self.batch_size
        if batch_pixels > _MAX_BATCH_PIXELS:
            raise ValueError(
                f"batch_size {self.batch_size} with input_size {self.input_size} "
                f"exceeds max batch pixels ({batch_pixels:,} > {_MAX_BATCH_PIXELS:,})"
            )

        # Validate paths are safe (no traversal outside project)
        project_root = _get_project_root()
        self._model_path_resolved = _resolve_project_path(self.model_path, project_root)
        self._labels_path_resolved = _resolve_project_path(self.labels_path, project_root)

        return self

    @property
    def class_names(self) -> list[str]:
        """CIFAR-10 class names loaded from labels.json with hardcoded fallback."""
        # Try to load from labels_path if it exists
        if self._labels_path_resolved.exists():
            try:
                with open(self._labels_path_resolved, encoding="utf-8") as f:
                    labels = json.load(f)
                if isinstance(labels, list) and len(labels) == 10:
                    return labels
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
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
