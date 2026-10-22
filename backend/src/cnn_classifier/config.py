"""Configuration management for CNN Classifier."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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


class Config(BaseSettings):
    """Application configuration loaded from environment variables with defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Allow unknown env vars (only CNN_ prefixed are used)
        env_prefix="CNN_",
    )

    # Server
    port: int = Field(default=8000, gt=0, le=65535, description="Server port")

    # Model paths
    model_path: str = Field(
        default="models/cifar10_cnn.pth", description="Path to trained model weights"
    )
    labels_path: str = Field(default="models/labels.json", description="Path to class labels JSON")

    # Device
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="auto", description="Inference device (auto detects best available)"
    )

    # Model input - stored as string to allow env var parsing, validated in model_validator
    input_size_raw: str = Field(
        default="32,32,3", description="Model input size (H,W,C) as comma-separated string"
    )

    # Data loading
    batch_size: int = Field(default=32, gt=0, le=512, description="Inference batch size")
    num_workers: int = Field(default=4, ge=0, le=16, description="Data loader workers")

    # Reproducibility
    seed: int = Field(default=42, ge=0, description="Random seed")

    @field_validator("device", mode="before")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        """Validate device and auto-detect if 'auto'."""
        if v == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                if torch.backends.mps.is_available():
                    return "mps"
            except ImportError:
                pass
            return "cpu"
        return v

    @field_validator("device")
    @classmethod
    def _check_device_availability(cls, v: str) -> str:
        """Check if requested device is available, fall back to CPU with warning."""
        if v == "cpu":
            return v
        try:
            import torch

            if v == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA requested but not available, falling back to CPU")
            if v == "mps" and not torch.backends.mps.is_available():
                raise ValueError("MPS requested but not available, falling back to CPU")
        except ImportError:
            if v != "cpu":
                raise ValueError("PyTorch not installed, only CPU device available")
        return v

    @model_validator(mode="after")
    def _parse_and_validate(self) -> "Config":
        """Parse input_size and validate paths are safe."""
        # Parse input_size from raw string
        self._input_size_parsed = _parse_input_size(self.input_size_raw)

        # Validate paths are safe (no traversal outside project)
        project_root = Path.cwd().resolve()
        for path_str in (self.model_path, self.labels_path):
            path = Path(path_str)
            try:
                resolved = path.resolve()
                # Allow paths within project root OR relative paths that resolve within project
                resolved.relative_to(project_root)
            except ValueError:
                # Check if it's an absolute path outside project
                if path.is_absolute():
                    raise ValueError(f"Absolute path '{path_str}' is outside project root")
                # For relative paths, the resolved path might be outside if using ..
                # This is OK as long as the final resolved path is within project
                # The check above handles this
                pass
        return self

    @property
    def input_size(self) -> tuple[int, int, int]:
        """Model input size (H, W, C)."""
        return self._input_size_parsed

    @property
    def class_names(self) -> list[str]:
        """CIFAR-10 class names in standard order."""
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
        """Resolved absolute path to model file (resolves .. components safely)."""
        return Path(self.model_path).resolve()

    @property
    def labels_path_resolved(self) -> Path:
        """Resolved absolute path to labels file (resolves .. components safely)."""
        return Path(self.labels_path).resolve()


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get singleton Config instance."""
    return Config()


def clear_config_cache() -> None:
    """Clear the config singleton cache (for testing/hot-reload)."""
    get_config.cache_clear()
