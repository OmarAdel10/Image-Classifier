"""Configuration management for CNN Classifier."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application configuration loaded from environment variables with defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server
    port: int = Field(default=8000, gt=0, description="Server port")

    # Model paths
    model_path: str = Field(
        default="models/cifar10_cnn.pth", description="Path to trained model weights"
    )
    labels_path: str = Field(default="models/labels.json", description="Path to class labels JSON")

    # Device
    device: Literal["cpu", "cuda", "mps"] = Field(default="cpu", description="Inference device")

    # Model input
    input_size: tuple[int, int, int] = Field(
        default=(32, 32, 3), description="Model input size (H, W, C)"
    )

    # Data loading
    batch_size: int = Field(default=32, gt=0, description="Inference batch size")
    num_workers: int = Field(default=4, ge=0, description="Data loader workers")

    # Reproducibility
    seed: int = Field(default=42, ge=0, description="Random seed")

    @field_validator("input_size", mode="before")
    @classmethod
    def _validate_input_size(cls, v):
        """Validate input_size is a tuple of 3 positive integers."""
        if isinstance(v, str):
            # Parse from env string like "32,32,3"
            parts = [int(x.strip()) for x in v.split(",")]
            if len(parts) != 3:
                raise ValueError("input_size must have 3 dimensions (H,W,C)")
            return tuple(parts)
        if isinstance(v, (list, tuple)):
            if len(v) != 3:
                raise ValueError("input_size must have 3 dimensions (H,W,C)")
            if any(x <= 0 for x in v):
                raise ValueError("input_size dimensions must be positive")
            return tuple(v)
        raise ValueError("input_size must be tuple/list of 3 ints or comma-separated string")

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
        """Resolved absolute path to model file."""
        return Path(self.model_path).resolve()

    @property
    def labels_path_resolved(self) -> Path:
        """Resolved absolute path to labels file."""
        return Path(self.labels_path).resolve()


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Get singleton Config instance."""
    return Config()
