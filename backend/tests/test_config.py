"""Tests for cnn_classifier.config module."""

import pytest

from cnn_classifier.config import Config, get_config


class TestConfig:
    """Test Config class loads defaults correctly."""

    def test_config_loads_defaults(self):
        """Config loads all default values from .env.example."""
        config = Config()

        assert config.port == 8000
        assert config.model_path == "models/cifar10_cnn.pth"
        assert config.labels_path == "models/labels.json"
        assert config.device in ("cpu", "cuda", "mps")
        assert config.input_size == (32, 32, 3)
        assert config.batch_size == 32
        assert config.num_workers == 4
        assert config.seed == 42

    def test_class_names_property(self):
        """class_names property returns 10 CIFAR-10 classes in correct order."""
        config = Config()
        names = config.class_names

        assert isinstance(names, list)
        assert len(names) == 10
        assert names == [
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

    def test_config_singleton_get_config(self):
        """get_config returns the same instance (singleton pattern)."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_config_env_override(self, monkeypatch):
        """Environment variables override defaults."""
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("BATCH_SIZE", "64")
        monkeypatch.setenv("SEED", "123")

        config = Config()

        assert config.port == 9000
        assert config.batch_size == 64
        assert config.seed == 123


class TestConfigValidation:
    """Test Config validation rules."""

    def test_port_must_be_positive(self):
        """Port must be a positive integer."""
        with pytest.raises(ValueError):
            Config(port=0)

    def test_batch_size_must_be_positive(self):
        """Batch size must be a positive integer."""
        with pytest.raises(ValueError):
            Config(batch_size=0)

    def test_num_workers_non_negative(self):
        """Num workers must be non-negative."""
        with pytest.raises(ValueError):
            Config(num_workers=-1)

    def test_input_size_tuple_of_three_positive_ints(self):
        """Input size must be a tuple of three positive integers."""
        with pytest.raises(ValueError):
            Config(input_size=(32, 32))  # wrong length
        with pytest.raises(ValueError):
            Config(input_size=(0, 32, 3))  # zero value
