"""Tests for cnn_classifier.config module."""

import json
from unittest.mock import patch

import pytest

from cnn_classifier.config import Config, clear_config_cache, get_config


class TestConfig:
    """Test Config class loads defaults correctly."""

    def test_config_loads_defaults(self):
        """Config loads all default values from .env.example."""
        clear_config_cache()
        config = Config()

        assert config.port == 8000
        assert config.model_path == "models/cifar10_cnn.pth"
        assert config.labels_path == "models/labels.json"
        assert config.device in ("cpu", "cuda", "mps")  # auto detects
        assert config.input_size == (32, 32, 3)
        assert config.batch_size == 32
        assert config.num_workers == 4
        assert config.seed == 42

    def test_class_names_property_fallback(self):
        """class_names property returns hardcoded CIFAR-10 classes when labels.json missing."""
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

    def test_class_names_property_from_file(self, tmp_path, monkeypatch):
        """class_names property loads from labels.json when present."""
        labels_file = tmp_path / "custom_labels.json"
        custom_labels = ["class_" + str(i) for i in range(10)]
        labels_file.write_text(json.dumps(custom_labels))

        monkeypatch.setenv("CNN_PROJECT_ROOT", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        config = Config(labels_path=str(labels_file))
        names = config.class_names
        assert names == custom_labels

    def test_config_singleton_get_config(self):
        """get_config returns the same instance (singleton pattern)."""
        clear_config_cache()
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_clear_config_cache(self):
        """clear_config_cache allows new config instance."""
        clear_config_cache()
        config1 = get_config()
        clear_config_cache()
        config2 = get_config()
        assert config1 is not config2

    def test_config_env_override(self, monkeypatch):
        """Environment variables with CNN_ prefix override defaults."""
        monkeypatch.setenv("CNN_PORT", "9000")
        monkeypatch.setenv("CNN_BATCH_SIZE", "64")
        monkeypatch.setenv("CNN_SEED", "123")

        clear_config_cache()
        config = Config()

        assert config.port == 9000
        assert config.batch_size == 64
        assert config.seed == 123

    def test_config_env_override_all_fields(self, monkeypatch):
        """All fields can be overridden via env with CNN_ prefix."""
        monkeypatch.setenv("CNN_PORT", "9000")
        monkeypatch.setenv("CNN_MODEL_PATH", "models/custom.pth")
        monkeypatch.setenv("CNN_LABELS_PATH", "models/custom_labels.json")
        monkeypatch.setenv("CNN_DEVICE", "auto")
        monkeypatch.setenv("CNN_INPUT_SIZE_RAW", "64,64,3")
        monkeypatch.setenv("CNN_BATCH_SIZE", "128")
        monkeypatch.setenv("CNN_NUM_WORKERS", "8")
        monkeypatch.setenv("CNN_SEED", "999")

        clear_config_cache()
        config = Config()

        assert config.port == 9000
        assert config.model_path == "models/custom.pth"
        assert config.labels_path == "models/custom_labels.json"
        assert config.device in ("cpu", "cuda", "mps")
        assert config.input_size == (64, 64, 3)
        assert config.batch_size == 128
        assert config.num_workers == 8
        assert config.seed == 999

    def test_env_prefix_required(self, monkeypatch):
        """Env vars without CNN_ prefix are ignored (case_sensitive=True)."""
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("BATCH_SIZE", "64")

        clear_config_cache()
        config = Config()

        assert config.port == 8000
        assert config.batch_size == 32

    def test_env_case_sensitive(self, monkeypatch):
        """Env vars with wrong case are ignored (case_sensitive=True)."""
        monkeypatch.setenv("cnn_port", "9000")
        monkeypatch.setenv("Cnn_Port", "9000")

        clear_config_cache()
        config = Config()

        assert config.port == 8000


class TestConfigValidation:
    """Test Config validation rules."""

    def test_port_must_be_positive(self):
        """Port must be a positive integer."""
        with pytest.raises(ValueError):
            Config(port=0)

    def test_port_must_be_within_range(self):
        """Port must be <= 65535."""
        with pytest.raises(ValueError):
            Config(port=70000)

    def test_port_valid_boundaries(self):
        """Port accepts valid boundary values."""
        assert Config(port=1).port == 1
        assert Config(port=65535).port == 65535

    def test_batch_size_must_be_positive(self):
        """Batch size must be a positive integer."""
        with pytest.raises(ValueError):
            Config(batch_size=0)

    def test_batch_size_max_512(self):
        """Batch size must be <= 512."""
        with pytest.raises(ValueError):
            Config(batch_size=513)

    def test_batch_size_valid_boundaries(self):
        """Batch size accepts valid boundary values."""
        assert Config(batch_size=1).batch_size == 1
        assert Config(batch_size=512).batch_size == 512

    def test_num_workers_non_negative(self):
        """Num workers must be non-negative."""
        with pytest.raises(ValueError):
            Config(num_workers=-1)

    def test_num_workers_max_16(self):
        """Num workers must be <= 16."""
        with pytest.raises(ValueError):
            Config(num_workers=17)

    def test_num_workers_valid_boundaries(self):
        """Num workers accepts valid boundary values."""
        assert Config(num_workers=0).num_workers == 0
        assert Config(num_workers=16).num_workers == 16

    def test_seed_valid_boundary(self):
        """Seed accepts 0."""
        assert Config(seed=0).seed == 0

    def test_input_size_tuple_of_three_positive_ints(self):
        """Input size must be a tuple of three positive integers."""
        with pytest.raises(ValueError):
            Config(input_size_raw="32,32")  # wrong length
        with pytest.raises(ValueError):
            Config(input_size_raw="0,32,3")  # zero value

    def test_input_size_max_4096(self):
        """Input size dimensions must be <= 4096."""
        with pytest.raises(ValueError):
            Config(input_size_raw="4097,32,3")
        with pytest.raises(ValueError):
            Config(input_size_raw="32,4097,3")
        with pytest.raises(ValueError):
            Config(input_size_raw="32,32,4097")

    def test_validate_input_size_valid_string(self):
        """Valid comma-separated string parses to tuple."""
        config = Config(input_size_raw="64,64,3")
        assert config.input_size == (64, 64, 3)

    def test_validate_input_size_valid_string_with_spaces(self):
        """String with spaces strips and parses."""
        config = Config(input_size_raw=" 32 , 32 , 3 ")
        assert config.input_size == (32, 32, 3)

    def test_validate_input_size_valid_tuple(self):
        """Valid tuple passes through (via internal function)."""
        from cnn_classifier.config import _parse_input_size

        assert _parse_input_size((64, 64, 3)) == (64, 64, 3)
        assert _parse_input_size([64, 64, 3]) == (64, 64, 3)

    def test_validate_input_size_invalid_string_length(self):
        """Wrong-length string raises ValueError."""
        with pytest.raises(ValueError, match="3 dimensions"):
            Config(input_size_raw="32,32")

    def test_validate_input_size_invalid_tuple_length(self):
        """Wrong-length list raises ValueError."""
        from cnn_classifier.config import _parse_input_size

        with pytest.raises(ValueError, match="3 dimensions"):
            _parse_input_size([32, 32])

    def test_validate_input_size_zero_or_negative(self):
        """Zero or negative dimension raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            Config(input_size_raw="0,32,3")
        with pytest.raises(ValueError, match="positive"):
            Config(input_size_raw="-1,32,3")

    def test_validate_input_size_invalid_type(self):
        """Invalid type (dict, int, None) raises ValueError."""
        from cnn_classifier.config import _parse_input_size

        with pytest.raises(ValueError, match="tuple/list of 3 ints or comma-separated string"):
            _parse_input_size({"h": 32, "w": 32, "c": 3})
        with pytest.raises(ValueError):
            _parse_input_size(123)
        with pytest.raises(ValueError):
            _parse_input_size(None)

    def test_validate_input_size_edge_cases(self):
        """Additional edge cases for input_size parsing."""
        from cnn_classifier.config import _parse_input_size

        with pytest.raises(ValueError):
            _parse_input_size("")  # empty string
        with pytest.raises(ValueError):
            _parse_input_size("32,abc,3")  # non-numeric
        with pytest.raises(ValueError):
            _parse_input_size("32.5,32,3")  # float
        # Boundary max values
        assert _parse_input_size("4096,32,3") == (4096, 32, 3)
        assert _parse_input_size("32,4096,3") == (32, 4096, 3)
        assert _parse_input_size("32,32,4096") == (32, 32, 4096)
        # Whitespace variants
        assert _parse_input_size("\t32\n,\r32\t,\n3\t") == (32, 32, 3)

    def test_tensor_size_limit(self):
        """Combined tensor elements must not exceed MAX_TENSOR_ELEMENTS."""
        # 4096 * 4096 * 3 = 50,331,648 > 2,000,000
        with pytest.raises(ValueError, match="exceeds max tensor elements"):
            Config(input_size_raw="4096,4096,3")

    def test_batch_elements_limit(self):
        """Combined batch elements (tensor * batch_size) must not exceed MAX_BATCH_ELEMENTS."""
        # 100 * 100 * 3 * 512 = 15,360,000 < 100,000,000 (OK)
        config = Config(input_size_raw="100,100,3", batch_size=512)
        assert config.input_size == (100, 100, 3)
        assert config.batch_size == 512

        # 200 * 200 * 3 * 512 = 61,440,000 < 100,000,000 (OK)
        config = Config(input_size_raw="200,200,3", batch_size=512)
        assert config.batch_size == 512

        # 300 * 300 * 3 * 512 = 138,240,000 > 100,000,000 (exceeds)
        with pytest.raises(ValueError, match="exceeds max batch elements"):
            Config(input_size_raw="300,300,3", batch_size=512)

    def test_seed_negative_raises(self):
        """Negative seed raises ValidationError."""
        with pytest.raises(ValueError):
            Config(seed=-1)

    def test_device_invalid_value_raises(self):
        """Invalid device string raises ValidationError."""
        with pytest.raises(ValueError):
            Config(device="gpu")
        with pytest.raises(ValueError):
            Config(device="cuda:0")

    @patch("cnn_classifier.config._detect_device")
    @patch("torch.cuda.is_available", return_value=True)
    def test_device_auto_detects_cuda(self, _mock_cuda, mock_detect):
        """Device 'auto' returns CUDA when available."""
        mock_detect.return_value = "cuda"
        config = Config(device="auto")
        assert config.device == "cuda"

    @patch("cnn_classifier.config._detect_device")
    @patch("torch.backends.mps.is_available", return_value=True)
    @patch("torch.cuda.is_available", return_value=False)
    def test_device_auto_detects_mps(self, _mock_cuda, _mock_mps, mock_detect):
        """Device 'auto' returns MPS when CUDA unavailable but MPS available."""
        mock_detect.return_value = "mps"
        config = Config(device="auto")
        assert config.device == "mps"

    @patch("cnn_classifier.config._detect_device")
    def test_device_auto_detects_cpu(self, mock_detect):
        """Device 'auto' returns CPU when neither CUDA nor MPS available."""
        mock_detect.return_value = "cpu"
        config = Config(device="auto")
        assert config.device == "cpu"

    def test_device_cuda_unavailable_raises(self):
        """Explicit CUDA request raises when unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                with pytest.raises(ValueError, match="CUDA requested but not available"):
                    Config(device="cuda")

    def test_device_mps_unavailable_raises(self):
        """Explicit MPS request raises when unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                with pytest.raises(ValueError, match="MPS requested but not available"):
                    Config(device="mps")

    def test_model_path_resolved_relative(self):
        """Relative path resolves to absolute."""
        config = Config(model_path="models/x.pth")
        resolved = config.model_path_resolved
        assert resolved.is_absolute()
        assert resolved.name == "x.pth"

    def test_model_path_resolved_absolute_within_project(self, tmp_path, monkeypatch):
        """Absolute path within project root works."""
        project_file = tmp_path / "models" / "x.pth"
        project_file.parent.mkdir(parents=True)
        project_file.write_text("test")

        monkeypatch.setenv("CNN_PROJECT_ROOT", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        config = Config(model_path=str(project_file))
        resolved = config.model_path_resolved
        assert resolved == project_file.resolve()

    def test_model_path_resolved_with_dots(self, tmp_path, monkeypatch):
        """Path with .. components resolves correctly within project."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_file = models_dir / "x.pth"
        model_file.write_text("test")

        subdir = tmp_path / "subdir"
        subdir.mkdir()

        monkeypatch.setenv("CNN_PROJECT_ROOT", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        config = Config(model_path=str(subdir / ".." / "models" / "x.pth"))
        resolved = config.model_path_resolved
        assert resolved == model_file.resolve()

    def test_path_traversal_blocked_relative(self):
        """Relative path traversal outside project root is blocked."""
        with pytest.raises(ValueError, match="resolves outside project root"):
            Config(model_path="../../../etc/passwd")

    def test_path_traversal_blocked_relative_nested(self):
        """Nested relative path traversal is blocked."""
        with pytest.raises(ValueError, match="resolves outside project root"):
            Config(model_path="models/../../../etc/passwd")

    def test_path_traversal_blocked_absolute(self):
        """Absolute path outside project root is blocked."""
        with pytest.raises(ValueError, match="resolves outside project root"):
            Config(model_path="/etc/passwd")

    def test_labels_path_traversal_blocked(self):
        """Labels path traversal outside project root is blocked."""
        with pytest.raises(ValueError, match="resolves outside project root"):
            Config(labels_path="../../../etc/passwd")


class TestConfigSecurity:
    """Test security-related config behavior."""

    def test_unknown_env_vars_ignored(self, monkeypatch):
        """Unknown env vars with CNN_ prefix are ignored (extra=ignore for env)."""
        monkeypatch.setenv("CNN_UNKNOWN_FIELD", "value")
        config = Config()
        assert config.port == 8000

    def test_env_file_permissions_warning(self, tmp_path, monkeypatch):
        """Test .env file permission check (0600 recommended)."""
        env_file = tmp_path / ".env"
        env_file.write_text("CNN_PORT=8000\n")
        env_file.chmod(0o644)

        monkeypatch.setenv("CNN_PROJECT_ROOT", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        config = Config()
        assert config.port == 8000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
