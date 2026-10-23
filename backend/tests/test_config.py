"""Tests for cnn_classifier.config module."""

import json
from unittest.mock import patch

import pytest

from cnn_classifier.config import Config, clear_config_cache, get_config

EXPECTED_CIFAR10 = [
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


class TestConfig:
    """Test Config class loads defaults correctly."""

    def test_config_loads_defaults(self):
        """Config loads all default values."""
        clear_config_cache()
        config = Config()

        assert config.port == 8000
        assert config.model_path == "checkpoints/best_model.pt"
        assert config.labels_path == "checkpoints/labels.json"
        assert config.device in ("cpu", "cuda", "mps")  # auto detects
        assert config.input_size == (32, 32)
        assert config.batch_size == 64
        assert config.num_workers == 4
        assert config.seed == 42

    def test_class_names_property_fallback(self):
        """class_names property returns hardcoded CIFAR-10 classes when labels.json missing."""
        config = Config()
        names = config.class_names

        assert isinstance(names, list)
        assert len(names) == 10
        assert names == EXPECTED_CIFAR10

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

    def test_class_names_property_wrong_length_file_falls_back(self, tmp_path, monkeypatch):
        """class_names falls back to CIFAR-10 when labels.json has wrong item count."""
        labels_file = tmp_path / "short_labels.json"
        labels_file.write_text(json.dumps(["a", "b", "c"]))

        monkeypatch.setenv("CNN_PROJECT_ROOT", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        config = Config(labels_path=str(labels_file))
        assert config.class_names == EXPECTED_CIFAR10

    def test_class_names_property_corrupt_file_falls_back(self, tmp_path, monkeypatch):
        """class_names falls back to CIFAR-10 when labels.json is invalid JSON."""
        labels_file = tmp_path / "corrupt_labels.json"
        labels_file.write_text("{oops")

        monkeypatch.setenv("CNN_PROJECT_ROOT", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        config = Config(labels_path=str(labels_file))
        assert config.class_names == EXPECTED_CIFAR10

    def test_class_names_invalid_utf8_falls_back(self, tmp_path, monkeypatch):
        """class_names falls back when labels.json contains invalid UTF-8 bytes."""
        labels_file = tmp_path / "labels.json"
        labels_file.write_bytes(b"\xff\xfe{oops")

        monkeypatch.setenv("CNN_PROJECT_ROOT", str(tmp_path))
        config = Config(labels_path=str(labels_file))
        assert config.class_names == EXPECTED_CIFAR10

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
        monkeypatch.setenv("CNN_INPUT_SIZE", "64,64")
        monkeypatch.setenv("CNN_BATCH_SIZE", "128")
        monkeypatch.setenv("CNN_NUM_WORKERS", "8")
        monkeypatch.setenv("CNN_SEED", "999")

        clear_config_cache()
        config = Config()

        assert config.port == 9000
        assert config.model_path == "models/custom.pth"
        assert config.labels_path == "models/custom_labels.json"
        assert config.device in ("cpu", "cuda", "mps")
        assert config.input_size == (64, 64)
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
        assert config.batch_size == 64

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

    def test_input_size_tuple_of_two_positive_ints(self):
        """Input size must be a tuple of two positive integers."""
        with pytest.raises(ValueError):
            Config(input_size="32,32,3")  # wrong length (3 parts)
        with pytest.raises(ValueError):
            Config(input_size="0,32")  # zero value

    def test_input_size_max_4096(self):
        """Input size dimensions must be <= 4096."""
        with pytest.raises(ValueError):
            Config(input_size="4097,32")
        with pytest.raises(ValueError):
            Config(input_size="32,4097")

    def test_validate_input_size_valid_string(self):
        """Valid comma-separated string parses to tuple."""
        config = Config(input_size="64,64")
        assert config.input_size == (64, 64)

    def test_validate_input_size_valid_string_with_spaces(self):
        """String with spaces strips and parses."""
        config = Config(input_size=" 32 , 32 ")
        assert config.input_size == (32, 32)

    def test_validate_input_size_valid_tuple(self):
        """Valid tuple/list pass through (via internal function)."""
        from cnn_classifier.config import _parse_input_size

        assert _parse_input_size((64, 64)) == (64, 64)
        assert _parse_input_size([64, 64]) == (64, 64)

    def test_input_size_kwarg_tuple_plan_contract(self):
        """Config accepts input_size as tuple kwarg (plan contract)."""
        assert Config(input_size=(32, 32), device="cpu").input_size == (32, 32)
        assert Config(input_size=(224, 224), device="cpu").input_size == (224, 224)
        assert Config(input_size=[64, 64], device="cpu").input_size == (64, 64)

    def test_validate_input_size_invalid_string_length(self):
        """Wrong-length string raises ValueError."""
        with pytest.raises(ValueError, match="2 dimensions"):
            Config(input_size="32,32,3")

    def test_validate_input_size_invalid_tuple_length(self):
        """Wrong-length list raises ValueError."""
        from cnn_classifier.config import _parse_input_size

        with pytest.raises(ValueError, match="2 dimensions"):
            _parse_input_size([32, 32, 3])

    def test_parse_input_size_list_out_of_range(self):
        """List branch range check raises for zero/negative and over-max values."""
        from cnn_classifier.config import _parse_input_size

        with pytest.raises(ValueError, match="positive"):
            _parse_input_size([0, 32])
        with pytest.raises(ValueError, match="positive"):
            _parse_input_size([-1, 32])
        with pytest.raises(ValueError, match="positive and <= 4096"):
            _parse_input_size([5000, 32])
        with pytest.raises(ValueError, match="positive and <= 4096"):
            _parse_input_size([32, 4097])

    def test_image_and_batch_pixel_limit_exact_boundaries_accepted(self):
        """Configs at exactly MAX_IMAGE_PIXELS / MAX_BATCH_PIXELS are accepted."""
        # 1000 * 2000 = 2,000,000 == _MAX_IMAGE_PIXELS (accepted)
        config = Config(input_size="1000,2000", batch_size=1)
        assert config.input_size == (1000, 2000)

        # 1000 * 2000 * 50 = 100,000,000 == _MAX_BATCH_PIXELS (accepted)
        config = Config(input_size="1000,2000", batch_size=50)
        assert config.input_size == (1000, 2000)
        assert config.batch_size == 50

    def test_validate_input_size_zero_or_negative(self):
        """Zero or negative dimension raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            Config(input_size="0,32")
        with pytest.raises(ValueError, match="positive"):
            Config(input_size="-1,32")

    def test_validate_input_size_invalid_type(self):
        """Invalid type (dict, int, None) raises ValueError."""
        from cnn_classifier.config import _parse_input_size

        with pytest.raises(ValueError, match="tuple/list of 2 ints or comma-separated string"):
            _parse_input_size({"h": 32, "w": 32})
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
            _parse_input_size("32,abc")  # non-numeric
        with pytest.raises(ValueError):
            _parse_input_size("32.5,32")  # float
        with pytest.raises(ValueError):
            _parse_input_size("32")  # single value
        # Boundary max values
        assert _parse_input_size("4096,32") == (4096, 32)
        assert _parse_input_size("32,4096") == (32, 4096)
        # Whitespace variants
        assert _parse_input_size("\t32\n,\r32\t") == (32, 32)

    def test_image_pixel_limit(self):
        """Combined image pixels (H*W) must not exceed MAX_IMAGE_PIXELS."""
        # 4096 * 4096 = 16,777,216 > 2,000,000
        with pytest.raises(ValueError, match="exceeds max image pixels"):
            Config(input_size="4096,4096")

    def test_batch_pixel_limit(self):
        """Combined batch pixels (H*W * batch_size) must not exceed MAX_BATCH_PIXELS."""
        # 100 * 100 * 512 = 5,120,000 < 100,000,000 (OK)
        config = Config(input_size="100,100", batch_size=512)
        assert config.input_size == (100, 100)
        assert config.batch_size == 512

        # 300 * 300 * 512 = 46,080,000 < 100,000,000 (OK)
        config = Config(input_size="300,300", batch_size=512)
        assert config.batch_size == 512

        # 700 * 700 * 512 = 250,880,000 > 100,000,000 (exceeds)
        with pytest.raises(ValueError, match="exceeds max batch pixels"):
            Config(input_size="700,700", batch_size=512)

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

    @patch("torch.cuda.is_available", return_value=True)
    def test_device_auto_detects_cuda(self, _mock_cuda):
        """Device 'auto' returns CUDA when available."""
        config = Config(device="auto")
        assert config.device == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    def test_device_auto_detects_mps(self, _mock_mps, _mock_cuda):
        """Device 'auto' returns MPS when CUDA unavailable but MPS available."""
        config = Config(device="auto")
        assert config.device == "mps"

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_device_auto_detects_cpu(self, _mock_mps, _mock_cuda):
        """Device 'auto' returns CPU when neither CUDA nor MPS available."""
        config = Config(device="auto")
        assert config.device == "cpu"

    def test_device_auto_detects_cpu_without_torch(self):
        """Device 'auto' returns CPU when torch is not importable."""
        import sys

        with patch.dict(sys.modules, {"torch": None}):
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

    def test_device_cuda_without_torch_raises(self):
        """CUDA request without torch installed raises ValueError."""
        import sys

        with patch.dict(sys.modules, {"torch": None}):
            with pytest.raises(ValueError, match="PyTorch not installed"):
                Config(device="cuda")

    def test_device_cpu_without_torch_ok(self):
        """CPU request passes even when torch is not importable."""
        import sys

        with patch.dict(sys.modules, {"torch": None}):
            config = Config(device="cpu")
            assert config.device == "cpu"

    def test_model_path_resolved_relative(self):
        """Relative path anchors to project root and resolves to absolute."""
        config = Config(model_path="models/x.pth")
        resolved = config.model_path_resolved
        assert resolved.is_absolute()
        assert resolved.name == "x.pth"

    def test_labels_path_resolved_relative(self):
        """Labels path resolves to absolute."""
        config = Config()
        resolved = config.labels_path_resolved
        assert resolved.is_absolute()
        assert resolved.name == "labels.json"

    def test_relative_paths_anchored_to_project_root_not_cwd(self, tmp_path, monkeypatch):
        """Relative defaults anchor to project root regardless of process CWD."""
        from cnn_classifier.config import _PROJECT_ROOT

        monkeypatch.chdir(tmp_path)  # unrelated CWD, no CNN_PROJECT_ROOT override
        config = Config()
        assert config.model_path_resolved == _PROJECT_ROOT / "checkpoints" / "best_model.pt"
        assert config.labels_path_resolved == _PROJECT_ROOT / "checkpoints" / "labels.json"

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

    def test_env_file_values_load(self, tmp_path, monkeypatch):
        """.env file values are loaded (CNN_ prefixed, case-sensitive)."""
        env_file = tmp_path / ".env"
        env_file.write_text("CNN_PORT=9000\n")

        monkeypatch.setenv("CNN_PROJECT_ROOT", str(tmp_path))
        monkeypatch.chdir(tmp_path)
        config = Config()
        assert config.port == 9000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
