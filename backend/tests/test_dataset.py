"""Tests for cnn_classifier.data (CIFAR-10 loaders + CustomImageFolder)."""

import os
import struct

import pytest
import torch
from PIL import Image
from torchvision import transforms

from cnn_classifier.config import Config
from cnn_classifier.data import CIFAR10_MEAN, CIFAR10_STD, CustomImageFolder, get_dataloaders
from cnn_classifier.data.dataset import (
    _build_eval_transform,
    _build_train_transform,
    _stratified_split,
)


@pytest.fixture(scope="session")
def cifar_data_dir(tmp_path_factory):
    """Data dir for the CIFAR-10 download.

    Env override CNN_TEST_DATA_DIR allows caching the ~170MB download
    across local runs; CI reuses the cached dataset dir
    (/home/runner/.cache/cnn-cifar10) via the same env var.
    """
    env_dir = os.environ.get("CNN_TEST_DATA_DIR")
    if env_dir:
        return env_dir
    return tmp_path_factory.mktemp("cifar_data")


@pytest.fixture(scope="session")
def loaders(cifar_data_dir):
    """Shared train/val/test loaders built once per test session."""
    cfg = Config(batch_size=4, num_workers=0)
    return get_dataloaders(cfg, data_dir=cifar_data_dir)


class TestCIFAR10Constants:
    """CIFAR-10 normalization constants."""

    def test_mean_has_three_channels(self):
        assert len(CIFAR10_MEAN) == 3

    def test_std_has_three_channels(self):
        assert len(CIFAR10_STD) == 3

    def test_mean_values_match_cifar10(self):
        assert CIFAR10_MEAN == (0.4914, 0.4822, 0.4465)

    def test_std_values_match_cifar10(self):
        assert CIFAR10_STD == (0.2470, 0.2435, 0.2616)


class TestStratifiedSplit:
    """Unit tests for the stratified train/val index split."""

    def test_split_sizes(self):
        targets = [0] * 10 + [1] * 10 + [2] * 10
        train_idx, val_idx = _stratified_split(targets, val_ratio=0.2, seed=42)
        assert len(train_idx) == 24
        assert len(val_idx) == 6

    def test_split_is_stratified_per_class(self):
        targets = [0] * 10 + [1] * 10
        _, val_idx = _stratified_split(targets, val_ratio=0.2, seed=42)
        val_targets = [targets[i] for i in val_idx]
        assert val_targets.count(0) == 2
        assert val_targets.count(1) == 2

    def test_split_disjoint_and_complete(self):
        targets = [0] * 7 + [1] * 5
        train_idx, val_idx = _stratified_split(targets, val_ratio=0.5, seed=0)
        assert len(val_idx) == 5  # non-empty: 3 from class 0, 2 from class 1
        assert set(train_idx) & set(val_idx) == set()
        assert sorted(train_idx + val_idx) == list(range(len(targets)))

    def test_split_reproducible_with_same_seed(self):
        targets = [i % 3 for i in range(30)]
        train_a, val_a = _stratified_split(targets, val_ratio=0.3, seed=7)
        train_b, val_b = _stratified_split(targets, val_ratio=0.3, seed=7)
        assert train_a == train_b
        assert val_a == val_b

    def test_split_varies_with_different_seed(self):
        targets = [i % 4 for i in range(40)]
        _, val_a = _stratified_split(targets, val_ratio=0.25, seed=1)
        _, val_b = _stratified_split(targets, val_ratio=0.25, seed=2)
        assert val_a != val_b

    def test_val_ratio_zero_gives_empty_val(self):
        targets = [0, 1, 0, 1]
        train_idx, val_idx = _stratified_split(targets, val_ratio=0.0, seed=42)
        assert val_idx == []
        assert len(train_idx) == 4

    def test_small_class_gives_zero_val_samples(self):
        """Single-sample classes contribute nothing to val (int(1 * 0.5) == 0)."""
        train_idx, val_idx = _stratified_split([0, 1], val_ratio=0.5, seed=0)
        assert val_idx == []
        assert sorted(train_idx) == [0, 1]

    def test_direct_negative_ratio_raises(self):
        """The internal guard fires even on direct calls, not just via the wrapper."""
        with pytest.raises(ValueError, match="val_ratio"):
            _stratified_split([0, 1, 2], val_ratio=-0.1)
        with pytest.raises(ValueError, match="val_ratio"):
            _stratified_split([0, 1, 2], val_ratio=1.0)


class TestGetDataloaders:
    """get_dataloaders returns train/val/test loaders with correct shapes."""

    def test_cifar10_loaders_shapes(self, loaders):
        """Plan TDD step 1: shapes and dtypes of one train batch."""
        train_loader, val_loader, test_loader = loaders
        x, y = next(iter(train_loader))
        assert x.shape == (4, 3, 32, 32)
        assert y.shape == (4,)
        assert x.dtype == torch.float32
        assert y.dtype == torch.long

    def test_returns_train_val_test_sizes(self, loaders):
        """90/10 stratified split of 50k train + 10k test CIFAR-10."""
        train_loader, val_loader, test_loader = loaders
        assert len(train_loader.dataset) == 45000
        assert len(val_loader.dataset) == 5000
        assert len(test_loader.dataset) == 10000

    def test_train_loader_shuffled(self, loaders):
        train_loader, _, _ = loaders
        assert isinstance(train_loader.sampler, torch.utils.data.RandomSampler)

    def test_val_and_test_not_shuffled(self, loaders):
        _, val_loader, test_loader = loaders
        assert isinstance(val_loader.sampler, torch.utils.data.SequentialSampler)
        assert isinstance(test_loader.sampler, torch.utils.data.SequentialSampler)

    def test_batch_size_from_config(self, loaders):
        train_loader, _, _ = loaders
        assert train_loader.batch_size == 4

    def test_batch_count_matches_dataset_size(self, loaders):
        train_loader, val_loader, _ = loaders
        assert len(train_loader) == 45000 // 4
        assert len(val_loader) == 5000 // 4

    def test_labels_in_class_range(self, loaders):
        train_loader, _, test_loader = loaders
        for loader in (train_loader, test_loader):
            _, y = next(iter(loader))
            assert ((y >= 0) & (y <= 9)).all()

    def test_train_batch_normalized_and_finite(self, loaders):
        train_loader, _, _ = loaders
        x, _ = next(iter(train_loader))
        assert torch.isfinite(x).all()
        assert float(x.min()) >= -5.0
        assert float(x.max()) <= 5.0

    def test_eval_batch_normalized(self, loaders):
        """Val batches carry no augmentation: values stay in normalized band."""
        _, val_loader, _ = loaders
        x, _ = next(iter(val_loader))
        assert torch.isfinite(x).all()
        assert float(x.min()) >= -2.5
        assert float(x.max()) <= 2.5
        # Normalize provably ran: raw [0, 1] pixels would never go negative.
        assert float(x.min()) < 0

    def test_default_data_dir_resolves_to_project_root(self, monkeypatch, tmp_path):
        """data_dir=None anchors CIFAR-10 at <project_root>/data (hermetic)."""
        from cnn_classifier.data import dataset as dataset_module

        captured_roots: list[str] = []

        class FakeCIFAR10:
            def __init__(self, root, train, download, transform):
                captured_roots.append(root)
                self.targets = [0] * 10 + [1] * 10

            def __getitem__(self, idx):
                return torch.zeros(3, 32, 32), 0

            def __len__(self):
                return 20

        monkeypatch.setattr(dataset_module.datasets, "CIFAR10", FakeCIFAR10)
        monkeypatch.setattr(dataset_module, "_get_project_root", lambda: tmp_path)
        cfg = Config(batch_size=4, num_workers=0)
        result = get_dataloaders(cfg)
        assert len(result) == 3
        assert captured_roots == [str(tmp_path / "data")] * 3

    def test_val_ratio_must_be_non_negative(self, cifar_data_dir):
        cfg = Config(batch_size=4, num_workers=0)
        with pytest.raises(ValueError, match="val_ratio"):
            get_dataloaders(cfg, data_dir=cifar_data_dir, val_ratio=-0.1)

    def test_val_ratio_must_be_below_one(self, cifar_data_dir):
        cfg = Config(batch_size=4, num_workers=0)
        with pytest.raises(ValueError, match="val_ratio"):
            get_dataloaders(cfg, data_dir=cifar_data_dir, val_ratio=1.0)

    def test_wrapper_val_ratio_zero_yields_empty_val(self, monkeypatch, tmp_path):
        """Wrapper-level val_ratio=0.0: val loader dataset is empty (hermetic)."""
        from cnn_classifier.data import dataset as dataset_module

        class FakeCIFAR10:
            def __init__(self, root, train, download, transform):
                self.targets = [0] * 10 + [1] * 10

            def __getitem__(self, idx):
                return torch.zeros(3, 32, 32), 0

            def __len__(self):
                return 20

        monkeypatch.setattr(dataset_module.datasets, "CIFAR10", FakeCIFAR10)
        monkeypatch.setattr(dataset_module, "_get_project_root", lambda: tmp_path)
        cfg = Config(batch_size=4, num_workers=0)
        _, val_loader, _ = get_dataloaders(cfg, val_ratio=0.0)
        assert len(val_loader.dataset) == 0

    def test_input_size_must_be_32x32(self, cifar_data_dir):
        """Loaders fail fast on non-CIFAR input sizes (eval has no Resize)."""
        cfg = Config(input_size=(64, 64), batch_size=4, num_workers=0)
        with pytest.raises(ValueError, match="input_size"):
            get_dataloaders(cfg, data_dir=cifar_data_dir)

    def test_train_loader_reproducible_given_seed(self, cifar_data_dir):
        """Same config + same global RNG state -> identical first train batch."""
        cfg = Config(batch_size=4, num_workers=0, seed=123)
        train_a, _, _ = get_dataloaders(cfg, data_dir=cifar_data_dir)
        torch.manual_seed(999)
        x_a, y_a = next(iter(train_a))
        train_b, _, _ = get_dataloaders(cfg, data_dir=cifar_data_dir)
        torch.manual_seed(999)
        x_b, y_b = next(iter(train_b))
        assert torch.equal(y_a, y_b)  # generator-seeded sampler order
        assert torch.equal(x_a, x_b)  # manual_seed-governed augmentation


class TestTransforms:
    """Structural + behavioral tests for the train/eval transform builders."""

    @staticmethod
    def _varied_image() -> Image.Image:
        """A spatially varied image so crop/flip changes are detectable."""
        img = Image.new("RGB", (32, 32))
        for x in range(32):
            for y in range(32):
                img.putpixel((x, y), (x * 8 % 256, y * 8 % 256, (x + y) * 4 % 256))
        return img

    def test_train_transform_structure(self):
        t = _build_train_transform((32, 32))
        assert [type(step) for step in t.transforms] == [
            transforms.RandomCrop,
            transforms.RandomHorizontalFlip,
            transforms.ToTensor,
            transforms.Normalize,
        ]
        assert t.transforms[0].padding == 4
        assert t.transforms[0].size == (32, 32)

    def test_eval_transform_structure(self):
        t = _build_eval_transform()
        assert [type(step) for step in t.transforms] == [
            transforms.ToTensor,
            transforms.Normalize,
        ]

    def test_train_and_eval_builders_differ(self):
        train_types = [type(s) for s in _build_train_transform((32, 32)).transforms]
        eval_types = [type(s) for s in _build_eval_transform().transforms]
        assert train_types != eval_types

    def test_train_transform_is_stochastic(self):
        t = _build_train_transform((32, 32))
        img = self._varied_image()
        torch.manual_seed(0)
        a = t(img)
        torch.manual_seed(1)
        b = t(img)
        assert not torch.equal(a, b)

    def test_train_transform_deterministic_given_seed(self):
        t = _build_train_transform((32, 32))
        img = self._varied_image()
        torch.manual_seed(42)
        a = t(img)
        torch.manual_seed(42)
        b = t(img)
        assert torch.equal(a, b)

    def test_eval_transform_is_deterministic(self):
        t = _build_eval_transform()
        img = self._varied_image()
        assert torch.equal(t(img), t(img))


class TestCustomImageFolder:
    """CustomImageFolder loads class-folder image trees."""

    @pytest.fixture
    def image_root(self, tmp_path):
        for class_name in ("cat", "dog"):
            class_dir = tmp_path / class_name
            class_dir.mkdir()
            for i in range(3):
                img = Image.new("RGB", (32, 32), color=(i * 60, i * 80, i * 100))
                img.save(class_dir / f"{class_name}_{i}.jpg")
        return tmp_path

    def test_custom_dataset_loads_folder(self, image_root):
        """Plan TDD step 1: fake class folders with dummy images load."""
        dataset = CustomImageFolder(root=str(image_root))
        assert len(dataset) == 6
        assert sorted(dataset.classes) == ["cat", "dog"]

    def test_returns_pil_image_without_transform(self, image_root):
        dataset = CustomImageFolder(root=str(image_root))
        img, label = dataset[0]
        assert label in (0, 1)
        assert img.size == (32, 32)

    def test_applies_transform(self, image_root):
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = CustomImageFolder(root=str(image_root), transform=transform)
        img, _ = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 32, 32)

    def test_class_to_idx_mapping(self, image_root):
        dataset = CustomImageFolder(root=str(image_root))
        assert dataset.class_to_idx == {"cat": 0, "dog": 1}


class TestSafeLoader:
    """Hardened scan + loader: pixel cap, allowlist, symlink containment."""

    def test_oversized_image_raises(self, tmp_path, monkeypatch):
        from cnn_classifier.data import custom_dataset as custom_dataset_module

        (tmp_path / "cat").mkdir()
        Image.new("RGB", (32, 32), color=(10, 20, 30)).save(tmp_path / "cat" / "img.jpg")
        monkeypatch.setattr(custom_dataset_module, "_MAX_IMAGE_PIXELS", 100)
        dataset = CustomImageFolder(root=str(tmp_path))
        with pytest.raises(ValueError, match="pixel cap"):
            dataset[0]

    def test_symlinked_image_rejected(self, tmp_path):
        """Leaf symlinks are rejected by the loader (target inside root)."""
        root = tmp_path / "classes"
        (root / "cat").mkdir(parents=True)
        Image.new("RGB", (32, 32), color=(120, 120, 120)).save(root / "real.jpg")
        (root / "cat" / "link.jpg").symlink_to(root / "real.jpg")
        dataset = CustomImageFolder(root=str(root))
        with pytest.raises(ValueError, match="symlink"):
            dataset[0]

    def test_symlinked_class_dir_rejected(self, tmp_path):
        """MED-1 regression: symlinked class dirs are rejected at scan time
        (before the directory walk can follow them outside the root)."""
        outside = tmp_path / "outside"
        outside.mkdir()
        Image.new("RGB", (32, 32), color=(5, 5, 5)).save(outside / "img.jpg")
        root = tmp_path / "classes"
        root.mkdir()
        (root / "cat").symlink_to(outside)
        with pytest.raises(ValueError, match="symlinked class directory"):
            CustomImageFolder(root=str(root))

    def test_symlinked_class_dir_cycle_rejected(self, tmp_path):
        """HIGH-1 regression: back-pointing class-dir symlinks (cycle) are
        rejected quickly at scan instead of hanging the walk forever."""
        root = tmp_path / "classes"
        root.mkdir()
        (root / "a").symlink_to(root)
        (root / "b").symlink_to(root)
        with pytest.raises(ValueError, match="symlinked class directory"):
            CustomImageFolder(root=str(root))

    def test_self_loop_class_dir_rejected(self, tmp_path):
        """HIGH-1 regression (mild form): a class dir looping back to root is
        rejected at scan instead of silently inflating len() via the walk."""
        root = tmp_path / "classes"
        root.mkdir()
        (root / "loop").symlink_to(root)
        with pytest.raises(ValueError, match="symlinked class directory"):
            CustomImageFolder(root=str(root))

    def test_outside_root_leaf_symlink_rejected(self, tmp_path):
        """MED-1 containment regression: a leaf symlink pointing outside the
        dataset root passes scan (stat follows links, so S_ISREG is True) and
        is rejected by the post-init containment check."""
        outside = tmp_path / "outside"
        outside.mkdir()
        Image.new("RGB", (32, 32), color=(7, 7, 7)).save(outside / "real.jpg")
        root = tmp_path / "classes"
        (root / "cat").mkdir(parents=True)
        (root / "cat" / "escape.jpg").symlink_to(outside / "real.jpg")
        with pytest.raises(ValueError, match="escapes dataset root"):
            CustomImageFolder(root=str(root))

    def test_is_regular_file_missing_path_returns_false(self, tmp_path):
        """_is_regular_file returns False when os.stat fails (vanished file),
        covering the OSError branch of the regular-file check."""
        from cnn_classifier.data.custom_dataset import _is_regular_file

        assert _is_regular_file(str(tmp_path / "missing.jpg")) is False

    def test_disallowed_extension_excluded_at_scan(self, tmp_path):
        """MED-2: scan-time allowlist — a class dir whose only files are
        unsupported formats fails loudly at init (torchvision semantics),
        proving the scanner excluded them before load time."""
        root = tmp_path / "classes"
        (root / "cat").mkdir(parents=True)
        (root / "cat" / "img.tiff").write_bytes(b"not really a tiff")
        with pytest.raises(FileNotFoundError, match="no valid file"):
            CustomImageFolder(root=str(root))

    def test_mixed_folder_counts_only_loadable(self, tmp_path):
        """MED-2: a stray unsupported file neither inflates len() nor aborts iteration."""
        cat = tmp_path / "cat"
        cat.mkdir()
        Image.new("RGB", (32, 32), color=(9, 9, 9)).save(cat / "valid.jpg")
        (cat / "stray.bmp").write_bytes(b"BM fake")
        dataset = CustomImageFolder(root=str(tmp_path), transform=transforms.ToTensor())
        assert len(dataset) == 1
        img, _ = dataset[0]  # iteration works, no abort
        assert img.shape == (3, 32, 32)

    def test_fifo_image_rejected(self, tmp_path):
        """MED-4 regression: special files (FIFOs) are excluded at scan and
        rejected by the loader instead of blocking Image.open forever."""
        from cnn_classifier.data import custom_dataset as custom_dataset_module

        cat = tmp_path / "cat"
        cat.mkdir()
        Image.new("RGB", (32, 32), color=(7, 7, 7)).save(cat / "valid.jpg")
        os.mkfifo(cat / "pipe.jpg")
        # Scan excludes the FIFO: len() counts only the regular file.
        dataset = CustomImageFolder(root=str(tmp_path))
        assert len(dataset) == 1
        # Loader guard (defense-in-depth) rejects it directly too.
        with pytest.raises(ValueError, match="not a regular image file"):
            custom_dataset_module._safe_loader(str(cat / "pipe.jpg"))

    def test_truncated_image_raises_value_error(self, tmp_path):
        """MED-3 regression: a header-valid but tail-truncated image raises
        the module ValueError contract, not a raw OSError from convert."""
        cat = tmp_path / "cat"
        cat.mkdir()
        full = tmp_path / "full.jpg"
        Image.new("RGB", (64, 64), color=(3, 3, 3)).save(full)
        data = full.read_bytes()
        (cat / "cut.jpg").write_bytes(data[: len(data) - 20])
        dataset = CustomImageFolder(root=str(tmp_path))
        with pytest.raises(ValueError, match="unreadable image file"):
            dataset[0]

    def test_decompression_bomb_raises_value_error(self, tmp_path):
        """MED-3 regression: a header-declared bomb (beyond PIL's own limit)
        raises the module ValueError contract, not a raw DecompressionBombError."""
        cat = tmp_path / "cat"
        cat.mkdir()
        full = tmp_path / "full.png"
        Image.new("RGB", (16, 16)).save(full)
        bomb = bytearray(full.read_bytes())
        # Patch IHDR width/height (big-endian u32 pair at offset 16) to 50000x50000.
        bomb[16:24] = struct.pack(">II", 50000, 50000)
        (cat / "bomb.png").write_bytes(bytes(bomb))
        dataset = CustomImageFolder(root=str(tmp_path))
        with pytest.raises(ValueError, match="unreadable image file"):
            dataset[0]

    def test_uppercase_extension_loads(self, tmp_path):
        """Case-insensitive suffix handling: .JPG loads."""
        cat = tmp_path / "cat"
        cat.mkdir()
        Image.new("RGB", (32, 32), color=(1, 2, 3)).save(cat / "upper.JPG")
        dataset = CustomImageFolder(root=str(tmp_path))
        assert len(dataset) == 1
        img, _ = dataset[0]
        assert img.size == (32, 32)

    def test_l_mode_png_converted_to_rgb(self, tmp_path):
        """Grayscale PNGs load as 3-channel RGB tensors."""
        cat = tmp_path / "cat"
        cat.mkdir()
        Image.new("L", (32, 32), color=128).save(cat / "gray.png")
        dataset = CustomImageFolder(root=str(tmp_path), transform=transforms.ToTensor())
        img, _ = dataset[0]
        assert img.shape == (3, 32, 32)

    def test_corrupt_image_raises_value_error(self, tmp_path):
        """Corrupt payloads raise the module ValueError contract, not raw OSError."""
        cat = tmp_path / "cat"
        cat.mkdir()
        (cat / "corrupt.jpg").write_bytes(b"this is not a jpeg")
        dataset = CustomImageFolder(root=str(tmp_path))
        with pytest.raises(ValueError, match="unreadable image file"):
            dataset[0]

    def test_safe_loader_direct_rejects_bad_extension(self, tmp_path):
        """Direct unit: _safe_loader's load-time extension check (defense-in-depth;
        unreachable via dataset flow because scan-time is_valid_file filters first)."""
        from cnn_classifier.data import custom_dataset as custom_dataset_module

        bad = tmp_path / "img.tiff"
        bad.write_bytes(b"dummy")
        with pytest.raises(ValueError, match="unsupported image extension"):
            custom_dataset_module._safe_loader(str(bad))

    def test_data_loader_with_workers_smoke(self, tmp_path):
        """num_workers>0 regime: workers spawn and iterate the full folder."""
        from torch.utils.data import DataLoader

        cat = tmp_path / "cat"
        cat.mkdir()
        for i in range(3):
            Image.new("RGB", (32, 32), color=(i * 40, 0, 0)).save(cat / f"img{i}.jpg")
        dataset = CustomImageFolder(root=str(tmp_path), transform=transforms.ToTensor())
        loader = DataLoader(dataset, batch_size=2, num_workers=2)
        total = sum(x.shape[0] for x, _ in loader)
        assert total == 3


class TestPackageExports:
    """Package surface: __all__ matches the shipped public names exactly."""

    def test_all_matches_public_names(self):
        import cnn_classifier.data as data_pkg

        assert set(data_pkg.__all__) == {
            "CIFAR10_MEAN",
            "CIFAR10_STD",
            "CustomImageFolder",
            "get_dataloaders",
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
