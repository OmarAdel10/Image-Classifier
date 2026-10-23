"""Tests for cnn_classifier.data (CIFAR-10 loaders + CustomImageFolder)."""

import os
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

from cnn_classifier.config import Config
from cnn_classifier.data import CIFAR10_MEAN, CIFAR10_STD, CustomImageFolder, get_dataloaders
from cnn_classifier.data.dataset import _stratified_split


@pytest.fixture(scope="session")
def cifar_data_dir(tmp_path_factory):
    """Data dir for the CIFAR-10 download.

    Env override CNN_TEST_DATA_DIR allows caching the ~170MB download
    across local runs; CI uses a fresh per-session tmp dir.
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
        train_idx, val_idx = _stratified_split(targets, val_ratio=0.1, seed=0)
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

    def test_default_data_dir_resolves_to_project_root(self, monkeypatch, cifar_data_dir):
        """data_dir=None anchors the download dir at the project root."""
        from cnn_classifier.data import dataset as dataset_module

        # Reuse the session-cached dataset: link the archive and extracted
        # folder into the default dir so no network download is triggered.
        default_dir = Path(cifar_data_dir) / "data"
        default_dir.mkdir(parents=True, exist_ok=True)
        batches_link = default_dir / "cifar-10-batches-py"
        archive_link = default_dir / "cifar-10-python.tar.gz"
        if not batches_link.exists():
            batches_link.symlink_to(Path(cifar_data_dir) / "cifar-10-batches-py")
        if not archive_link.exists():
            archive_link.symlink_to(Path(cifar_data_dir) / "cifar-10-python.tar.gz")
        monkeypatch.setattr(dataset_module, "_get_project_root", lambda: Path(cifar_data_dir))
        cfg = Config(batch_size=4, num_workers=0)
        result = get_dataloaders(cfg)
        assert len(result) == 3

    def test_val_ratio_must_be_non_negative(self, cifar_data_dir):
        cfg = Config(batch_size=4, num_workers=0)
        with pytest.raises(ValueError, match="val_ratio"):
            get_dataloaders(cfg, data_dir=cifar_data_dir, val_ratio=-0.1)

    def test_val_ratio_must_be_below_one(self, cifar_data_dir):
        cfg = Config(batch_size=4, num_workers=0)
        with pytest.raises(ValueError, match="val_ratio"):
            get_dataloaders(cfg, data_dir=cifar_data_dir, val_ratio=1.0)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
