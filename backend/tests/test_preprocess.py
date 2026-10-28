"""Tests for cnn_classifier.inference.preprocess (Task 5)."""

import pytest
import torch
from PIL import Image

from cnn_classifier.inference.preprocess import preprocess_image


class TestPreprocessImage:
    """preprocess_image: PIL image -> normalized (1, 3, H, W) tensor."""

    def test_preprocess_resizes_and_normalizes(self):
        """Spec test: resize to input_size, normalize, output (1, 3, 32, 32)."""
        img = Image.new("RGB", (100, 100), color="red")
        tensor = preprocess_image(img, (32, 32))
        assert tensor.shape == (1, 3, 32, 32)
        assert tensor.min() >= -3 and tensor.max() <= 3

    def test_preprocess_converts_rgba_to_rgb(self):
        """RGBA input is converted to 3-channel RGB before the pipeline."""
        img = Image.new("RGBA", (64, 64), color=(255, 0, 0, 128))
        tensor = preprocess_image(img, (32, 32))
        assert tensor.shape == (1, 3, 32, 32)

    def test_preprocess_converts_grayscale_to_rgb(self):
        """Grayscale (L mode) input is converted to 3-channel RGB."""
        img = Image.new("L", (64, 64), color=128)
        tensor = preprocess_image(img, (32, 32))
        assert tensor.shape == (1, 3, 32, 32)

    def test_preprocess_output_dtype_and_finiteness(self):
        """Output is a finite float32 tensor with a leading batch dim."""
        img = Image.new("RGB", (50, 50), color="blue")
        tensor = preprocess_image(img, (32, 32))
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()
        assert tensor.shape[0] == 1
        assert tensor.shape[1] == 3

    def test_preprocess_cifar10_normalization_values(self):
        """Pure-red pixels normalize to the known CIFAR-10 stats values."""
        img = Image.new("RGB", (32, 32), color="red")
        tensor = preprocess_image(img, (32, 32))
        red = (1.0 - 0.4914) / 0.2470
        green = (0.0 - 0.4822) / 0.2435
        blue = (0.0 - 0.4465) / 0.2616
        assert tensor[0, 0].mean().item() == pytest.approx(red, abs=1e-4)
        assert tensor[0, 1].mean().item() == pytest.approx(green, abs=1e-4)
        assert tensor[0, 2].mean().item() == pytest.approx(blue, abs=1e-4)

    def test_preprocess_non_square_input(self):
        """Non-square inputs are resized to the exact (H, W) target."""
        img = Image.new("RGB", (64, 100), color="green")
        tensor = preprocess_image(img, (32, 32))
        assert tensor.shape == (1, 3, 32, 32)


class TestFixRoundOnePreprocess:
    """Fix-round-1: pixel cap and non-square targets."""

    def test_preprocess_pixel_cap_exceeded(self):
        """Oversized source images are rejected before decode-heavy work."""
        big = Image.new("RGB", (3000, 3000), color="green")
        with pytest.raises(ValueError, match="pixel cap"):
            preprocess_image(big, (32, 32))

    def test_preprocess_non_square_target(self):
        """Non-square targets resize to (1, 3, H, W) with H != W."""
        img = Image.new("RGB", (64, 100), color="blue")
        out = preprocess_image(img, (32, 48))
        assert out.shape == (1, 3, 32, 48)

    def test_pixel_cap_boundary_at_cap_allowed(self):
        """Exactly _MAX_IMAGE_PIXELS (2000x1000) is allowed - no off-by-one."""
        out = preprocess_image(Image.new("RGB", (2000, 1000), color="green"), (32, 32))
        assert out.shape == (1, 3, 32, 32)

    def test_pixel_cap_boundary_just_over_rejected(self):
        """One pixel over the cap (2000x1001) is rejected."""
        with pytest.raises(ValueError, match="pixel cap"):
            preprocess_image(Image.new("RGB", (2000, 1001), color="green"), (32, 32))

    def test_pixel_cap_enforced_on_lazy_image(self, tmp_path):
        """The cap fires before decode on a lazily-opened image file."""
        img_path = tmp_path / "big.png"
        Image.new("RGB", (2000, 1001), color="green").save(img_path)
        with pytest.raises(ValueError, match="pixel cap"):
            preprocess_image(Image.open(img_path), (32, 32))


class TestFixRoundSevenPreprocess:
    """Fix-round-7: mode-conversion flavors + upscale path."""

    @pytest.mark.parametrize(
        "mode",
        [
            ("RGBA", (255, 0, 0, 255)),
            ("L", 128),
            ("P", 0),
            ("CMYK", (0, 0, 0, 255)),
            ("LA", (128, 255)),
        ],
    )
    def test_preprocess_converts_palette_modes(self, mode):
        """All non-RGB modes convert to a 3-channel normalized tensor."""
        kind, fill = mode
        img = Image.new(kind, (64, 48), fill)
        out = preprocess_image(img, (32, 32))
        assert out.shape == (1, 3, 32, 32)

    def test_preprocess_upscales_small_source(self):
        """Sources smaller than the target upscale to the target size."""
        out = preprocess_image(Image.new("RGB", (16, 16), color="red"), (32, 32))
        assert out.shape == (1, 3, 32, 32)
