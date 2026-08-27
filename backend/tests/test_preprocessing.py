import numpy as np
import pytest

from backend.app.preprocessing import preprocess_image, resize_if_needed, to_grayscale


def test_resize_if_needed_keeps_small_image_size() -> None:
    image = np.zeros((80, 120, 3), dtype=np.uint8)

    resized = resize_if_needed(image, max_width=200, max_height=200)

    assert resized.shape == image.shape
    assert resized is not image


def test_resize_if_needed_downsizes_large_image_proportionally() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    resized = resize_if_needed(image, max_width=100, max_height=100)

    assert resized.shape == (50, 100, 3)


def test_resize_if_needed_rejects_invalid_bounds() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="positive integers"):
        resize_if_needed(image, max_width=0, max_height=100)


def test_to_grayscale_converts_bgr_image_to_2d_array() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    grayscale = to_grayscale(image)

    assert grayscale.shape == (10, 20)
    assert grayscale.dtype == np.uint8


def test_to_grayscale_keeps_existing_grayscale_shape() -> None:
    image = np.zeros((10, 20), dtype=np.uint8)

    grayscale = to_grayscale(image)

    assert grayscale.shape == image.shape
    assert grayscale is not image


def test_preprocess_image_resizes_then_converts_to_grayscale() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    processed = preprocess_image(image, max_width=100, max_height=100)

    assert processed.shape == (50, 100)
    assert processed.dtype == np.uint8
