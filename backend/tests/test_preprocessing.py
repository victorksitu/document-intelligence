from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pytest

from backend.app.preprocessing import (
    apply_binary_threshold,
    apply_median_denoise,
    preprocess_image,
    resize_if_needed,
    save_processed_image,
    to_grayscale,
)


@contextmanager
def local_temp_dir() -> Generator[Path, None, None]:
    with TemporaryDirectory(
        prefix=".preprocessing-test-",
        dir=Path(__file__).parent,
    ) as directory:
        yield Path(directory)


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


def test_apply_binary_threshold_converts_grayscale_to_black_and_white() -> None:
    image = np.array([[0, 127, 128, 255]], dtype=np.uint8)

    thresholded = apply_binary_threshold(image, threshold_value=127)

    assert thresholded.tolist() == [[0, 0, 255, 255]]
    assert thresholded.dtype == np.uint8


def test_apply_binary_threshold_rejects_color_images() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="requires a grayscale image"):
        apply_binary_threshold(image)


def test_apply_binary_threshold_rejects_invalid_threshold_value() -> None:
    image = np.zeros((10, 20), dtype=np.uint8)

    with pytest.raises(ValueError, match="threshold_value"):
        apply_binary_threshold(image, threshold_value=300)


def test_preprocess_image_can_apply_threshold_after_grayscale() -> None:
    image = np.array([[0, 128, 255]], dtype=np.uint8)

    processed = preprocess_image(image, apply_threshold=True, threshold_value=127)

    assert processed.tolist() == [[0, 255, 255]]


def test_apply_median_denoise_reduces_single_pixel_noise() -> None:
    image = np.full((5, 5), fill_value=200, dtype=np.uint8)
    image[2, 2] = 0

    denoised = apply_median_denoise(image, kernel_size=3)

    assert denoised[2, 2] == 200
    assert denoised.shape == image.shape
    assert denoised.dtype == np.uint8


def test_apply_median_denoise_rejects_even_kernel_size() -> None:
    image = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="kernel_size"):
        apply_median_denoise(image, kernel_size=4)


def test_apply_median_denoise_rejects_too_small_kernel_size() -> None:
    image = np.zeros((5, 5), dtype=np.uint8)

    with pytest.raises(ValueError, match="kernel_size"):
        apply_median_denoise(image, kernel_size=1)


def test_preprocess_image_can_denoise_before_thresholding() -> None:
    image = np.full((5, 5), fill_value=200, dtype=np.uint8)
    image[2, 2] = 0

    processed = preprocess_image(
        image,
        apply_denoise=True,
        denoise_kernel_size=3,
        apply_threshold=True,
        threshold_value=127,
    )

    assert processed.tolist() == np.full((5, 5), fill_value=255).tolist()


def test_save_processed_image_writes_readable_png() -> None:
    with local_temp_dir() as temp_dir:
        output_path = temp_dir / "processed.png"
        image = np.full((10, 20), fill_value=180, dtype=np.uint8)

        saved_path = save_processed_image(image, output_path)
        reloaded = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)

        assert saved_path == output_path
        assert output_path.exists()
        assert reloaded is not None
        assert reloaded.shape == image.shape
        assert reloaded.dtype == np.uint8


def test_save_processed_image_rejects_unsupported_extension() -> None:
    with local_temp_dir() as temp_dir:
        output_path = temp_dir / "processed.txt"
        image = np.zeros((10, 20), dtype=np.uint8)

        with pytest.raises(ValueError, match="Unsupported output image type"):
            save_processed_image(image, output_path)


def test_save_processed_image_rejects_missing_output_directory() -> None:
    with local_temp_dir() as temp_dir:
        output_path = temp_dir / "missing" / "processed.png"
        image = np.zeros((10, 20), dtype=np.uint8)

        with pytest.raises(FileNotFoundError, match="Output directory does not exist"):
            save_processed_image(image, output_path)


def test_save_processed_image_raises_when_opencv_save_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with local_temp_dir() as temp_dir:
        output_path = temp_dir / "processed.png"
        image = np.zeros((10, 20), dtype=np.uint8)

        monkeypatch.setattr(cv2, "imwrite", lambda *_args: False)

        with pytest.raises(ValueError, match="could not be saved"):
            save_processed_image(image, output_path)
