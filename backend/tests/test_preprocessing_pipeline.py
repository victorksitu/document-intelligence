from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pytest

from backend.app.preprocessing_pipeline import preprocess_local_image


@contextmanager
def local_temp_dir() -> Generator[Path, None, None]:
    with TemporaryDirectory(
        prefix=".preprocessing-pipeline-test-",
        dir=Path(__file__).parent,
    ) as directory:
        yield Path(directory)


def test_preprocess_local_image_saves_grayscale_output() -> None:
    with local_temp_dir() as temp_dir:
        input_path = temp_dir / "document.png"
        output_path = temp_dir / "processed.png"
        original = np.full((100, 200, 3), fill_value=200, dtype=np.uint8)
        assert cv2.imwrite(str(input_path), original)

        saved_path = preprocess_local_image(
            input_path,
            output_path,
            max_width=100,
            max_height=100,
        )
        reloaded = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)

        assert saved_path == output_path
        assert output_path.exists()
        assert reloaded is not None
        assert reloaded.shape == (50, 100)
        assert reloaded.dtype == np.uint8


def test_preprocess_local_image_can_save_thresholded_output() -> None:
    with local_temp_dir() as temp_dir:
        input_path = temp_dir / "document.png"
        output_path = temp_dir / "thresholded.png"
        original = np.array([[[0, 0, 0], [200, 200, 200]]], dtype=np.uint8)
        assert cv2.imwrite(str(input_path), original)

        preprocess_local_image(
            input_path,
            output_path,
            apply_threshold=True,
            threshold_value=127,
        )
        reloaded = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)

        assert reloaded is not None
        assert reloaded.tolist() == [[0, 255]]


def test_preprocess_local_image_can_save_denoised_thresholded_output() -> None:
    with local_temp_dir() as temp_dir:
        input_path = temp_dir / "document.png"
        output_path = temp_dir / "denoised.png"
        original = np.full((5, 5, 3), fill_value=200, dtype=np.uint8)
        original[2, 2] = [0, 0, 0]
        assert cv2.imwrite(str(input_path), original)

        preprocess_local_image(
            input_path,
            output_path,
            apply_denoise=True,
            denoise_kernel_size=3,
            apply_threshold=True,
            threshold_value=127,
        )
        reloaded = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)

        assert reloaded is not None
        assert reloaded.tolist() == np.full((5, 5), fill_value=255).tolist()


def test_preprocess_local_image_rejects_missing_input() -> None:
    with local_temp_dir() as temp_dir:
        input_path = temp_dir / "missing.png"
        output_path = temp_dir / "processed.png"

        with pytest.raises(FileNotFoundError):
            preprocess_local_image(input_path, output_path)


def test_preprocess_local_image_rejects_unsupported_output_extension() -> None:
    with local_temp_dir() as temp_dir:
        input_path = temp_dir / "document.png"
        output_path = temp_dir / "processed.txt"
        original = np.zeros((10, 20, 3), dtype=np.uint8)
        assert cv2.imwrite(str(input_path), original)

        with pytest.raises(ValueError, match="Unsupported output image type"):
            preprocess_local_image(input_path, output_path)

        assert not output_path.exists()
