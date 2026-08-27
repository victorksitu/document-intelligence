from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pytest

from backend.app.image_loader import load_image, validate_image_path


@contextmanager
def local_temp_dir() -> Generator[Path, None, None]:
    with TemporaryDirectory(
        prefix=".image-loader-test-",
        dir=Path(__file__).parent,
    ) as directory:
        yield Path(directory)


def test_validate_image_path_accepts_supported_image() -> None:
    with local_temp_dir() as temp_dir:
        image_path = temp_dir / "document.PNG"
        image_path.write_bytes(b"placeholder")

        assert validate_image_path(image_path) == image_path


def test_validate_image_path_rejects_missing_file() -> None:
    with local_temp_dir() as temp_dir:
        missing_path = temp_dir / "missing.png"

        with pytest.raises(FileNotFoundError):
            validate_image_path(missing_path)


def test_validate_image_path_rejects_unsupported_extension() -> None:
    with local_temp_dir() as temp_dir:
        text_path = temp_dir / "notes.txt"
        text_path.write_text("not an image")

        with pytest.raises(ValueError, match="Unsupported image type"):
            validate_image_path(text_path)


def test_load_image_returns_opencv_array_for_png() -> None:
    with local_temp_dir() as temp_dir:
        image_path = temp_dir / "document.png"
        original = np.full((12, 20, 3), fill_value=128, dtype=np.uint8)
        assert cv2.imwrite(str(image_path), original)

        loaded = load_image(image_path)

        assert isinstance(loaded, np.ndarray)
        assert loaded.shape == original.shape
        assert loaded.dtype == np.uint8


def test_load_image_rejects_corrupt_supported_file() -> None:
    with local_temp_dir() as temp_dir:
        corrupt_path = temp_dir / "corrupt.jpg"
        corrupt_path.write_bytes(b"this is not a real jpg")

        with pytest.raises(ValueError, match="could not be read"):
            load_image(corrupt_path)
