from pathlib import Path

import cv2
import numpy as np

SUPPORTED_OUTPUT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def resize_if_needed(
    image: np.ndarray,
    max_width: int = 1600,
    max_height: int = 1600,
) -> np.ndarray:
    """Downsize an image only when it exceeds the configured bounds."""
    _validate_image_array(image)

    if max_width <= 0 or max_height <= 0:
        raise ValueError("max_width and max_height must be positive integers")

    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)

    if scale == 1.0:
        return image.copy()

    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))

    return cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an OpenCV BGR/BGRA image to grayscale."""
    _validate_image_array(image)

    if image.ndim == 2:
        return image.copy()

    channels = image.shape[2]

    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Expected a 1, 3, or 4 channel image, got {channels} channels")


def preprocess_image(
    image: np.ndarray,
    max_width: int = 1600,
    max_height: int = 1600,
) -> np.ndarray:
    """Apply the first preprocessing slice: resize if needed, then grayscale."""
    resized = resize_if_needed(image, max_width=max_width, max_height=max_height)
    return to_grayscale(resized)


def save_processed_image(image: np.ndarray, output_path: str | Path) -> Path:
    """Save a processed image to a supported local image path."""
    _validate_image_array(image)
    path = _validate_output_image_path(output_path)

    try:
        saved = cv2.imwrite(str(path), image)
    except cv2.error as exc:
        raise ValueError(f"Image could not be saved to: {path}") from exc

    if not saved:
        raise ValueError(f"Image could not be saved to: {path}")

    return path


def _validate_image_array(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    if image.size == 0:
        raise ValueError("image must not be empty")

    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D grayscale or 3D color array")


def _validate_output_image_path(output_path: str | Path) -> Path:
    path = Path(output_path)

    if path.suffix.lower() not in SUPPORTED_OUTPUT_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_IMAGE_EXTENSIONS))
        raise ValueError(
            f"Unsupported output image type '{path.suffix}'. "
            f"Supported types: {supported}"
        )

    if path.exists() and path.is_dir():
        raise ValueError(f"Output path is a directory: {path}")

    if not path.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {path.parent}")

    if not path.parent.is_dir():
        raise ValueError(f"Output parent path is not a directory: {path.parent}")

    return path
