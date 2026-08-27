from pathlib import Path

import cv2
import numpy as np

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_image_path(image_path: str | Path) -> Path:
    """Validate that an image path exists and has a supported extension."""
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        raise ValueError(
            f"Unsupported image type '{path.suffix}'. Supported types: {supported}"
        )

    return path


def load_image(image_path: str | Path) -> np.ndarray:
    """Load a JPG or PNG image as an OpenCV BGR NumPy array."""
    path = validate_image_path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Image file could not be read as an image: {path}")

    return image
