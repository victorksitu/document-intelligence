from pathlib import Path

from backend.app.image_loader import load_image
from backend.app.preprocessing import preprocess_image, save_processed_image


def preprocess_local_image(
    input_path: str | Path,
    output_path: str | Path,
    max_width: int = 1600,
    max_height: int = 1600,
) -> Path:
    """Load a local image, preprocess it, and save the processed output."""
    image = load_image(input_path)
    processed_image = preprocess_image(
        image,
        max_width=max_width,
        max_height=max_height,
    )
    return save_processed_image(processed_image, output_path)
