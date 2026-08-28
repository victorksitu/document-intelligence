from argparse import ArgumentParser, Namespace
from pathlib import Path

from backend.app.preprocessing_pipeline import preprocess_local_image


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Preprocess a local document image.")
    parser.add_argument("input_path", type=Path, help="Path to the source JPG/PNG image.")
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path where the processed JPG/PNG image should be saved.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1600,
        help="Maximum output width before grayscale conversion.",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=1600,
        help="Maximum output height before grayscale conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved_path = preprocess_local_image(
        args.input_path,
        args.output_path,
        max_width=args.max_width,
        max_height=args.max_height,
    )
    print(f"Saved processed image to {saved_path}")


if __name__ == "__main__":
    main()
