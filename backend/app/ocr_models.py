from dataclasses import dataclass


@dataclass(frozen=True)
class OCRBlock:
    """Normalized OCR text with its bounding box and confidence."""

    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text must not be empty")

        if self.x < 0 or self.y < 0:
            raise ValueError("x and y must be greater than or equal to 0")

        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be greater than 0")

        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
