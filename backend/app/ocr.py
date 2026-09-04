from collections.abc import Mapping, Sequence

from backend.app.ocr_models import OCRBlock

REQUIRED_TESSERACT_KEYS = ("text", "left", "top", "width", "height", "conf")


def normalize_tesseract_data(data: Mapping[str, Sequence[object]]) -> list[OCRBlock]:
    """Convert Tesseract-style OCR rows into normalized OCR blocks."""
    _validate_tesseract_data(data)

    blocks: list[OCRBlock] = []
    row_count = len(data["text"])

    for index in range(row_count):
        text = str(data["text"][index]).strip()

        if not text:
            continue

        block = _normalize_tesseract_row(data, index, text)

        if block is not None:
            blocks.append(block)

    return blocks


def _validate_tesseract_data(data: Mapping[str, Sequence[object]]) -> None:
    missing_keys = [key for key in REQUIRED_TESSERACT_KEYS if key not in data]

    if missing_keys:
        raise ValueError(f"Missing Tesseract OCR keys: {', '.join(missing_keys)}")

    row_count = len(data["text"])

    for key in REQUIRED_TESSERACT_KEYS:
        if len(data[key]) != row_count:
            raise ValueError("Tesseract OCR columns must have the same length")


def _normalize_tesseract_row(
    data: Mapping[str, Sequence[object]],
    index: int,
    text: str,
) -> OCRBlock | None:
    try:
        x = _parse_int(data["left"][index])
        y = _parse_int(data["top"][index])
        width = _parse_int(data["width"][index])
        height = _parse_int(data["height"][index])
    except (TypeError, ValueError):
        return None

    try:
        confidence = _normalize_tesseract_confidence(data["conf"][index])
        return OCRBlock(
            text=text,
            x=x,
            y=y,
            width=width,
            height=height,
            confidence=confidence,
        )
    except ValueError:
        return None


def _parse_int(value: object) -> int:
    return int(str(value).strip())


def _normalize_tesseract_confidence(raw_confidence: object) -> float | None:
    confidence_text = str(raw_confidence).strip()

    if not confidence_text:
        return None

    confidence = float(confidence_text)

    if confidence < 0:
        return None

    if confidence > 100:
        raise ValueError("Tesseract confidence must be between -1 and 100")

    return confidence / 100
