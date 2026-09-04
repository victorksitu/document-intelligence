import pytest

from backend.app.ocr import normalize_tesseract_data
from backend.app.ocr_models import OCRBlock


def test_normalize_tesseract_data_converts_rows_to_ocr_blocks() -> None:
    data = {
        "text": ["TOTAL", "22.55"],
        "left": [60, 150],
        "top": [340, 340],
        "width": [80, 70],
        "height": [25, 25],
        "conf": ["96", "91"],
    }

    blocks = normalize_tesseract_data(data)

    assert blocks == [
        OCRBlock(text="TOTAL", x=60, y=340, width=80, height=25, confidence=0.96),
        OCRBlock(text="22.55", x=150, y=340, width=70, height=25, confidence=0.91),
    ]


def test_normalize_tesseract_data_strips_text() -> None:
    data = {
        "text": ["  TOTAL  "],
        "left": [60],
        "top": [340],
        "width": [80],
        "height": [25],
        "conf": ["96"],
    }

    blocks = normalize_tesseract_data(data)

    assert blocks[0].text == "TOTAL"


def test_normalize_tesseract_data_skips_empty_text_rows() -> None:
    data = {
        "text": ["", "   ", "TOTAL"],
        "left": [0, 0, 60],
        "top": [0, 0, 340],
        "width": [0, 0, 80],
        "height": [0, 0, 25],
        "conf": ["-1", "-1", "96"],
    }

    blocks = normalize_tesseract_data(data)

    assert blocks == [
        OCRBlock(text="TOTAL", x=60, y=340, width=80, height=25, confidence=0.96)
    ]


def test_normalize_tesseract_data_skips_invalid_boxes() -> None:
    data = {
        "text": ["BAD", "TOTAL", "ALSO_BAD"],
        "left": [-1, 60, 90],
        "top": [0, 340, 100],
        "width": [10, 80, 0],
        "height": [10, 25, 10],
        "conf": ["90", "96", "90"],
    }

    blocks = normalize_tesseract_data(data)

    assert blocks == [
        OCRBlock(text="TOTAL", x=60, y=340, width=80, height=25, confidence=0.96)
    ]


def test_normalize_tesseract_data_converts_negative_confidence_to_none() -> None:
    data = {
        "text": ["TOTAL"],
        "left": [60],
        "top": [340],
        "width": [80],
        "height": [25],
        "conf": ["-1"],
    }

    blocks = normalize_tesseract_data(data)

    assert blocks == [
        OCRBlock(text="TOTAL", x=60, y=340, width=80, height=25, confidence=None)
    ]


def test_normalize_tesseract_data_converts_decimal_confidence() -> None:
    data = {
        "text": ["TOTAL"],
        "left": [60],
        "top": [340],
        "width": [80],
        "height": [25],
        "conf": ["95.5"],
    }

    blocks = normalize_tesseract_data(data)

    assert blocks[0].confidence == 0.955


def test_normalize_tesseract_data_rejects_missing_required_keys() -> None:
    data = {
        "text": ["TOTAL"],
        "left": [60],
        "top": [340],
        "width": [80],
        "height": [25],
    }

    with pytest.raises(ValueError, match="Missing Tesseract OCR keys"):
        normalize_tesseract_data(data)


def test_normalize_tesseract_data_rejects_mismatched_column_lengths() -> None:
    data = {
        "text": ["TOTAL", "22.55"],
        "left": [60],
        "top": [340, 340],
        "width": [80, 70],
        "height": [25, 25],
        "conf": ["96", "91"],
    }

    with pytest.raises(ValueError, match="same length"):
        normalize_tesseract_data(data)
