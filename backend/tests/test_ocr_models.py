import pytest

from backend.app.ocr_models import OCRBlock


def test_ocr_block_stores_text_box_and_confidence() -> None:
    block = OCRBlock(
        text="TOTAL",
        x=60,
        y=340,
        width=80,
        height=25,
        confidence=0.97,
    )

    assert block.text == "TOTAL"
    assert block.x == 60
    assert block.y == 340
    assert block.width == 80
    assert block.height == 25
    assert block.confidence == 0.97


def test_ocr_block_allows_missing_confidence() -> None:
    block = OCRBlock(
        text="TOTAL",
        x=60,
        y=340,
        width=80,
        height=25,
    )

    assert block.confidence is None


def test_ocr_block_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="text"):
        OCRBlock(text=" ", x=0, y=0, width=10, height=10, confidence=0.5)


def test_ocr_block_rejects_negative_coordinates() -> None:
    with pytest.raises(ValueError, match="x and y"):
        OCRBlock(text="TOTAL", x=-1, y=0, width=10, height=10, confidence=0.5)


def test_ocr_block_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="width and height"):
        OCRBlock(text="TOTAL", x=0, y=0, width=0, height=10, confidence=0.5)


def test_ocr_block_rejects_confidence_outside_normalized_range() -> None:
    with pytest.raises(ValueError, match="confidence"):
        OCRBlock(text="TOTAL", x=0, y=0, width=10, height=10, confidence=97.0)
