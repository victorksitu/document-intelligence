# Milestone 1: Local Image Preprocessing

Milestone 1 accepts a local JPG/PNG document image and produces a processed image file that can be used by a future OCR step.

This milestone does not perform OCR or extract text. It only prepares image files.

## Current Flow

```text
local JPG/PNG path
    -> validate and load with OpenCV
    -> resize only if the image exceeds max dimensions
    -> convert to grayscale
    -> optionally apply median denoising
    -> optionally apply binary thresholding
    -> save the processed image locally
```

## Files

- `backend/app/image_loader.py`: validates and loads local JPG/PNG images as OpenCV/NumPy arrays.
- `backend/app/preprocessing.py`: resizes, converts to grayscale, optionally denoises, optionally thresholds, and saves processed images.
- `backend/app/preprocessing_pipeline.py`: connects loading, preprocessing, and saving into one local workflow.
- `backend/scripts/preprocess_image.py`: provides a small command-line runner for manual testing.
- `sample_data/synthetic/synthetic_receipt.png`: safe synthetic image for local preprocessing experiments.

## Commands

Run the grayscale-only sample from the repository root:

```bash
python -m backend.scripts.preprocess_image sample_data/synthetic/synthetic_receipt.png processed/synthetic_receipt_grayscale.png --max-width 600 --max-height 600
```

Run the same sample with thresholding enabled:

```bash
python -m backend.scripts.preprocess_image sample_data/synthetic/synthetic_receipt.png processed/synthetic_receipt_thresholded.png --max-width 600 --max-height 600 --threshold --threshold-value 127
```

Run the sample with denoising and thresholding enabled:

```bash
python -m backend.scripts.preprocess_image sample_data/synthetic/synthetic_receipt.png processed/synthetic_receipt_denoised_thresholded.png --max-width 600 --max-height 600 --denoise --denoise-kernel-size 3 --threshold --threshold-value 127
```

The output file is written to `processed/`, which is intentionally ignored by Git.

Run the tests:

```bash
python -m pytest backend/tests
```

## Current Choices

- The original input image is preserved.
- Resizing only downsizes large images; it does not upscale small images.
- Grayscale conversion removes color channels and keeps brightness information.
- Median denoising is optional and uses a small odd kernel size to reduce isolated speckles.
- Binary thresholding is optional so grayscale and black-and-white outputs can be compared.
- The save helper accepts `.jpg`, `.jpeg`, and `.png` outputs.

## Tradeoffs

- Thresholding can make dark text stand out, but it can also remove faint or anti-aliased text pixels.
- Higher threshold values preserve more gray text pixels, but may also preserve more background noise.
- Median denoising can remove isolated speckles, but larger kernel sizes can blur small text.
- Grayscale remains the safest default because it keeps more image detail than thresholding.

## Intentionally Skipped

- deskewing or rotation correction, intentionally skipped for this milestone
- OCR
- field extraction
- API/database/frontend/ML/Docker
