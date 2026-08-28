# Milestone 1: Preprocessing Notes

The current local preprocessing flow is:

```text
local JPG/PNG path
    -> validate and load with OpenCV
    -> resize only if the image exceeds max dimensions
    -> convert to grayscale
    -> save the processed image locally
```

Run the sample preprocessing command from the repository root:

```bash
python -m backend.scripts.preprocess_image sample_data/synthetic/synthetic_receipt.png processed/synthetic_receipt_grayscale.png --max-width 600 --max-height 600
```

The output file is written to `processed/`, which is intentionally ignored by Git.

## Current Choices

- The original input image is preserved.
- Resizing only downsizes large images; it does not upscale small images.
- Grayscale conversion removes color channels and keeps brightness information.
- The save helper accepts `.jpg`, `.jpeg`, and `.png` outputs.

## Not Implemented Yet

- thresholding
- denoising
- deskewing or rotation correction
- OCR
- field extraction
- API/database/frontend/ML/Docker
