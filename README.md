# LCD Number Recognition

A production-grade OCR pipeline for **seven-segment LCD displays**, built with **YOLOv8 + CRNN**. Trained from scratch on ~300 labeled images, the system reaches **100% validation accuracy** and **~96% real-world accuracy** on photos taken under varying lighting, angle, and distance.

This repository is a complete, reproducible reference for a domain-specific OCR problem where generic OCR tools (EasyOCR, PaddleOCR, Tesseract) fail.

<p align="center">
  <img src="docs/crnn_pipeline.png" alt="End-to-end pipeline: YOLOv8 detection → CRNN with CTC decoding" width="100%">
  <br>
  <em>End-to-end pipeline. <a href="docs/crnn_pipeline.pdf">High-resolution PDF</a></em>
</p>

## Why this exists

Seven-segment LCD readouts (industrial scales, multimeters, instrument panels, etc.) are a common OCR target but a notoriously poor fit for general OCR engines:

- **Glyphs are not real fonts.** Seven-segment digits are spatial patterns of bright bars on dark backgrounds, with no anti-aliasing, no kerning, and no font priors that pretrained OCR models can leverage.
- **Spacing is irregular.** Industrial panels often align digits to fixed cells with wide gaps between them, breaking fixed-pitch character splitters.
- **Lighting is hostile.** Reflective glass, glare, oblique angles, and camera auto-exposure produce wildly different appearances of the same digit.

Generic OCR models trained on natural-scene text or scanned documents fall apart on this distribution. Training a small, domain-specific model from ~300 labeled photos beats every off-the-shelf engine I tested.

## Performance

| Stage | Model   | Training data         | Result                       |
|------:|---------|-----------------------|------------------------------|
|   1   | YOLOv8n | 314 labeled photos    | mAP50 = 0.995, Recall = 1.0  |
|   2   | CRNN    | 314 cropped LCDs      | Val 100%, Real-world ~96%    |

## Engineering highlights

### Two-stage decomposition over end-to-end
The full task (photo → digits) is decomposed into **detect-then-read** because the two sub-tasks have very different data requirements. Detection needs spatial context and benefits enormously from ImageNet/COCO pretraining (so a fine-tuned YOLOv8n hits 0.995 mAP with 314 images). Recognition needs only clean, normalized crops and is data-cheap to train from scratch. End-to-end models would force both sub-tasks to share a backbone and would need 10× more data to converge.

### `smart_resize`: handling resolution drift
A subtle but high-impact issue: modern phone cameras produce LCD crops 1000+ pixels wide, while older training data has crops around 200–400 pixels. After the standard 64 × 256 resize, high-res crops compress digits into a different frequency distribution than the training set, causing silent accuracy degradation on new photos. `smart_resize` pre-downscales any crop wider than 400 px to ≤300 px **before** the standard transform, restoring distributional consistency. This single change moved real-world accuracy from ~88% to ~96%.

### CTC decoding over fixed-pitch segmentation
Character spacing varies across LCD models. Any fixed-width or fixed-count segmentation scheme breaks the moment a new device is introduced. CTC handles variable-length output without explicit segmentation and degrades gracefully on partially occluded or blurred digits.

### CRNN trained from scratch (no pretraining)
The CRNN backbone is small (~3M parameters) and trains from random initialization on 314 crops in ~15 minutes on a free Kaggle T4. Pretraining on synthetic seven-segment data was tried; it offered no improvement, because the real-world variation is in lighting and angle, not glyph shape — and those variations are already covered by the augmentation pipeline (brightness/contrast jitter, random rescale).

### Iterative error-driven retraining loop
The repository includes a workflow for closed-loop improvement:

```
visualize predictions on new photos  (visualize_yolo_results.py)
    ↓
identify misclassified samples
    ↓
extract originals by filename       (extract_error_samples.py)
    ↓
label, crop, append to training set
    ↓
retrain CRNN (~15 min)
```

This is how the system went from ~88% to ~96% real-world accuracy without growing the dataset beyond ~300 images: each retraining round targets the model's actual failure modes rather than adding random samples.

## Repository layout

```
.
├── main.py                      # CLI inference entry point
├── ocr_reader.py                # YOLO + CRNN inference pipeline
├── visualize_yolo_results.py    # Batch annotated-image visualization
├── requirements.txt
├── models/                      # (download weights into this directory)
│   ├── best.pt                  # YOLOv8 weights
│   ├── final_crnn.pth           # CRNN weights
│   └── crnn_config.json         # CRNN charset config
└── training/
    └── scripts/
        ├── label_lcd_values.py     # 1. Interactively label digit values
        ├── crop_lcd_regions.py     # 2. Crop LCD regions via bbox annotations
        ├── prepare_yolo_dataset.py # 3a. Build YOLO train/val split
        ├── train_yolo.py           # 3b. Train YOLOv8n
        ├── train_crnn.py           # 4. Train CRNN
        └── extract_error_samples.py# Helper: recover originals for re-labeling
```

> Trained weights, source photos, and labels are not included in this repository. See **Download weights** below.

## CRNN architecture

```
Input: (B, 3, 64, 256)
  ↓ Conv2d(3→32)   + BN + ReLU + MaxPool(2,2)  → (B,  32, 32, 128)
  ↓ Conv2d(32→64)  + BN + ReLU + MaxPool(2,2)  → (B,  64, 16,  64)
  ↓ Conv2d(64→128) + BN + ReLU + MaxPool(2,1)  → (B, 128,  8,  64)
  ↓ Conv2d(128→256)+ BN + ReLU + MaxPool(2,1)  → (B, 256,  4,  64)
  ↓ Conv2d(256→256)+ BN + ReLU + AvgPool(1,*)  → (B, 256,  1,  64)
  ↓ squeeze + permute                          → (B,  64, 256)
  ↓ BiLSTM(256→256, 2 layers)                  → (B,  64, 512)
  ↓ Linear(512→12)                             → (B,  64,  12)
  ↓ permute                                    → (64,  B,  12)
  ↓ CTC greedy decode
Output: digit string (charset 0123456789. + CTC blank = 12 classes)
```

## Quick start

```bash
pip install -r requirements.txt
```

Place trained weights in `models/`:
```
models/best.pt
models/final_crnn.pth
models/crnn_config.json
```

Run inference:
```bash
# Single image
python main.py path/to/photo.jpg

# Whole directory (recursive)
python main.py path/to/folder

# Multiple files
python main.py img1.jpg img2.jpg img3.jpg
```

Or use the pipeline programmatically:
```python
from ocr_reader import read_lcd_number

value = read_lcd_number("path/to/photo.jpg")
print(value)  # e.g. 1027.5
```

Visualize predictions over a folder:
```bash
python visualize_yolo_results.py --input path/to/photos --output path/to/output
```

## Training your own model

All training scripts accept paths via command-line flags.

**1. Label LCD bounding boxes** with [labelImg](https://github.com/HumanSignal/labelImg) — single class `lcd`, YOLO format.

**2. Label digit values** for each photo:
```bash
python training/scripts/label_lcd_values.py \
    --photos /path/to/photos \
    --labels /path/to/labels.csv
```

**3. Crop LCD regions** for CRNN training:
```bash
python training/scripts/crop_lcd_regions.py \
    --photos      /path/to/photos \
    --annotations /path/to/annotations \
    --dataset     /path/to/crops \
    --labels      /path/to/labels.csv
```

**4. Train YOLOv8**:
```bash
python training/scripts/prepare_yolo_dataset.py \
    --photos      /path/to/photos \
    --annotations /path/to/annotations \
    --output      /path/to/yolo_dataset

python training/scripts/train_yolo.py \
    --dataset /path/to/yolo_dataset \
    --output  /path/to/yolo_runs
```

**5. Train CRNN**:
```bash
python training/scripts/train_crnn.py \
    --data   /path/to/crops \
    --output /path/to/crnn_out
```

The CRNN output directory will contain `final_crnn.pth` and `crnn_config.json` ready to copy into `models/`.

## Download weights

Pretrained weights are published as **GitHub Releases** in this repository — see the **Releases** tab. Download `best.pt`, `final_crnn.pth`, and `crnn_config.json` and place them in `models/`.

## License

This project is licensed under **AGPL-3.0**. It depends on [ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8), which is itself AGPL-3.0 licensed; any derivative work must comply with the AGPL-3.0 terms.

Trained model weights distributed via Releases are released under **CC BY 4.0** — free to use, including commercially, with attribution.
