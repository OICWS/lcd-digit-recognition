# LCD Number Recognition

Seven-segment LCD display OCR using **YOLOv8 (detection) + CRNN (recognition)**. Trained on ~800 labeled photos, reaching **100% validation accuracy** and **~96% real-world accuracy** on unseen production photos.

> **Scope.** This is a methodology reference, not a drop-in production tool. Source photos are internal industrial data and cannot be released for privacy reasons. Code, training pipeline, and pretrained weights are open — adapting this to your own LCD device requires collecting and labeling your own photos following the workflow below.

<p align="center">
  <img src="docs/crnn_pipeline.png" alt="End-to-end pipeline: YOLOv8 detection → CRNN with CTC decoding" width="100%">
  <br>
  <em>End-to-end pipeline. <a href="docs/crnn_pipeline.pdf">High-resolution PDF</a></em>
</p>

## Why this exists

Seven-segment LCD readouts (industrial scales, multimeters, instrument panels) are a common OCR target but a poor fit for general-purpose OCR engines: the glyphs are bar-segment patterns with no font priors, spacing is irregular across devices, and real-world lighting (reflections, glare, oblique angles) varies wildly. Generic OCR models trained on natural-scene text or documents fail on this distribution. A small, domain-specific model trained from a few hundred labeled photos outperforms every off-the-shelf engine tested.

## Performance

| Stage | Model   | Training data      | Result                      |
|------:|---------|---------------------|------------------------------|
|   1   | YOLOv8n | ~800 labeled photos | mAP50 = 0.995                |
|   2   | CRNN    | ~800 cropped LCDs   | Val 100%, real-world ~96%    |

The training set was assembled incrementally through production use and error-driven retraining, covering front-facing and oblique angles, indoor/outdoor lighting, varying distances, and multiple camera types.

**Known limitation:** the CRNN uses CTC decoding, which merges adjacent identical characters unless the model explicitly emits a separator between them. On values with repeated adjacent digits (e.g. `1006.5`), this occasionally causes a digit to be silently dropped (`1006.5` → `106.5`) — and this error is *not* reliably flagged by decode confidence, since the model can be highly confident on every individual character while still misaligning the sequence. This is a structural property of greedy CTC decoding, not a data-quantity problem. Beam search decoding and geometric consistency checks (expected digit count from bounding-box width) are the two most promising mitigations, not yet implemented here.

## Engineering highlights

**Two-stage decomposition over end-to-end.** Detection and recognition have very different data requirements — detection benefits from ImageNet/COCO-pretrained backbones, recognition needs only clean normalized crops. Splitting the task lets each stage train efficiently on a small dataset; an end-to-end model would need substantially more data to converge.

**CTC over fixed-pitch segmentation.** Character spacing varies across LCD models, and any fixed-width segmentation scheme breaks when a new device is introduced. CTC handles variable-length output without explicit character segmentation.

**`smart_resize` for resolution drift.** Photos come from different cameras with wildly different crop resolutions (LCD crops from ~200px to 1000+ px wide). Resizing everything directly to the fixed 64×256 model input compresses high-resolution crops into a different pixel-level pattern than the training distribution. `smart_resize` pre-downscales any crop wider than 400px to ≤300px before the standard resize, keeping the distribution consistent:

```python
def smart_resize(img):
    w, h = img.size
    if w > 400:
        scale = 300 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img
```

**Iterative error-driven retraining.** New photos are periodically run through the pipeline, misreads are visualized and manually reviewed, corrected labels are appended to the training set, and the CRNN is retrained (~15 min on a free Kaggle GPU). This is how accuracy improved over time without a large upfront dataset.

**Minimal inference requirements.** The pipeline runs on CPU, no GPU or network connection required. Total model footprint is ~22 MB (YOLOv8n + CRNN combined).

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

**Why not ResNet/ViT for recognition?** Those are classification models — they map an image to a single label. Recognizing a variable-length digit sequence needs either a fixed output length (breaks when digit count varies) or explicit character segmentation (the problem CTC avoids). CRNN + CTC handles variable-length sequences without segmentation.

**Why not an attention-based decoder (e.g. TrOCR)?** Attention decoders need substantially more training data to learn a reliable input-output alignment than CTC's monotonic-alignment constraint requires — not a good fit for a dataset this size.

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
        ├── label_lcd_values.py      # 1. Interactively label digit values
        ├── crop_lcd_regions.py      # 2. Crop LCD regions via bbox annotations
        ├── prepare_yolo_dataset.py  # 3a. Build YOLO train/val split
        ├── train_yolo.py            # 3b. Train YOLOv8n
        ├── train_crnn.py            # 4. Train CRNN
        └── extract_error_samples.py # Helper: recover originals for re-labeling
```

> Trained weights, source photos, and labels are not included in this repository — see **Download weights** below.

## Quick start

```bash
pip install -r requirements.txt
```

Place trained weights in `models/`: `best.pt`, `final_crnn.pth`, `crnn_config.json`.

```bash
# Single image
python main.py path/to/photo.jpg

# Whole directory (recursive)
python main.py path/to/folder
```

Or use the pipeline programmatically:
```python
from ocr_reader import read_lcd_number, read_lcd_batch

value, conf = read_lcd_number("path/to/photo.jpg")   # (value, confidence); value is None on failure

results = read_lcd_batch(["a.jpg", "b.jpg", "c.jpg"])  # batched inference
```

Visualize predictions over a folder:
```bash
python visualize_yolo_results.py --input path/to/photos --output path/to/output
```

## Training your own model

**1. Label bounding boxes** with [labelImg](https://github.com/HumanSignal/labelImg) — single class `lcd`, YOLO format.

**2. Label digit values**:
```bash
python training/scripts/label_lcd_values.py --photos /path/to/photos --labels /path/to/labels.csv
```

**3. Crop LCD regions** for CRNN training:
```bash
python training/scripts/crop_lcd_regions.py \
    --photos /path/to/photos --annotations /path/to/annotations \
    --dataset /path/to/crops --labels /path/to/labels.csv
```

**4. Train YOLOv8**:
```bash
python training/scripts/prepare_yolo_dataset.py \
    --photos /path/to/photos --annotations /path/to/annotations --output /path/to/yolo_dataset
python training/scripts/train_yolo.py --dataset /path/to/yolo_dataset --output /path/to/yolo_runs
```

**5. Train CRNN**:
```bash
python training/scripts/train_crnn.py --data /path/to/crops --output /path/to/crnn_out
```

Copy the resulting `final_crnn.pth` and `crnn_config.json` into `models/`.

## Download weights

Pretrained weights are published as **GitHub Releases** — see the **Releases** tab. Download `best.pt`, `final_crnn.pth`, and `crnn_config.json` into `models/`.

## License

Licensed under **AGPL-3.0** (depends on [ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8, itself AGPL-3.0). Trained model weights distributed via Releases are under **CC BY 4.0** — free to use, including commercially, with attribution.
