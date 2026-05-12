"""
Crop LCD regions from photos using their YOLO-format bbox annotations,
producing a CRNN training dataset.

Usage:
    python crop_lcd_regions.py \
        --photos <photos_dir> \
        --annotations <annotations_dir> \
        --dataset <output_crops_dir> \
        --labels <labels.csv>

Only photos that have BOTH a matching annotation .txt and an entry in
labels.csv are cropped.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def read_image(path):
    with open(path, "rb") as f:
        arr = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def save_image(path, img):
    ret, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Crop LCD regions using YOLO bbox annotations.")
    parser.add_argument("--photos",      required=True, help="Directory of source photos")
    parser.add_argument("--annotations", required=True, help="Directory of YOLO .txt annotations")
    parser.add_argument("--dataset",     required=True, help="Output directory for crops")
    parser.add_argument("--labels",      required=True, help="labels.csv with digit values")
    args = parser.parse_args()

    photos_dir      = Path(args.photos)
    annotations_dir = Path(args.annotations)
    dataset_dir     = Path(args.dataset)
    labels_file     = Path(args.labels)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with open(labels_file, "r", encoding="utf-8") as f:
        labels = {row["filename"]: row["lcd_value"] for row in csv.DictReader(f)}

    photo_names = {f.name for f in photos_dir.iterdir()
                   if f.suffix.lower() == ".jpg"}

    count_ok   = 0
    count_skip = 0

    for fname in sorted(photo_names):
        txt_path = annotations_dir / (Path(fname).stem + ".txt")
        if not txt_path.exists():
            print(f"  [skip] no annotation: {fname}")
            count_skip += 1
            continue

        if fname not in labels:
            print(f"  [skip] no label: {fname}")
            count_skip += 1
            continue

        out_path = dataset_dir / fname
        if out_path.exists():
            print(f"  [exists] {fname}")
            count_ok += 1
            continue

        img = read_image(str(photos_dir / fname))
        if img is None:
            print(f"  [error] failed to read: {fname}")
            count_skip += 1
            continue

        lines = txt_path.read_text().strip().splitlines()
        if not lines:
            print(f"  [skip] empty annotation: {fname}")
            count_skip += 1
            continue

        parts = lines[0].strip().split()
        if len(parts) < 5:
            print(f"  [skip] invalid format: {fname}")
            count_skip += 1
            continue

        _, cx, cy, bw, bh = map(float, parts[:5])
        h, w = img.shape[:2]

        x1 = max(0, int((cx - bw / 2) * w) - 5)
        y1 = max(0, int((cy - bh / 2) * h) - 5)
        x2 = min(w, int((cx + bw / 2) * w) + 5)
        y2 = min(h, int((cy + bh / 2) * h) + 5)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"  [skip] empty crop: {fname}")
            count_skip += 1
            continue

        save_image(str(out_path), crop)
        print(f"  {fname} -> {crop.shape[1]}x{crop.shape[0]}px")
        count_ok += 1

    print(f"\nDone. ok:{count_ok}  skipped:{count_skip}")
    print(f"Dataset now contains: {len(list(dataset_dir.glob('*.jpg')))} crops")


if __name__ == "__main__":
    main()
