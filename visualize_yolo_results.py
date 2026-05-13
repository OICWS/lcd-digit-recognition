"""
Recursively scan a given directory for .jpg files, detect LCDs with YOLO,
recognize digits with CRNN, and produce annotated output images.

Usage:
    python visualize_yolo_results.py --input <input_dir> --output <output_dir>
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import cv2
import numpy as np
from pathlib import Path

from ocr_reader import read_lcd_number, get_yolo


def read_image(path):
    with open(path, 'rb') as f:
        arr = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def save_image(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    with open(path, 'wb') as f:
        f.write(buf.tobytes())


def draw_result(img, box, lcd_val, conf):
    h, w = img.shape[:2]

    if w > 1200:
        scale = 1200 / w
        img = cv2.resize(img, (1200, int(h * scale)))
        h, w = img.shape[:2]
        if box:
            x1, y1, x2, y2 = box
            x1 = int(x1 * scale)
            y1 = int(y1 * scale)
            x2 = int(x2 * scale)
            y2 = int(y2 * scale)
            box = (x1, y1, x2, y2)

    result = img.copy()

    if box:
        x1, y1, x2, y2 = box
        color = (0, 200, 80) if lcd_val is not None else (0, 100, 255)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)

        if lcd_val is not None:
            label = f"{lcd_val}  conf:{conf:.2f}"
        else:
            label = f"detected but unreadable  conf:{conf:.2f}"

        font_scale = 1.0
        thickness  = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        lx = x1
        ly = max(y1 - 10, th + 10)
        cv2.rectangle(result, (lx, ly - th - 6), (lx + tw + 8, ly + 4), color, -1)
        cv2.putText(result, label, (lx + 4, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    else:
        cv2.rectangle(result, (0, 0), (260, 36), (60, 60, 60), -1)
        cv2.putText(result, "No LCD detected, skipped",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Visualize YOLO+CRNN inference results on a directory of images."
    )
    parser.add_argument("--input",  required=True, help="Input directory containing .jpg images")
    parser.add_argument("--output", required=True, help="Output directory for annotated images")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    yolo = get_yolo()

    all_jpgs = sorted([
        p for p in input_dir.rglob("*")
        if p.suffix.lower() == '.jpg'
    ])
    print(f"Found {len(all_jpgs)} .jpg files")

    total   = 0
    success = 0
    skipped = 0

    for jpg_path in all_jpgs:
        img = read_image(str(jpg_path))
        if img is None:
            continue

        total += 1
        h, w = img.shape[:2]

        results  = yolo(str(jpg_path), verbose=False, conf=0.3)
        has_lcd  = results and len(results[0].boxes) > 0

        if has_lcd:
            boxes    = results[0].boxes
            best_idx = boxes.conf.argmax().item()
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
            conf = boxes.conf[best_idx].item()
            pad  = 5
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            box = (x1, y1, x2, y2)

            lcd_val, _ = read_lcd_number(str(jpg_path))
            if lcd_val is not None:
                success += 1
            vis = draw_result(img, box, lcd_val, conf)
            print(f"  + {jpg_path.name}: {lcd_val}")
        else:
            skipped += 1
            vis = draw_result(img, None, None, 0)
            print(f"  - {jpg_path.name}: no LCD, skipped")

        rel = jpg_path.relative_to(input_dir)
        out = output_dir / rel
        save_image(out, vis)

    print(f"\nDone. Total:{total}  Recognized:{success}  Skipped (no LCD):{skipped}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
