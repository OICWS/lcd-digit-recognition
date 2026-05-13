"""
LCD digit recognition inference entry point.

Usage:
    python main.py path/to/image.jpg
    python main.py path/to/folder           # recursively reads all .jpg files
    python main.py img1.jpg img2.jpg ...    # multiple files

Outputs the recognized numeric value for each image, or "FAILED" if detection
or recognition was unsuccessful.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path

from ocr_reader import read_lcd_number


def collect_images(inputs):
    images = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            images.extend(sorted(p.rglob("*.jpg")))
            images.extend(sorted(p.rglob("*.jpeg")))
            images.extend(sorted(p.rglob("*.png")))
        elif p.is_file():
            images.append(p)
        else:
            print(f"[WARN] Path not found: {p}")
    return images


def main():
    parser = argparse.ArgumentParser(
        description="LCD digit recognition using YOLOv8 + CRNN."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Image file(s) or directory to run inference on.",
    )
    args = parser.parse_args()

    images = collect_images(args.inputs)
    if not images:
        print("No images found.")
        return

    success = 0
    failed = 0
    for img in images:
        print(f"\n>>> {img}")
        value, conf = read_lcd_number(str(img))
        if value is None:
            print(f"    Result: FAILED (conf={conf:.3f})")
            failed += 1
        else:
            print(f"    Result: {value} (conf={conf:.3f})")
            success += 1

    print("\n" + "=" * 60)
    print(f"Total: {len(images)}   Success: {success}   Failed: {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
