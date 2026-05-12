"""
Interactively label LCD digit values for a folder of photos, appending to a
labels.csv file.

Usage:
    python label_lcd_values.py --photos <photos_dir> --labels <labels.csv>

Controls:
    Enter a numeric value (e.g. 1027.5) and press Enter to record it.
    s  skip this image
    q  save and quit
    u  undo the last entry
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from PIL import Image


def show_image(path, preview_path):
    """Open a scaled-up preview of the image with the system default viewer."""
    img = Image.open(path)
    w, h = img.size
    scale = max(2, 300 // max(h, 1))
    img_big = img.resize((w * scale, h * scale), Image.NEAREST)
    img_big.save(str(preview_path))

    if sys.platform.startswith("win"):
        return subprocess.Popen(["mspaint", str(preview_path)])
    elif sys.platform == "darwin":
        return subprocess.Popen(["open", str(preview_path)])
    else:
        return subprocess.Popen(["xdg-open", str(preview_path)])


def main():
    parser = argparse.ArgumentParser(description="Interactively label LCD digit values.")
    parser.add_argument("--photos", required=True, help="Directory of photos to label")
    parser.add_argument("--labels", required=True, help="Path to labels.csv (created if missing)")
    parser.add_argument("--preview", default=None,
                        help="Path to write a temporary preview image (optional)")
    args = parser.parse_args()

    photos_dir  = Path(args.photos)
    labels_file = Path(args.labels)
    preview_path = Path(args.preview) if args.preview else \
        labels_file.parent / "tmp_preview.jpg"

    existing = set()
    if labels_file.exists():
        with open(labels_file, "r", encoding="utf-8") as f:
            existing = {row["filename"] for row in csv.DictReader(f)}
        print(f"Already labeled: {len(existing)} entries")

    todo = [f for f in sorted(photos_dir.iterdir())
            if f.suffix.lower() == ".jpg" and f.name not in existing]

    print(f"Pending: {len(todo)} images")
    print()
    print("Commands: numeric value + Enter | s=skip | q=save and quit | u=undo")
    print("-" * 40)

    new_labels = []

    def save_all():
        all_rows = []
        if labels_file.exists():
            with open(labels_file, "r", encoding="utf-8") as f:
                all_rows = list(csv.DictReader(f))
        new_fnames = {r["filename"] for r in new_labels}
        all_rows = [r for r in all_rows if r["filename"] not in new_fnames]
        all_rows.extend(new_labels)
        labels_file.parent.mkdir(parents=True, exist_ok=True)
        with open(labels_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "lcd_value"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  Saved {len(all_rows)} entries -> {labels_file}")

    i = 0
    current_proc = None

    while i < len(todo):
        img_path = todo[i]
        if current_proc:
            try:
                current_proc.terminate()
            except Exception:
                pass
        current_proc = show_image(img_path, preview_path)

        print(f"\n[{i+1}/{len(todo)}] {img_path.name}")
        val = input("  LCD value > ").strip()

        if val.lower() == "q":
            save_all()
            break
        elif val.lower() == "s":
            i += 1
        elif val.lower() == "u":
            if new_labels:
                removed = new_labels.pop()
                print(f"  Undo: {removed['filename']} = {removed['lcd_value']}")
                i -= 1
            else:
                print("  Nothing to undo")
        elif re.match(r"^\d+\.?\d*$", val):
            new_labels.append({"filename": img_path.name, "lcd_value": val})
            print(f"  {val}")
            i += 1
            if len(new_labels) % 10 == 0:
                save_all()
        else:
            print("  Invalid format, please enter a number like 1027.5")

    if current_proc:
        try:
            current_proc.terminate()
        except Exception:
            pass
    save_all()
    print("Done.")


if __name__ == "__main__":
    main()
