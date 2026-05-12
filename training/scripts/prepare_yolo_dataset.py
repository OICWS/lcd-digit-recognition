"""
Prepare a YOLOv8 training dataset.

Reads source photos and YOLO-format annotations, splits them 85/15 into
train/val, and writes a ready-to-train dataset with data.yaml.

Usage:
    python prepare_yolo_dataset.py \
        --photos <photos_dir> \
        --annotations <annotations_dir> \
        --output <output_dir> \
        [--yaml-path <runtime_dataset_path>]

--yaml-path is the path that data.yaml will reference at training time
(useful when training in a different environment, e.g. Kaggle). Defaults to
the absolute path of --output.
"""
import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare a YOLOv8 training dataset.")
    parser.add_argument("--photos",      required=True, help="Directory of source photos")
    parser.add_argument("--annotations", required=True, help="Directory of YOLO .txt annotations")
    parser.add_argument("--output",      required=True, help="Output dataset directory")
    parser.add_argument("--yaml-path",   default=None,
                        help="Path placed in data.yaml (defaults to absolute --output)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=float, default=0.85,
                        help="Fraction of samples used for training (default 0.85)")
    args = parser.parse_args()

    photos_dir      = Path(args.photos)
    annotations_dir = Path(args.annotations)
    output_dir      = Path(args.output)
    yaml_path       = args.yaml_path or str(output_dir.resolve())

    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"Cleared old directory: {output_dir}")

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    pairs = []
    for jpg in sorted(photos_dir.iterdir()):
        if jpg.suffix.lower() != ".jpg":
            continue
        txt = annotations_dir / (jpg.stem + ".txt")
        if txt.exists():
            pairs.append((jpg, txt))

    print(f"Found annotated images: {len(pairs)}")

    random.seed(args.seed)
    random.shuffle(pairs)
    split_idx   = int(len(pairs) * args.split)
    train_pairs = pairs[:split_idx]
    val_pairs   = pairs[split_idx:]
    print(f"Train: {len(train_pairs)}  Val: {len(val_pairs)}")

    def copy_pairs(pairs, split):
        for jpg, txt in pairs:
            shutil.copy2(str(jpg), str(output_dir / "images" / split / jpg.name))
            shutil.copy2(str(txt), str(output_dir / "labels" / split / txt.name))

    copy_pairs(train_pairs, "train")
    copy_pairs(val_pairs,   "val")

    yaml = (
        f"path: {yaml_path}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: 1\n"
        f"names: ['lcd']\n"
    )
    (output_dir / "data.yaml").write_text(yaml)

    print(f"\nDone. Dataset saved to: {output_dir}")
    print(f"data.yaml path field: {yaml_path}")


if __name__ == "__main__":
    main()
