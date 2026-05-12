"""
Given a directory of misclassified visualization images, copy the matching
original photos out of a source photo directory into an output folder, so
they can be re-labeled and added to the training set.

Usage:
    python extract_error_samples.py \
        --vis <visualization_dir> \
        --source <original_photos_dir> \
        --output <output_dir>
"""
import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Extract original photos for misclassified visualization images."
    )
    parser.add_argument("--vis",    required=True, help="Directory of misclassified visualization images")
    parser.add_argument("--source", required=True, help="Directory of original photos (searched recursively)")
    parser.add_argument("--output", required=True, help="Output directory for extracted originals")
    args = parser.parse_args()

    vis_dir    = Path(args.vis)
    source_dir = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    vis_names = set()
    for jpg in vis_dir.rglob("*"):
        if jpg.suffix.lower() == ".jpg":
            vis_names.add(jpg.name)
    print(f"Found {len(vis_names)} filenames in vis directory")

    found = 0
    missing = []

    for name in sorted(vis_names):
        matches = list(source_dir.rglob(name))
        if not matches:
            matches = [p for p in source_dir.rglob("*")
                       if p.name.lower() == name.lower()]

        if matches:
            src = matches[0]
            dst = output_dir / name
            shutil.copy2(str(src), str(dst))
            print(f"  + {name}")
            found += 1
        else:
            missing.append(name)
            print(f"  - not found: {name}")

    print(f"\nDone. Found:{found}  Missing:{len(missing)}")
    if missing:
        print("Missing files:")
        for m in missing:
            print(f"  {m}")
    print(f"\nOriginals extracted to: {output_dir}")


if __name__ == "__main__":
    main()
