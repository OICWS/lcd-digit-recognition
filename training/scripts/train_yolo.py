"""
Train YOLOv8n on a prepared LCD detection dataset.

Usage:
    python train_yolo.py --dataset <path_to_yolo_dataset> --output <output_dir>

The dataset directory must contain a data.yaml (produced by
prepare_yolo_dataset.py). This script overwrites data.yaml's `path` field
to match --dataset before training, which is useful when the dataset is
moved between machines.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n for LCD detection.")
    parser.add_argument("--dataset", required=True, help="Path to prepared YOLO dataset directory")
    parser.add_argument("--output",  required=True, help="Output directory for training runs")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--imgsz",    type=int, default=640)
    parser.add_argument("--batch",    type=int, default=16)
    parser.add_argument("--device",   default=0, help="CUDA device id or 'cpu'")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--name",     default="lcd_detector")
    args = parser.parse_args()

    dataset    = Path(args.dataset)
    output_dir = Path(args.output)
    yaml_path  = dataset / "data.yaml"

    yaml_content = (
        f"path: {dataset.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: 1\n"
        f"names: ['lcd']\n"
    )
    yaml_path.write_text(yaml_content)
    print(f"Updated data.yaml at {yaml_path}")

    model = YOLO("yolov8n.pt")

    model.train(
        data     = str(yaml_path),
        epochs   = args.epochs,
        imgsz    = args.imgsz,
        batch    = args.batch,
        device   = args.device,
        project  = str(output_dir),
        name     = args.name,
        patience = args.patience,
        save     = True,
        exist_ok = True,
    )

    print("\nTraining complete.")
    print(f"Best weights: {output_dir}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
