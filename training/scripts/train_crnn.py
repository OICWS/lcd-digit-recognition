"""
Train the CRNN digit recognizer.

Usage:
    python train_crnn.py --data <data_dir> --output <output_dir>

The data directory must contain:
    - cropped LCD .jpg images
    - labels.csv with columns: filename, lcd_value
"""
import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


CHARS     = "0123456789."
BLANK     = len(CHARS)
NUM_CLASS = len(CHARS) + 1


def encode(text):
    return [CHARS.index(c) for c in text if c in CHARS]


def decode(indices):
    result, prev = [], None
    for i in indices:
        if i != BLANK and i != prev:
            result.append(CHARS[i])
        prev = i
    return ''.join(result)


def smart_resize(img):
    """Down-scale very wide crops before the unified resize, preserving detail."""
    w, h = img.size
    if w > 400:
        scale = 300 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


class LCDDataset(Dataset):
    def __init__(self, rows, image_cache, augment=False):
        self.rows        = rows
        self.image_cache = image_cache
        self.augment     = augment
        self.transform = T.Compose([
            T.Resize((64, 256)),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = self.image_cache[row['filename']].copy()
        img = smart_resize(img)

        if self.augment:
            if random.random() > 0.3:
                img = TF.adjust_brightness(img, random.uniform(0.6, 1.4))
            if random.random() > 0.3:
                img = TF.adjust_contrast(img, random.uniform(0.6, 1.4))
            if random.random() > 0.5:
                w, h = img.size
                scale = random.uniform(0.7, 1.3)
                img = img.resize(
                    (max(32, int(w * scale)), max(16, int(h * scale))),
                    Image.LANCZOS,
                )

        img   = self.transform(img)
        label = encode(row['lcd_value'])
        return img, torch.tensor(label, dtype=torch.long), len(label)


def collate_fn(batch):
    imgs, labels, lens = zip(*batch)
    return torch.stack(imgs), torch.cat(labels), torch.tensor(lens, dtype=torch.long)


class CRNN(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.rnn = nn.LSTM(256, 256, num_layers=2,
                           bidirectional=True, batch_first=True, dropout=0.3)
        self.fc  = nn.Linear(512, num_class)

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)
        o, _ = self.rnn(f)
        return self.fc(o).permute(1, 0, 2)


def main():
    parser = argparse.ArgumentParser(description="Train the CRNN digit recognizer.")
    parser.add_argument("--data",   required=True, help="Directory containing crops and labels.csv")
    parser.add_argument("--output", required=True, help="Output directory for checkpoints")
    parser.add_argument("--epochs",     type=int, default=200)
    parser.add_argument("--early-stop", type=int, default=50)
    parser.add_argument("--batch",      type=int, default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    data_dir   = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path   = data_dir / "labels.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        if "." not in r["lcd_value"]:
            r["lcd_value"] = r["lcd_value"] + ".0"

    rows = [r for r in rows if (data_dir / r["filename"]).exists()]
    print(f"Valid samples: {len(rows)}")

    random.seed(args.seed)
    random.shuffle(rows)
    split      = int(len(rows) * 0.9)
    train_rows = rows[:split]
    val_rows   = rows[split:]
    print(f"Train: {len(train_rows)}  Val: {len(val_rows)}")

    print("Pre-loading images...")
    image_cache = {}
    for r in rows:
        img = Image.open(data_dir / r["filename"]).convert("RGB")
        image_cache[r["filename"]] = img
    print("Pre-load done.")

    train_loader = DataLoader(
        LCDDataset(train_rows, image_cache, augment=True),
        batch_size=args.batch, shuffle=True, collate_fn=collate_fn,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        LCDDataset(val_rows, image_cache, augment=False),
        batch_size=args.batch, shuffle=False, collate_fn=collate_fn,
        num_workers=0, pin_memory=True,
    )

    model     = CRNN(NUM_CLASS).to(device)
    ctc_loss  = nn.CTCLoss(blank=BLANK, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    best_acc   = 0.0
    best_epoch = 0
    no_improve = 0

    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for imgs, labels, label_lens in train_loader:
            imgs       = imgs.to(device, non_blocking=True)
            logits     = model(imgs)
            log_probs  = logits.log_softmax(2)
            input_lens = torch.full((imgs.size(0),), logits.size(0), dtype=torch.long)
            loss = ctc_loss(log_probs, labels, input_lens, label_lens)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        correct = 0
        preds   = []
        with torch.no_grad():
            for imgs, labels, label_lens in val_loader:
                imgs     = imgs.to(device, non_blocking=True)
                pred_ids = model(imgs).log_softmax(2).argmax(2).permute(1, 0)
                offset = 0
                for pred_row, llen in zip(pred_ids, label_lens):
                    pred_str  = decode(pred_row.cpu().tolist())
                    label_str = ''.join(CHARS[j] for j in labels[offset:offset + llen].tolist())
                    offset += llen
                    if pred_str == label_str:
                        correct += 1
                    preds.append((pred_str, label_str))

        acc = correct / len(val_rows)
        if acc > best_acc:
            best_acc   = acc
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), str(output_dir / "best_crnn.pth"))
        else:
            no_improve += 1

        if epoch % 20 == 0:
            print(f"Ep{epoch:3d} | loss:{total_loss / len(train_loader):.4f}"
                  f" | val:{acc:.2%} | best:{best_acc:.2%}(ep{best_epoch})")
            print(f"  samples: {preds[:4]}")

        if no_improve >= args.early_stop:
            print(f"Early stop at epoch {epoch}, best:{best_acc:.2%}(ep{best_epoch})")
            break

    print(f"\nTraining done. Best accuracy: {best_acc:.2%} (epoch {best_epoch})")

    model.load_state_dict(torch.load(str(output_dir / "best_crnn.pth")))
    torch.save(model.state_dict(), str(output_dir / "final_crnn.pth"))

    with open(str(output_dir / "crnn_config.json"), "w") as f:
        json.dump({
            "num_class": NUM_CLASS,
            "chars": CHARS,
            "blank": BLANK,
            "min_val": 0.1,
            "max_val": 9999.9,
        }, f)

    print("Saved: final_crnn.pth + crnn_config.json")

    print("\nFull validation results:")
    model.eval()
    with torch.no_grad():
        for imgs, labels, label_lens in val_loader:
            imgs     = imgs.to(device, non_blocking=True)
            pred_ids = model(imgs).log_softmax(2).argmax(2).permute(1, 0)
            offset = 0
            for pred_row, llen in zip(pred_ids, label_lens):
                pred_str  = decode(pred_row.cpu().tolist())
                label_str = ''.join(CHARS[j] for j in labels[offset:offset + llen].tolist())
                offset += llen
                match = "OK " if pred_str == label_str else "BAD"
                print(f"  {match} pred:{pred_str}  actual:{label_str}")


if __name__ == "__main__":
    main()
