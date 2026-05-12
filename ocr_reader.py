import os
import json
import re
import numpy as np
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

MODEL_DIR = Path(__file__).parent / "models"

with open(MODEL_DIR / "crnn_config.json") as f:
    _config = json.load(f)
CHARS     = _config['chars']
BLANK     = _config['blank']
NUM_CLASS = _config['num_class']


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


_crnn = None
_yolo = None


def get_crnn():
    global _crnn
    if _crnn is None:
        print("  [CRNN] Loading model...")
        _crnn = CRNN(NUM_CLASS)
        _crnn.load_state_dict(torch.load(
            str(MODEL_DIR / "final_crnn.pth"), map_location='cpu'))
        _crnn.eval()
    return _crnn


def get_yolo():
    global _yolo
    if _yolo is None:
        print("  [YOLO] Loading model...")
        from ultralytics import YOLO
        _yolo = YOLO(str(MODEL_DIR / "best.pt"))
    return _yolo


TRANSFORM = T.Compose([
    T.Resize((64, 256)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])


def decode(indices):
    result, prev = [], None
    for i in indices:
        if i != BLANK and i != prev:
            result.append(CHARS[i])
        prev = i
    return ''.join(result)


def read_image(path):
    try:
        with open(path, 'rb') as f:
            arr = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"  [ERROR] Failed to read: {path} ({e})")
        return None


def smart_resize(img):
    w, h = img.size
    if w > 400:
        scale = 300 / w
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def read_lcd_number(image_path):
    img_cv = read_image(image_path)
    if img_cv is None:
        return None

    h, w = img_cv.shape[:2]

    yolo    = get_yolo()
    results = yolo(image_path, verbose=False, conf=0.3)

    if not results or len(results[0].boxes) == 0:
        print("  [YOLO] No LCD detected")
        return None

    boxes    = results[0].boxes
    best_idx = boxes.conf.argmax().item()
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)

    pad = 5
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = img_cv[y1:y2, x1:x2]
    conf = boxes.conf[best_idx].item()
    print(f"  [YOLO] confidence:{conf:.2f}, region:{x2-x1}x{y2-y1}px")

    if crop.size == 0:
        print("  [ERROR] Empty crop region")
        return None

    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    crop_pil = smart_resize(crop_pil)
    tensor   = TRANSFORM(crop_pil).unsqueeze(0)

    crnn = get_crnn()
    with torch.no_grad():
        pred_ids = crnn(tensor).log_softmax(2).argmax(2).squeeze(1)

    pred_str = decode(pred_ids.tolist())
    print(f"  [CRNN] recognized: '{pred_str}'")

    if not re.match(r'^\d+\.?\d*$', pred_str):
        print(f"  [WARN] Invalid format: '{pred_str}'")
        return None

    try:
        val = float(pred_str)
        if not (0.1 <= val <= 9999.9):
            print(f"  [WARN] Value out of range: {val}")
            return None
        return val
    except ValueError:
        return None