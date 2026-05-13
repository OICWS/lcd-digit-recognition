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

_config = None


def get_config():
    global _config
    if _config is None:
        with open(MODEL_DIR / "crnn_config.json") as f:
            _config = json.load(f)
        _config.setdefault("min_val", 0.1)
        _config.setdefault("max_val", 9999.9)
    return _config


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
        cfg = get_config()
        _crnn = CRNN(cfg["num_class"])
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


def decode_with_confidence(log_probs_t):
    # log_probs_t: (T, num_class) for a single sample
    cfg = get_config()
    chars = cfg["chars"]
    blank = cfg["blank"]
    probs = log_probs_t.exp()
    ids = log_probs_t.argmax(-1).tolist()
    max_p = probs.max(-1).values.tolist()
    out_chars, char_confs, prev = [], [], None
    for i, p in zip(ids, max_p):
        if i != blank and i != prev:
            out_chars.append(chars[i])
            char_confs.append(p)
        prev = i
    pred = ''.join(out_chars)
    conf = float(min(char_confs)) if char_confs else 0.0
    return pred, conf


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


def _crop_lcd(img_cv, box_xyxy, pad=5):
    h, w = img_cv.shape[:2]
    x1, y1, x2, y2 = box_xyxy.astype(int)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return img_cv[y1:y2, x1:x2], (x2 - x1, y2 - y1)


def _validate(pred_str, min_val, max_val):
    if not re.match(r'^\d+\.?\d*$', pred_str):
        print(f"  [WARN] Invalid format: '{pred_str}'")
        return None
    try:
        val = float(pred_str)
    except ValueError:
        return None
    if not (min_val <= val <= max_val):
        print(f"  [WARN] Value out of range [{min_val}, {max_val}]: {val}")
        return None
    return val


def recognize_crop(crop_bgr, min_val=None, max_val=None):
    """Recognize digits from a pre-cropped LCD region (BGR ndarray).

    Skips YOLO entirely — use this when the caller has already located the
    LCD (e.g. visualization or evaluation scripts that batch YOLO separately).
    Returns (value, confidence). value is None if the CRNN output fails the
    format / range checks.
    """
    cfg = get_config()
    if min_val is None:
        min_val = cfg["min_val"]
    if max_val is None:
        max_val = cfg["max_val"]

    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0

    crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    crop_pil = smart_resize(crop_pil)
    tensor   = TRANSFORM(crop_pil).unsqueeze(0)

    crnn = get_crnn()
    with torch.no_grad():
        log_probs = crnn(tensor).log_softmax(2)

    pred_str, crnn_conf = decode_with_confidence(log_probs[:, 0, :])
    val = _validate(pred_str, min_val, max_val)
    return val, crnn_conf


def read_lcd_number(image_path, min_val=None, max_val=None):
    """Read a single LCD image. Returns (value, confidence) or (None, 0.0).

    confidence is the min per-character softmax probability from the CRNN.
    min_val / max_val override the range from crnn_config.json.
    """
    img_cv = read_image(image_path)
    if img_cv is None:
        return None, 0.0

    yolo    = get_yolo()
    results = yolo(image_path, verbose=False, conf=0.3)

    if not results or len(results[0].boxes) == 0:
        print("  [YOLO] No LCD detected")
        return None, 0.0

    boxes    = results[0].boxes
    best_idx = boxes.conf.argmax().item()
    box      = boxes.xyxy[best_idx].cpu().numpy()
    yolo_conf = boxes.conf[best_idx].item()

    crop, (cw, ch) = _crop_lcd(img_cv, box)
    print(f"  [YOLO] confidence:{yolo_conf:.2f}, region:{cw}x{ch}px")

    val, crnn_conf = recognize_crop(crop, min_val=min_val, max_val=max_val)
    print(f"  [CRNN] conf={crnn_conf:.3f}, value={val}")
    return val, crnn_conf


def read_lcd_batch(image_paths, min_val=None, max_val=None, batch_size=16):
    """Batched inference over a list of image paths.

    Returns a list of (value, confidence) tuples, one per input path (None for
    images where detection or recognition failed). YOLO and CRNN are each run
    in batches of up to ``batch_size``.
    """
    cfg = get_config()
    if min_val is None:
        min_val = cfg["min_val"]
    if max_val is None:
        max_val = cfg["max_val"]

    paths = [str(p) for p in image_paths]
    results_out = [(None, 0.0)] * len(paths)

    yolo = get_yolo()
    crnn = get_crnn()

    crop_tensors = []
    crop_indices = []

    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        det = yolo(chunk, verbose=False, conf=0.3)
        for j, res in enumerate(det):
            idx = start + j
            if res is None or len(res.boxes) == 0:
                continue
            img_cv = read_image(paths[idx])
            if img_cv is None:
                continue
            boxes = res.boxes
            best  = boxes.conf.argmax().item()
            box   = boxes.xyxy[best].cpu().numpy()
            crop, _ = _crop_lcd(img_cv, box)
            if crop.size == 0:
                continue
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crop_pil = smart_resize(crop_pil)
            crop_tensors.append(TRANSFORM(crop_pil))
            crop_indices.append(idx)

    if not crop_tensors:
        return results_out

    for start in range(0, len(crop_tensors), batch_size):
        batch = torch.stack(crop_tensors[start:start + batch_size])
        with torch.no_grad():
            log_probs = crnn(batch).log_softmax(2)  # (T, B, C)
        for k in range(log_probs.shape[1]):
            idx = crop_indices[start + k]
            pred_str, conf = decode_with_confidence(log_probs[:, k, :])
            val = _validate(pred_str, min_val, max_val)
            results_out[idx] = (val, conf)

    return results_out
