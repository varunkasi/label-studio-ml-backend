#!/usr/bin/env python3
"""Quick color analysis of crops using disk frame cache."""
import sys, os
sys.path.insert(0, "/app")
from interview.cache_manager import load_session
from interview.state import CropLabel
from PIL import Image
import numpy as np

session = load_session("p225664_t245750647")
accepted = [c for c in session.crops.values()
            if c.label == CropLabel.ACCEPTED and c.features is not None]
accepted.sort(key=lambda c: c.frame_idx)

cache_dir = "/data/adapters/p225664_t245750647/frames"

print("=== Color analysis of ALL %d accepted crops ===" % len(accepted))
print("%6s  %10s  %15s  %15s  %15s  %s" % (
    "frame", "size", "avg_rgb", "top_rgb", "bot_rgb", "crop_id"))
print("-" * 90)

for c in accepted:
    fi = c.frame_idx
    nearest = round(fi / 3) * 3
    fpath = os.path.join(cache_dir, "%08d.jpg" % nearest)
    if not os.path.exists(fpath):
        fpath = os.path.join(cache_dir, "%08d.jpg" % fi)
    if not os.path.exists(fpath):
        print("%6d  SKIP (no frame)" % fi)
        continue

    img = Image.open(fpath)
    x1, y1, x2, y2 = [int(v) for v in c.xyxy]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    crop_img = img.crop((x1, y1, x2, y2))

    arr = np.array(crop_img)
    h, w = arr.shape[:2]
    top_half = arr[:h//2].mean(axis=(0, 1))
    bot_half = arr[h//2:].mean(axis=(0, 1))
    overall = arr.mean(axis=(0, 1))

    print("%6d  %4dx%-4d  [%3.0f,%3.0f,%3.0f]  [%3.0f,%3.0f,%3.0f]  [%3.0f,%3.0f,%3.0f]  %s" % (
        fi, x2-x1, y2-y1,
        overall[0], overall[1], overall[2],
        top_half[0], top_half[1], top_half[2],
        bot_half[0], bot_half[1], bot_half[2],
        c.crop_id[:16]))
