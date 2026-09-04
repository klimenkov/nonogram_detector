"""Shared corpus loading + preprocessing + augmentation for the digit models.

prepare_input() mirrors the FIXED C++ normalizer in
nonogram_detector/src/digit_recognizer.cpp (union-of-contours crop + area gate).
Keep the two in sync: changing one without the other breaks the
train-inference distribution match.
"""
import glob
import os
import random

import cv2
import numpy as np

CORPUS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "digits_marked"))

SINGLE, TWO_DIGIT, EMPTY, EXCLUDED = "single", "2digit", "empty", "excluded"

DIGIT_LABELS = ("1", "2", "3", "4", "5", "6", "7", "8", "9",
                "10", "11", "12", "13", "14", "15", "16", "18", "20", "21", "22")
TWO_DIGIT_LABELS = ("10", "11", "12", "13", "14", "15", "16", "18", "20", "21", "22")


def prepare_input(cell):
    """Return the 1x1x28x28 whitened blob, or None when the cell is empty.

    Mirrors the fixed C++ prepare_input exactly: Otsu inverse, 20% margin ring,
    all-foreground union bbox (>=2 px per axis), area gate (>=8 px), pad 2,
    aspect-preserving resize to 28x28 canvas, MNIST whitening.
    """
    if cell is None or cell.size == 0:
        return None
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    margin = max(2, int(round(binary.shape[1] * 0.2)))
    inner = binary[margin:binary.shape[0] - margin, margin:binary.shape[1] - margin]
    if inner.shape[0] < 3 or inner.shape[1] < 3:
        return None
    ys, xs = np.where(inner > 0)
    if ys.size < 8:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    if bw < 2 or bh < 2:
        return None
    pad = 2
    rx0 = max(0, margin + x0 - pad)
    ry0 = max(0, margin + y0 - pad)
    rx1 = min(binary.shape[1], margin + x1 + 1 + pad)
    ry1 = min(binary.shape[0], margin + y1 + 1 + pad)
    digit = binary[ry0:ry1, rx0:rx1]
    if digit.size == 0:
        return None
    k = 28
    scale = (k - 4) / max(digit.shape[1], digit.shape[0])
    nw = max(1, int(round(digit.shape[1] * scale)))
    nh = max(1, int(round(digit.shape[0] * scale)))
    resized = cv2.resize(digit, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((k, k), np.uint8)
    dx, dy = (k - nw) // 2, (k - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return cv2.dnn.blobFromImage(canvas, 1.0 / (255.0 * 0.3081), (k, k),
                                 (0.1307 / 0.3081,), swapRB=False, crop=False)


def cell_photo(name):
    """Photo id from a corpus filename like `20180811_114632_left_0105.png`."""
    return name.split("_", 1)[0]


def load_cells():
    """Return {label: [(photo, cell_image), ...]} per folder label."""
    out = {}
    for label in DIGIT_LABELS + ("-1", "e"):
        label_dir = os.path.join(CORPUS_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        out[label] = []
        for path in glob.glob(os.path.join(label_dir, "*.png")):
            img = cv2.imread(path)
            if img is None:
                continue
            name = os.path.relpath(path, CORPUS_DIR)
            out[label].append((cell_photo(name), img))
        out[label].sort(key=lambda p: p[0])
    return out


def augment_raw(cell):
    """Mild geometric + photometric jitter on a raw 20x20 cell (train only)."""
    h, w = cell.shape[:2]
    ang = random.uniform(-4.0, 4.0)
    sc = random.uniform(0.92, 1.08)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    a = cv2.warpAffine(cell, m, (w, h), flags=cv2.INTER_CUBIC,
                       borderMode=cv2.BORDER_REPLICATE)
    dx, dy = random.randint(-1, 1), random.randint(-1, 1)
    if dx or dy:
        m2 = np.float32([[1, 0, dx], [0, 1, dy]])
        a = cv2.warpAffine(a, m2, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if random.random() < 0.5:
        a = cv2.GaussianBlur(a, (3, 3), 0)
    return a


def synthetic_zero():
    """A synthetic '0' render for classes with too few real halves."""
    img = np.full((20, 20), 205, np.uint8)
    cv2.ellipse(img, (10, 10), (4, 7), 0, 0, 360, 30, thickness=3)
    return img


def split_halves(cell):
    """Left/right halves of a two-digit 20x20 cell (each 20x10)."""
    hw = cell.shape[1] // 2
    return cell[:, :hw], cell[:, hw:]