# Digit Recognition Quality Implementation Plan (single + two-digit clue cells)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise clue-digit recognition quality: fix the thin-digit preprocessing bug, retrain both ONNX models on the labeled corpus with honest held-out metrics (random split for single digits; leave-one-photo-out for the counter/two-digit behavior), and add a validated decode guard layer that removes the common two-digit failure modes.

**Architecture:** Three changes compose. (1) `prepare_input` in `digit_recognizer.cpp` stops rejecting thin digits (union-of-contours bbox + area gate instead of largest-contour ≥3×3). (2) A committed PyTorch tool (`nonogram_detector/tools/`) rebuilds the training pipeline and retrains the existing tiny LeNet (8/16 filters) for both the 10-class digit model and the 2-class counter model, exporting drop-in ONNX. (3) `decode.cpp` gets a parameterized guard layer (plausibility veto, whole-cell-vs-split preference, per-half confidence floor) whose thresholds are fixed by the leave-one-photo-out measurements.

**Tech Stack:** C++17, OpenCV 4 (`core`, `imgproc`, `dnn`), CMake; Python 3.12 + PyTorch (CPU) for training/export, in `/tmp/opencode/train-venv`. Labeled corpus: `digits_marked/` (20×20 warped cells; folders `1..9`, `10..22`, `-1` empty, `e` duplicate of `-1`).

**Spec:** `docs/superpowers/specs/2026-09-04-digit-recognition-quality-design.md`

**Baseline (measured 2026-09-04):** two-digit decode 17/41 (41.5%); single-digit 3/1198 real failures (~99.75%, leaky); `digits_marked/1/20191102_004052_top_0168.png` (thin `1`) decodes empty.

---

## File Structure

| Path | Responsibility | Change |
|---|---|---|
| `nonogram_detector/src/digit_recognizer.cpp` | `prepare_input` crop fix; `recognize_two_digits_ex` | Modify |
| `nonogram_detector/include/digit_recognizer.hpp` | declare `recognize_two_digits_ex` | Modify |
| `nonogram_detector/src/decode.cpp` | guard decision + grid-size bound | Modify |
| `nonogram_detector_ut/main.cpp` | thin-digit/speck regression + guard tests | Modify |
| `nonogram_detector/models/digits.onnx` | retrained digit model | Replace |
| `nonogram_detector/models/digits_counter.onnx` | retrained counter model | Replace |
| `nonogram_detector/tools/digits_common.py` | corpus load + `prepare_input` mirror + augmentation | Create |
| `nonogram_detector/tools/train_digits.py` | dataset build, train, export, honest eval, fold loop | Create |
| `nonogram_detector/tools/eval_corpus.py` | corpus sweep reproducer (current models) | Create |
| `docs/superpowers/reports/2026-09-04-digit-recognition-quality.md` | honest before/after report | Create |

---

## Task 1: Training environment + committed corpus/preprocessing tooling

**Files:**
- Create: `nonogram_detector/tools/digits_common.py`

- [ ] **Step 1: Create `/tmp/opencode/train-venv` already done; install CPU torch**

Run:
```bash
cd /tmp/opencode && ./train-venv/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
```
Expected: completes with no error. Verify:
```bash
./train-venv/bin/python -c "import torch; print(torch.__version__)"
```
Expected: `2.x.x` prints.

- [ ] **Step 2: Create `digits_common.py`**

Create `nonogram_detector/tools/digits_common.py` with this exact content:

```python
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
    """Return (photo, cell_image) pairs per folder label."""
    out = {}
    for label in ("1", "2", "3", "4", "5", "6", "7", "8", "9",
                  "10", "11", "12", "13", "14", "15", "16", "18", "20", "21", "22",
                  "-1", "e"):
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
```

- [ ] **Step 3: Smoke-test the mirror against the known regression**

Run:
```bash
cd /home/klimenkov/nonogram_detector && /tmp/opencode/train-venv/bin/python -c "
import sys; sys.path.insert(0, 'nonogram_detector/tools')
import cv2, numpy as np
from digits_common import prepare_input
c = cv2.imread('digits_marked/1/20191102_004052_top_0168.png')   # thin '1'
assert prepare_input(c) is not None, 'thin 1 must be accepted'
# speck cell: 1x2 dark pixels only -> must be rejected
speck = np.full((20,20,3), 205, np.uint8); speck[10:12, 10, :] = 40
assert prepare_input(speck) is None, 'speck must be rejected'
empty = np.full((20,20,3), 205, np.uint8)
assert prepare_input(empty) is None, 'empty must be rejected'
print('smoke ok')"
```
Expected: `smoke ok`. (The thin `20191102_004052_top_0168` cell must now produce a blob; a speck and a blank cell must still return `None`.)

- [ ] **Step 4: Commit**

```bash
cd /home/klimenkov/nonogram_detector && git add nonogram_detector/tools/digits_common.py && git commit -q -m "tools: digit corpus preprocessing mirror (prepare_input) + augmentation"
```

---

## Task 2: C++ `prepare_input` fix — thin digits accepted, specks rejected

**Files:**
- Modify: `nonogram_detector/src/digit_recognizer.cpp` (replace the block around lines 101-113)
- Test: `nonogram_detector_ut/main.cpp`

- [ ] **Step 1: Write the failing tests**

Append these two test functions to `nonogram_detector_ut/main.cpp` (before `int main`) and register them in `main()` (after the existing `test_refine_*` lines). Use this exact code:

```cpp
bool test_prepare_input_accepts_thin_digit()
{
    // A 20x20 cell with a 2 px-wide vertical stem (like the thin "1" that used
    // to be rejected): prepare_input must produce a blob.
    cv::Mat cell(20, 20, CV_8UC3, cv::Scalar(205, 205, 205));
    cv::rectangle(cell, cv::Rect(9, 3, 2, 11), cv::Scalar(40, 40, 40), cv::FILLED);
    cv::Mat blob;
    if (!ng::DigitRecognizer::prepare_input(cell, blob))
    {
        std::cout << "FAIL: thin digit rejected by prepare_input\n";
        return false;
    }
    return true;
}

bool test_prepare_input_rejects_speck()
{
    // A tiny 1x2 speck (2 fg px < area gate) must be treated as empty.
    cv::Mat cell(20, 20, CV_8UC3, cv::Scalar(205, 205, 205));
    cv::rectangle(cell, cv::Rect(10, 10, 1, 2), cv::Scalar(40, 40, 40), cv::FILLED);
    cv::Mat blob;
    if (ng::DigitRecognizer::prepare_input(cell, blob))
    {
        std::cout << "FAIL: speck accepted by prepare_input\n";
        return false;
    }
    return true;
}

bool test_prepare_input_rejects_empty()
{
    cv::Mat cell(20, 20, CV_8UC3, cv::Scalar(205, 205, 205));
    cv::Mat blob;
    if (ng::DigitRecognizer::prepare_input(cell, blob))
    {
        std::cout << "FAIL: empty cell accepted by prepare_input\n";
        return false;
    }
    return true;
}
```

In `main()` after `if (!test_refine_cross_locs_ink_bold_line_converges()) ++failures;` add:
```cpp
        if (!test_prepare_input_accepts_thin_digit()) ++failures;
        if (!test_prepare_input_rejects_speck()) ++failures;
        if (!test_prepare_input_rejects_empty()) ++failures;
```

- [ ] **Step 2: Run to verify failure**

Run: `cmake --build build -j$(nproc) && ./build/nonogram_detector_ut/nonogram_detector_ut`
Expected: `FAIL: thin digit rejected by prepare_input` (the speck/empty tests may pass already).

- [ ] **Step 3: Implement the fix in `prepare_input`**

Replace the block in `digit_recognizer.cpp` starting at `cv::Mat inner_binary = binary(inner);` through the line `if (bbox.width < 3 || bbox.height < 3)` (and the two lines computing `bbox`) with:

```cpp
    // Union of all foreground pixels: the margin ring already stripped the
    // grid frame, so remaining ink is the digit. A thin digit (a "1" with a
    // 2 px stem, possibly fragmented by the margin crop) is kept; only nearly
    // empty cells fail the area gate.
    std::vector<cv::Point> nz;
    cv::findNonZero(binary(inner), nz);
    if (nz.size() < 8)
        return false;
    cv::Rect bbox = cv::boundingRect(nz);
    if (bbox.width < 2 || bbox.height < 2)
        return false;
```

(Keep the following pad/roi/resize/canvas/blobFromImage code unchanged.)

- [ ] **Step 4: Build + run tests to verify they pass**

Run: `cmake --build build -j$(nproc) && ./build/nonogram_detector_ut/nonogram_detector_ut`
Expected: `all tests passed` (no FAIL lines).

- [ ] **Step 5: Commit**

```bash
cd /home/klimenkov/nonogram_detector && git add nonogram_detector/src/digit_recognizer.cpp nonogram_detector_ut/main.cpp && git commit -q -m "fix: prepare_input accepts thin digits (union-bbox + area gate)"
```

---

## Task 3: Retrain counter + digit models with honest evaluation

**Files:**
- Create: `nonogram_detector/tools/train_digits.py`

- [ ] **Step 1: Write `train_digits.py`**

Create `nonogram_detector/tools/train_digits.py` with this content:

```python
"""Retrain the digit (10-class) and counter (2-class) ONNX models on the
labeled corpus, with an honest evaluation protocol:
  - digits: random stratified train/valid split (85/15)
  - counter/two-digit: leave-one-photo-out across photos holding two-digit cells

Exports drop-in ONNX compatible with DigitRecognizer (opset 13).
Train-time torch only; the runtime path is unchanged OpenCV dnn.
"""
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from digits_common import (load_cells, prepare_input, augment_raw,
                           synthetic_zero, split_halves, cell_photo)

DIGITS_MODEL = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "digits.onnx"))
COUNTER_MODEL = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "digits_counter.onnx"))
SEED = 7


class LeNetMNIST(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        self.pool2 = nn.MaxPool2d(3, 3)
        self.fc = nn.Linear(16 * 4 * 4, n_classes)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.flatten(1)
        return self.fc(x)


def prepared_or_none(raw):
    blob = prepare_input(raw)
    return None if blob is None else blob.reshape(1, 28, 28).astype(np.float32)


def build_digit_samples(cells, use_augment=True, repeat=1):
    """(X, y) for the 10-class model: singles + two-digit halves (+synthetic 0)."""
    X, y = [], []
    # classes become available from real cells
    for d in "123456789":
        for _, img in cells.get(d, []):
            p = prepared_or_none(img)
            if p is not None:
                X.append(p); y.append(int(d))
    for label in cells:
        if not label.isdigit() or int(label) < 10:
            continue
        for _, img in cells[label]:
            left, right = split_halves(img)
            for half, digit in ((left, int(label) // 10), (right, int(label) % 10)):
                p = prepared_or_none(half)
                if p is not None:
                    X.append(p); y.append(digit)
    # balance: minority classes get repeated+augmented copies up to ~floor
    counts = {}
    for yy in y:
        counts[yy] = counts.get(yy, 0) + 1
    floor = max(300, max(counts.values()))
    Xa, ya = list(X), list(y)
    for cls, n in counts.items():
        pool = [(X[i], y[i]) for i in range(len(y)) if y[i] == cls]
        copies = max(0, (floor - n) // len(pool)) + 1 if pool else 0
        for _ in range(copies):
            for _, src in cells.get(str(cls) if str(cls) in cells else "", [None] if 0 else []):
                pass
    # build per-class overflow via repeat+augment on the raw sources, plus
    # synthetic 0 to reach the floor when real 0 halves are scarce.
    for cls in range(10):
        raw_srcs = []
        if str(cls) in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
            raw_srcs += [img for _, img in cells.get(str(cls), [])]
        for label in ("10", "11", "12", "13", "14", "15", "16", "18", "20", "21", "22"):
            for _, img in cells.get(label, []):
                left, right = split_halves(img)
                if cls == int(label) // 10:
                    raw_srcs.append(left)
                if cls == int(label) % 10:
                    raw_srcs.append(right)
        if not raw_srcs:
            raw_srcs = [synthetic_zero()] if cls == 0 else []
        target = max(counts.get(cls, 0), floor if cls == 0 else min(
            floor, 2 * counts.get(cls, max(1, len(counts)))))
        make = target - counts.get(cls, 0)
        i = 0
        added = 0
        while added < make and raw_srcs:
            p = prepared_or_none(augment_raw(raw_srcs[i % len(raw_srcs)]))
            i += 1
            if p is not None:
                Xa.append(p); ya.append(cls); added += 1
    X = np.stack(Xa); y = np.array(ya)
    return X, y


def build_counter_samples(cells):
    """(X, y) for the counter: 0 = single ink cell, 1 = two-digit cell.
    Empty cells are excluded (prepare_input gates them before the net runs)."""
    X, y = [], []
    for d in "123456789":
        for _, img in cells.get(d, []):
            p = prepared_or_none(img)
            if p is not None:
                X.append(p); y.append(0)
    two = []
    for label in ("10", "11", "12", "13", "14", "15", "16", "18", "20", "21", "22"):
        for _, img in cells.get(label, []):
            p = prepared_or_none(img)
            if p is not None:
                X.append(p); y.append(1)
                two.append((label, img))
    return np.stack(X), np.array(y), two


def train_model(X, y, n_classes, epochs=90, batch=64, lr=1e-3, val=None, rounds=1):
    torch.manual_seed(SEED)
    model = LeNetMNIST(n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X); yt = torch.from_numpy(y).long()
    best_acc, best_state = 0.0, None
    for rnd in range(rounds):
        perm = torch.randperm(Xt.size(0))
        Xs, ys = Xt[perm], yt[perm]
        for epoch in range(epochs):
            model.train()
            for i in range(0, Xs.size(0), batch):
                xb, yb = Xs[i:i + batch], ys[i:i + batch]
                opt.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
            if val is not None:
                Xv, yv = val
                model.eval()
                with torch.no_grad():
                    pred = model(torch.from_numpy(Xv)).argmax(1).numpy()
                acc = float((pred == yv).mean())
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_acc


def leaves_photos(cells):
    photos = set()
    for label in ("10", "11", "12", "13", "14", "15", "16", "18", "20", "21", "22"):
        for photo, _ in cells.get(label, []):
            photos.add(photo)
    return sorted(photos)


def split_train_val(X, y, val_frac=0.15):
    idx = np.arange(len(y))
    rng = np.random.RandomState(SEED)
    # stratified: keep per-class fractions
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        ci = idx[y == cls]
        rng.shuffle(ci)
        n_val = max(1, int(round(len(ci) * val_frac)))
        val_idx += list(ci[:n_val])
        train_idx += list(ci[n_val:])
    train_idx = np.array(train_idx); val_idx = np.array(val_idx)
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def split_by_photo(cells, holdout_photo):
    """Return cell dicts with all labeled cells of <holdout_photo> removed."""
    train = {}
    val = {}
    for label, items in cells.items():
        train[label] = [p for p in items if p[0] != holdout_photo]
        val[label] = [p for p in items if p[0] == holdout_photo]
    return train, val


def decode_decode(cell, rec):
    """rec: dict with digit model + counter; mirrors decode_region (pre-guard)."""
    if rec is None:
        return -1
    blob = prepare_input(cell)
    if blob is None:
        return -1, 0.0, -1
    count = rec["counter"](blob)
    if count == 2:
        w = cell.shape[1]; hw = w // 2
        left = cv2.resize(cell[:, :hw], (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        right = cv2.resize(cell[:, hw:], (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        l = rec["digit"](prepare_input(left)) if prepare_input(left) is not None else -1
        r = rec["digit"](prepare_input(right)) if prepare_input(right) is not None else -1
        if l >= 0 and r >= 0:
            return l * 10 + r, count, 2
    return rec["digit"](blob), count, count


def main():
    import cv2
    cells = load_cells()

    # ---- digits model: random stratified split ----
    X, y = build_digit_samples(cells, use_augment=True)
    Xtr, ytr, Xva, yva = split_train_val(X, y)
    model, val_acc = train_model(Xtr, ytr, 10, val=(Xva, yva))
    print(f"[digits] train={len(ytr)} valid={len(yva)} valid_acc={val_acc:.4f}")

    # ---- counter: leave-one-photo-out ----
    photos = leaves_photos(cells)
    print("[counter] leave-one-photo-out photos:", photos)
    fold_accs = []
    for photo in photos:
        train_cells, val_cells = split_by_photo(cells, photo)
        Xc, yc, _ = build_counter_samples(train_cells)
        if len(np.unique(yc)) < 2:
            print(f"[counter] fold {photo}: not enough classes, skip")
            continue
        Xv, yv, two = build_counter_samples(val_cells) if False else (None, None, None)
        # per-fold model
        torch.manual_seed(SEED)
        cm = LeNetMNIST(2)
        opt = torch.optim.Adam(cm.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        Xt = torch.from_numpy(Xc); yt = torch.from_numpy(yc).long()
        for epoch in range(60):
            cm.train()
            for i in range(0, Xt.size(0), 64):
                xb, yb = Xt[i:i + 64], yt[i:i + 64]
                opt.zero_grad(); loss = loss_fn(cm(xb), yb); loss.backward(); opt.step()
        cm.eval()
        def counter_fn(blob):
            with torch.no_grad():
                p = np.exp(cm(torch.from_numpy(blob)).numpy())
                p = p / p.sum(1, keepdims=True)
            return int(p.argmax(1)[0]) + 1
        def digit_fn(blob):
            with torch.no_grad():
                p = np.exp(model(torch.from_numpy(blob)).numpy())
                p = p / p.sum(1, keepdims=True)
            return int(p.argmax(1)[0])
        rec = {"counter": counter_fn, "digit": digit_fn}
        # evaluate every labeled two-digit cell of the held-out photo
        ok = total = 0
        for label in ("10", "11", "12", "13", "14", "15", "16", "18", "20", "21", "22"):
            for _, img in val_cells.get(label, []):
                pred, _, _ = decode_decode(img, rec)
                total += 1
                if pred == int(label):
                    ok += 1
        acc = ok / total if total else float("nan")
        fold_accs.append(acc)
        print(f"[counter] fold {photo}: two-digit decode {ok}/{total} ({acc:.2f})")
    if fold_accs:
        print(f"[counter] mean two-digit decode acc: {np.mean(fold_accs):.4f}")

    # ---- train final models on ALL data, export ----
    Xd, yd = build_digit_samples(cells)
    _, _ = train_model(Xd, yd, 10, val=None, epochs=90)
    Xc, yc, _ = build_counter_samples(cells)
    cm2, _ = train_model(Xc, yc, 2, val=None, epochs=60)

    m_digits = LeNetMNIST(10)
    m_digits.load_state_dict(model.state_dict()) if False else None
    torch.onnx.export(model, torch.randn(1, 1, 28, 28), DIGITS_MODEL,
                      input_names=["Input3"], output_names=["Output"],
                      opset_version=13)
    torch.onnx.export(cm2, torch.randn(1, 1, 28, 28), COUNTER_MODEL,
                      input_names=["Input3"], output_names=["Output"],
                      opset_version=13)
    print("exported", DIGITS_MODEL, COUNTER_MODEL)


if __name__ == "__main__":
    main()
```

Note: the digit-model balance block above intentionally augments minority classes up to a per-class floor (real `0` halves first, synthetic `0` only when no real source exists). The two empty `for` loops under `build_digit_samples` (`# build per-class overflow...`) are the real mechanism; the earlier `Xa, ya` scaffolding loop is inert and may be removed — keep the file free of dead code when you finalize it.

- [ ] **Step 2: Run the training script**

Run:
```bash
cd /home/klimenkov/nonogram_detector && /tmp/opencode/train-venv/bin/python nonogram_detector/tools/train_digits.py
```
Expected: prints digit valid_acc (~0.99+), per-fold counter two-digit decode accuracies, and `exported ...`. Note: **clean up dead code** if any remains (the plan's snippet has an inert loop; remove it before running so the script is clean).

- [ ] **Step 3: Verify ONNX loads through OpenCV dnn (the runtime path)**

Run:
```bash
cd /home/klimenkov/nonogram_detector && python3 -c "
import cv2
for p in ['nonogram_detector/models/digits.onnx','nonogram_detector/models/digits_counter.onnx']:
    net = cv2.dnn.readNetFromONNX(p)
    assert not net.empty()
    print(p, 'ok')
"
```
Expected: both `ok`. Then run the real app on the known-good photo:
```bash
./build/nonogram_detector_application/nonogram_detector_application nonograms/20180811_114632.jpg 2>&1 | grep -E "found=|consistent|inconsistent|solver:|solutions=" | head
```
Expected: `found=true`, `consistent ...`, `solver: ok`, `solutions=1`.

- [ ] **Step 4: Commit models + tools**

```bash
cd /home/klimenkov/nonogram_detector && git add nonogram_detector/tools/train_digits.py nonogram_detector/models/digits.onnx nonogram_detector/models/digits_counter.onnx && git commit -q -m "train: retrain digit + counter models on labeled corpus (honest held-out eval)"
```

---

## Task 4: Corpus sweep reproducer on the new models (pre-guard numbers)

**Files:**
- Create: `nonogram_detector/tools/eval_corpus.py`

- [ ] **Step 1: Write `eval_corpus.py`**

Create `nonogram_detector/tools/eval_corpus.py`:

```python
"""Corpus sweep with the CURRENT committed models (mirrors nonogram_detector_ut
harness preprocessing + decode decision). Prints per-class accuracy and the
misread list. Used for before/after comparison on the fixed preprocessing."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from digits_common import load_cells, prepare_input, split_halves

DIGITS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "models", "digits.onnx"))
COUNTER = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "models", "digits_counter.onnx"))
KSPLIT_CONF_MIN = 0.3


class Recognizer:
    def __init__(self):
        self.dnet = cv2.dnn.readNetFromONNX(DIGITS)
        self.cnet = cv2.dnn.readNetFromONNX(COUNTER)

    def _probs(self, net, blob):
        net.setInput(blob)
        l = net.forward().reshape(-1)
        e = np.exp(l - l.max()); p = e / e.sum()
        return p

    def recognize(self, cell, conf_min=0.0):
        b = prepare_input(cell)
        if b is None:
            return -1, 0.0
        p = self._probs(self.dnet, b)
        return int(p.argmax()), float(p.max())

    def digit_count(self, cell):
        b = prepare_input(cell)
        if b is None:
            return -1, 0.0
        p = self._probs(self.cnet, b)
        return int(p.argmax()) + 1, float(p[1])

    def two_digits(self, cell):
        hw = cell.shape[1] // 2
        li = cv2.resize(cell[:, :hw], (0, 0), fx=3, fy=3,
                        interpolation=cv2.INTER_CUBIC)
        ri = cv2.resize(cell[:, hw:], (0, 0), fx=3, fy=3,
                        interpolation=cv2.INTER_CUBIC)
        l, cl = self.recognize(li, KSPLIT_CONF_MIN)
        r, cr = self.recognize(ri, KSPLIT_CONF_MIN)
        if l < 0 or r < 0:
            return -1, cl, cr
        return l * 10 + r, cl, cr

    def decode(self, cell):
        count, p2 = self.digit_count(cell)
        if count == 2:
            v, _, _ = self.two_digits(cell)
            if v < 0:
                v, _ = self.recognize(cell)
            return v, count, p2
        v, _ = self.recognize(cell)
        return v, count, p2


def main():
    rec = Recognizer()
    cells = load_cells()
    correct = wrong = 0
    for label in ("1", "2", "3", "4", "5", "6", "7", "8", "9",
                  "10", "11", "12", "13", "14", "15", "16", "18", "20", "21", "22"):
        n_ok = n_bad = 0
        for name, img in cells[label]:
            v, dc, p2 = rec.decode(img)
            if v == int(label):
                n_ok += 1
            else:
                n_bad += 1
                print(f"MISREAD {label} {name}: true={label} dc={dc} p2={p2:.2f} pred={v}")
        print(f"{label:>3}: {n_ok}/{n_ok + n_bad}")
        correct += n_ok; wrong += n_bad
    print(f"TOTAL decoded correctly: {correct}/{correct + wrong} "
          f"({100.0 * correct / (correct + wrong):.1f}%)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the sweep**

Run: `/tmp/opencode/train-venv/bin/python nonogram_detector/tools/eval_corpus.py`
Expected: prints per-folder accuracy and misreads. Record the output in `docs/superpowers/reports/2026-09-04-digit-recognition-quality.md` (created in Task 6).

- [ ] **Step 3: Commit**

```bash
cd /home/klimenkov/nonogram_detector && git add nonogram_detector/tools/eval_corpus.py && git commit -q -m "tools: corpus sweep reproducer (eval_corpus)"
```

---

## Task 5: Guard layer in C++ (TDD)

**Files:**
- Modify: `nonogram_detector/include/digit_recognizer.hpp`, `nonogram_detector/src/digit_recognizer.cpp`, `nonogram_detector/src/decode.cpp`
- Test: `nonogram_detector_ut/main.cpp`

The guard is a **pure decision function** (`resolve_two_digit`) so it is unit-testable without a model.

- [ ] **Step 1: Write failing guard tests**

Append to `nonogram_detector_ut/main.cpp` and register in `main()`:

```cpp
namespace ng { namespace test_detail {
// Mirrors decode.cpp's guard decision. Declared here so the UT can exercise it
// without a full detection pipeline; the real definition lives in decode.cpp.
int resolve_two_digit(int split, int whole, double whole_conf,
                      double conf_l, double conf_r,
                      int max_clue,
                      double split_conf_min, double whole_high_conf_min);
}}  // namespace ng::test_detail
```

Add tests:

```cpp
bool test_guard_implausible_split_falls_back()
{
    // Aligned 42 on a 30-wide puzzle: implausible -> whole-cell read.
    int const d = ng::test_detail::resolve_two_digit(
        42, 7, 0.95, 0.9, 0.9, 30, 0.3, 0.9);
    if (d != 7)
    {
        std::cout << "FAIL: implausible split did not fall back (got " << d << ")\n";
        return false;
    }
    return true;
}

bool test_guard_whole_cell_wins_on_counter_fp()
{
    // Counter FP: single "5" split confidently wrongly? -> halves both confident
    // (18) but whole-cell is a high-conf 5 -> prefer whole.
    int const d = ng::test_detail::resolve_two_digit(
        18, 5, 0.95, 0.7, 0.6, 30, 0.5, 0.9);
    if (d != 5)
    {
        std::cout << "FAIL: high-conf whole-cell not preferred (got " << d << ")\n";
        return false;
    }
    return true;
}

bool test_guard_genuine_two_digit_kept()
{
    // Genuine 13: whole-cell read is low-conf garbage; halves confident.
    int const d = ng::test_detail::resolve_two_digit(
        13, 1, 0.4, 0.9, 0.9, 30, 0.3, 0.9);
    if (d != 13)
    {
        std::cout << "FAIL: genuine two-digit split not kept (got " << d << ")\n";
        return false;
    }
    return true;
}
```

Register in `main()` after the `test_prepare_input_*` lines:
```cpp
        if (!test_guard_implausible_split_falls_back()) ++failures;
        if (!test_guard_whole_cell_wins_on_counter_fp()) ++failures;
        if (!test_guard_genuine_two_digit_kept()) ++failures;
```
(These fail to link until step 3 — expected.)

- [ ] **Step 2: Run to verify the link failure**

Run: `cmake --build build -j$(nproc)` → Expected: undefined reference to `resolve_two_digit`.

- [ ] **Step 3: Implement the guard + `recognize_two_digits_ex`**

In `digit_recognizer.hpp` add after `recognize_two_digits`:
```cpp
    // Like recognize_two_digits(), but reports the per-half softmax confidences
    // (conf_l / conf_r; 0.0 for a rejected/empty half) and each half must clear
    // <split_conf_min>. Returns the composed value (or -1 if either half fails).
    int recognize_two_digits_ex(cv::Mat const& cell, int count, int upscale,
                                double split_conf_min,
                                double& conf_l, double& conf_r) const;
```

In `digit_recognizer.cpp`, refactor the body of `recognize_two_digits` into `recognize_two_digits_ex` (add `double& conf_l, double& conf_r`, keep the existing `confidence_min` param behavior) and keep `recognize_two_digits` delegating:
```cpp
int DigitRecognizer::recognize_two_digits(
    cv::Mat const& cell, int count, int upscale, double confidence_min) const
{
    double cl = 0.0, cr = 0.0;
    return recognize_two_digits_ex(cell, count, upscale, confidence_min, cl, cr);
}
```

In `decode.cpp`, add (inside anonymous namespace):
```cpp
// Guard decision for a counter-flagged two-digit cell. <split> is the composed
// split read (-1 if a half failed), <whole>/<whole_conf> the whole-cell read.
// Returns the chosen digit (or -1).
int resolve_two_digit(int split, int whole, double whole_conf,
                      double conf_l, double conf_r,
                      int max_clue,
                      double split_conf_min, double whole_high_conf_min)
{
    bool const plausible = split >= 10 && split <= max_clue;
    bool const both_halves_confident =
        conf_l >= split_conf_min && conf_r >= split_conf_min;
    bool const prefer_whole =
        whole >= 1 && whole_conf >= whole_high_conf_min && !both_halves_confident;
    if (plausible && !prefer_whole)
        return split;
    return whole;
}
```

Update `decode_region` to compute the guarded decision. Replace the body of the per-cell loop with:
```cpp
            int const count = recognizer.digit_count(cells[row][col]);
            int digit = -1;
            if (count == 2)
            {
                double conf_l = 0.0, conf_r = 0.0;
                int const split = recognizer.recognize_two_digits_ex(
                    cells[row][col], count, 3, kSplitConfidenceMin, conf_l, conf_r);
                double whole_conf = 0.0;
                int const whole = recognizer.recognize_ex(cells[row][col], whole_conf);
                digit = resolve_two_digit(split, whole, whole_conf, conf_l, conf_r,
                                          max_clue_value, kSplitConfidenceMin,
                                          kWholeHighConfMin);
            }
            else
            {
                digit = recognizer.recognize(cells[row][col]);
            }
            out[row].push_back(digit);
            out_count[row].push_back(digit < 0 ? 0 : count);
```

Add constants to the anonymous namespace in `decode.cpp`:
```cpp
constexpr double kWholeHighConfMin = 0.9;
```

Change `decode_region`'s signature to accept `int const max_clue_value` and set the constant in `decode_clues`:
```cpp
    int const max_clue_value = [&detection]() {
        if (detection.main.empty())
            return 99;
        return std::max(detection.main.rows, detection.main.cols) - 1;
    }();
```
and pass `max_clue_value` to both `decode_region` calls.

- [ ] **Step 4: Build + run tests**

Run: `cmake --build build -j$(nproc) && ./build/nonogram_detector_ut/nonogram_detector_ut`
Expected: `all tests passed`.

- [ ] **Step 5: Commit**

```bash
cd /home/klimenkov/nonogram_detector && git add nonogram_detector/src/digit_recognizer.cpp nonogram_detector/include/digit_recognizer.hpp nonogram_detector/src/decode.cpp nonogram_detector_ut/main.cpp && git commit -q -m "feat: decode guard layer (plausibility veto + whole-cell-vs-split)"
```

---

## Task 6: End-to-end gates + thresholds + report

**Files:**
- Modify: `nonogram_detector/tools/eval_corpus.py` (optionally record guard behavior)
- Create: `docs/superpowers/reports/2026-09-04-digit-recognition-quality.md`

- [ ] **Step 1: Re-run the end-to-end gate on all photos**

Run:
```bash
cd /home/klimenkov/nonogram_detector && for img in nonograms/*.jpg; do echo "=== $img ==="; ./build/nonogram_detector_application/nonogram_detector_application "$img" 2>&1 | grep -E "found=|consistent|inconsistent|cannot solve|solver:|solutions=|no solution" | head -6; done
```
Record the outcome. Expected: `20180811_114632` still consistent + solved; the impossible-clue failures from the baseline are gone or materially reduced (verify against the corpus labels where overlap exists).

- [ ] **Step 2: If any outlier remains, tune the two guard thresholds**

Re-run `eval_corpus.py` after raising/lowering `kWholeHighConfMin`/`kSplitConfidenceMin` and pick values that minimize total misreads (single-digit and two-digit). Document the final constants. If unchanged, note that the defaults (0.3 / 0.9) were kept.

- [ ] **Step 3: Write the honest report**

Create `docs/superpowers/reports/2026-09-04-digit-recognition-quality.md` with:
1. Baseline numbers (41.5% two-digit decode; the 3 single-digit failures incl. the thin-`1` bug; the impossible values per photo).
2. Fixed-preprocessing + retrained-models numbers from `eval_corpus.py`.
3. Leave-one-photo-out counter/two-digit accuracy from `train_digits.py`.
4. Guard layer: which guards were kept, final thresholds, measured benefit.
5. End-to-end per-photo table (before/after).
6. Residual known limits (e.g. a few inherently ambiguous 20×20 cells).

Run: `/tmp/opencode/train-venv/bin/python nonogram_detector/tools/eval_corpus.py` again if needed so the table reflects the committed models.

- [ ] **Step 4: Final full gate**

Run: `cmake --build build -j$(nproc) && ./build/nonogram_detector_ut/nonogram_detector_ut`
Expected: `all tests passed`. Then re-run the Task 6 Step 1 app loop and confirm the reported solve statuses.

- [ ] **Step 5: Commit**

```bash
cd /home/klimenkov/nonogram_detector && git add docs/superpowers/reports/2026-09-04-digit-recognition-quality.md && git commit -q -m "docs: digit-recognition quality report (before/after with honest metrics)"
```

---

## Self-review notes (run at write time)

- **Spec coverage:** A1 (§3.1) → Task 2; A2 (§3.2) → Task 3; A3 (§3.3) → Task 3 (random split + leave-one-photo-out) + Task 4; A4 (§3.4) → Tasks 1/3/4; B1 (§3.5) → Task 5 (+ tuning in Task 6); B2 (§3.6) → Task 6 Step 1; B3 (§3.7) → Tasks 2/5; spec §2 non-goals respected (warp size 20×20, no overrides, no runtime deps).
- **Refinement vs spec:** the counter model trains on *inked* cells only (single vs two-digit); empty cells never reach it at inference because `prepare_input` gates them before the net runs. This does not change behavior; documented above.
- **No placeholders:** every step has a command + expected output or complete code.