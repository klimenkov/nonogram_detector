"""Retrain the digit (10-class) and counter (2-class) ONNX models on the
labeled corpus, with an honest evaluation protocol:
  - digits: random stratified train/valid split (85/15)
  - counter/two-digit: leave-one-photo-out across photos holding two-digit cells

Exports drop-in ONNX compatible with DigitRecognizer (opset 9).
Train-time torch only; the runtime path is unchanged OpenCV dnn.
"""
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from digits_common import (load_cells, prepare_input, augment_raw,
                            synthetic_zero, split_halves)

DIGITS_MODEL = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "digits.onnx"))
COUNTER_MODEL = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models", "digits_counter.onnx"))
SEED = 7


class LeNetMNIST(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5, padding=2, bias=False)
        self.b1 = nn.Parameter(torch.zeros(8))
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.b2 = nn.Parameter(torch.zeros(16))
        self.pool2 = nn.MaxPool2d(3, 3, ceil_mode=True)
        self.fc = nn.Linear(16 * 5 * 5, n_classes)

    def forward(self, x):
        x = self.conv1(x) + self.b1.view(1, -1, 1, 1)
        x = self.pool1(self.relu(x))
        x = self.conv2(x) + self.b2.view(1, -1, 1, 1)
        x = self.pool2(self.relu(x))
        x = x.flatten(1)
        return self.fc(x)


def prepared_or_none(raw):
    blob = prepare_input(raw)
    return None if blob is None else blob.reshape(1, 28, 28).astype(np.float32)


def build_digit_samples(cells, use_augment=True, repeat=1):
    """(X, y) for the 10-class model: singles + two-digit halves (+synthetic 0)."""
    X, y = [], []
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

    counts = {}
    for yy in y:
        counts[yy] = counts.get(yy, 0) + 1
    floor = max(300, max(counts.values()))
    Xa, ya = list(X), list(y)

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
        target = max(counts.get(cls, 0),
                     floor if cls == 0 else min(floor, 2 * counts.get(cls, 1)))
        make = target - counts.get(cls, 0)
        i = 0
        added = 0
        while added < make and raw_srcs:
            p = prepared_or_none(augment_raw(raw_srcs[i % len(raw_srcs)]))
            i += 1
            if p is not None:
                Xa.append(p); ya.append(cls); added += 1
    return np.stack(Xa), np.array(ya)


def build_counter_samples(cells):
    """(X, y) for the counter: 0 = single ink cell, 1 = two-digit cell."""
    X, y, two = [], [], []
    for d in "123456789":
        for _, img in cells.get(d, []):
            p = prepared_or_none(img)
            if p is not None:
                X.append(p); y.append(0)
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
    train_idx, val_idx = [], []
    for cls in np.unique(y):
        ci = idx[y == cls]
        rng.shuffle(ci)
        n_val = max(1, int(round(len(ci) * val_frac)))
        val_idx += list(ci[:n_val])
        train_idx += list(ci[n_val:])
    return np.array(train_idx), np.array(val_idx)


def split_by_photo(cells, holdout_photo):
    train = {}; val = {}
    for label, items in cells.items():
        train[label] = [p for p in items if p[0] != holdout_photo]
        val[label] = [p for p in items if p[0] == holdout_photo]
    return train, val


def decode_decode(cell, rec):
    """rec: dict with 'counter' (blob->1|2) and 'digit' (blob->0..9)."""
    b = prepare_input(cell)
    if b is None:
        return -1, 0.0, -1
    count = rec["counter"](b)
    if count == 2:
        w = cell.shape[1]; hw = w // 2
        left = cv2.resize(cell[:, :hw], (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        right = cv2.resize(cell[:, hw:], (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        lp = prepare_input(left); rp = prepare_input(right)
        l = rec["digit"](lp) if lp is not None else -1
        r = rec["digit"](rp) if rp is not None else -1
        if l >= 0 and r >= 0:
            return l * 10 + r, count, 2
        return -1, count, 2
    d = rec["digit"](b)
    return d, count, count


def main():
    cells = load_cells()

    # ---- digits model: random stratified split ----
    X, y = build_digit_samples(cells, use_augment=True)
    train_idx, val_idx = split_train_val(X, y)
    model, val_acc = train_model(X[train_idx], y[train_idx], 10, val=(X[val_idx], y[val_idx]))
    print(f"[digits] train={len(train_idx)} valid={len(val_idx)} valid_acc={val_acc:.4f}")

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
    md, _ = train_model(Xd, yd, 10, val=None, epochs=90)
    Xc, yc, _ = build_counter_samples(cells)
    cm2, _ = train_model(Xc, yc, 2, val=None, epochs=60)

    torch.onnx.export(md, torch.randn(1, 1, 28, 28), DIGITS_MODEL,
                      input_names=["Input3"], output_names=["Output"],
                      opset_version=9, dynamo=False)
    torch.onnx.export(cm2, torch.randn(1, 1, 28, 28), COUNTER_MODEL,
                      input_names=["Input3"], output_names=["Output"],
                      opset_version=9, dynamo=False)

    import onnx
    for p in [DIGITS_MODEL, COUNTER_MODEL]:
        m = onnx.load(p)
        # Convert pads to auto_pad=SAME_UPPER (matches original model format).
        for n in m.graph.node:
            if n.op_type == 'Conv':
                new_attrs = []
                for a in n.attribute:
                    if a.name == 'pads':
                        ap = onnx.AttributeProto()
                        ap.name = 'auto_pad'
                        ap.type = onnx.AttributeProto.STRING
                        ap.s = b'SAME_UPPER'
                        new_attrs.append(ap)
                    else:
                        new_attrs.append(a)
                del n.attribute[:]
                n.attribute.extend(new_attrs)
        # Convert Gemm -> MatMul + Add. Gemm(1,1,1) does alpha*A*B+beta*C.
        # With transB=1 (default in PyTorch linear): B is transposed.
        # So output = A @ B^T + C. For fc: x @ W^T + bias.
        # MatMul computes x @ W. We need W transposed in the initializer.
        new_nodes = []
        for n in m.graph.node:
            if n.op_type == 'Gemm' and len(n.input) == 3:
                matmul_out = n.output[0] + '_matmul'
                # Check if transB=1 (default for PyTorch linear)
                transB = any(a.name == 'transB' and a.i == 1 for a in n.attribute)
                if transB:
                    # Transpose the weight initializer: fc.weight is [out, in]
                    # Find it in initializers and transpose
                    for init in m.graph.initializer:
                        if init.name == n.input[1]:
                            w = onnx.numpy_helper.to_array(init)
                            w_t = w.T  # [out, in] -> [in, out]
                            m.graph.initializer.remove(init)
                            new_init = onnx.numpy_helper.from_array(w_t, init.name)
                            m.graph.initializer.append(new_init)
                            break
                mm = onnx.helper.make_node('MatMul', [n.input[0], n.input[1]], [matmul_out])
                add = onnx.helper.make_node('Add', [matmul_out, n.input[2]], n.output)
                new_nodes.extend([mm, add])
            else:
                new_nodes.append(n)
        m.graph.node.clear()
        m.graph.node.extend(new_nodes)
        m.ir_version = 3
        onnx.save(m, p)
        print(f"post-processed: {p} ir={m.ir_version} opset={[o.version for o in m.opset_import]}")
    print("exported", DIGITS_MODEL, COUNTER_MODEL)


if __name__ == "__main__":
    main()
