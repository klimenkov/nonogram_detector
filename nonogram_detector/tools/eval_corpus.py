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