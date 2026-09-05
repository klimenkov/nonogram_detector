#!/usr/bin/env python3
"""Render detection / recognition / solving results for nonogram photos.

For each photo in nonograms/*.jpg this reads the pipeline artifacts left by
the C++ application (grid.png detection overlay, solved_overlay.png, and the
puzzle.json clue dump) and produces:

  rendered/<name>/detection.png    photo + detected grid overlay (whole image)
  rendered/<name>/recognition.png  clean board with the decoded clue digits
  rendered/<name>/solving.png      tribunal board with the solved fill (or the
                                   solver failure message)
  rendered/<name>/result.png       3-panel collage: detection | recognition | solving
  rendered/index.html              contact sheet linking every result

Usage: render_results.py <results_dir> [nonograms_dir]
  <results_dir>    directory containing one subdir per photo with run.txt,
                   grid.png, solved_overlay.png and puzzle.json (the C++ app
                   artifacts, e.g. /tmp/opencode/nonogram_results).
  [nonograms_dir]  directory with the source photos (default: nonograms/
                   next to the results dir).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CELL = 34
FILLED = (28, 28, 38)
GRID_LINE = (150, 150, 155)
PLAY_W = (252, 252, 255)
CLUE_TXT = (30, 30, 40)
TITLE_TXT = (20, 20, 25)
PAD = 10
PANEL_H = 760

FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _clue_box(draw, x, y, size, text, font):
    x0, y0 = x * size, y * size
    draw.rectangle([x0, y0, x0 + size, y0 + size], outline=GRID_LINE, width=1)
    box = draw.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    draw.text((x0 + (size - tw) / 2 - box[0], y0 + (size - th) / 2 - box[1]),
              text, fill=CLUE_TXT, font=font)


def render_board(width, height, rows, columns, goal=None, message=None):
    """Clean nonogram board with clue digits; optionally fill the solution."""
    left_w = 0
    for line in rows:
        left_w = max(left_w, sum(len(str(v)) for v in line))
    top_h = 0
    for line in columns:
        top_h = max(top_h, sum(len(str(v)) for v in line))

    size = CELL
    digit_font = load_font(int(size * 0.62))
    W, H = left_w + width, top_h + height

    if message:
        msg_font = load_font(30)
        msg_line = MathText(message, msg_font, W * size)
        board = Image.new("RGB", (W * size, H * size + msg_line.h + 2 * PAD), PLAY_W)
    else:
        board = Image.new("RGB", (W * size, H * size), PLAY_W)
    draw = ImageDraw.Draw(board)

    gx0, gy0 = left_w * size, top_h * size

    for r in range(height):
        cols = rows[r]
        total = sum(len(str(c)) for c in cols)
        x = left_w + width - total
        col_i = 0
        for c in cols:
            for ch in str(c):
                _clue_box(draw, x, top_h + r, size, ch, digit_font)
                x += 1

    for c in range(width):
        line = columns[c]
        total = sum(len(str(v)) for v in line)
        y = top_h + height - total
        for v in line:
            for ch in str(v):
                _clue_box(draw, left_w + c, y, size, ch, digit_font)
                y += 1

    if goal:
        for r in range(height):
            for c in range(width):
                if goal[r * width + c] == "1":
                    x0, y0 = (left_w + c) * size, (top_h + r) * size
                    draw.rectangle([x0, y0, x0 + size, y0 + size], fill=FILLED)

    for r in range(height + 1):
        y0 = (top_h + r) * size
        draw.line([gx0, y0, gx0 + width * size, y0], fill=GRID_LINE, width=1)
    for c in range(width + 1):
        x0 = (left_w + c) * size
        draw.line([x0, gy0, x0, gy0 + height * size], fill=GRID_LINE, width=1)

    if message:
        draw = ImageDraw.Draw(board, "RGBA")
        msg_line.draw(draw, PAD, H * size + PAD, (190, 40, 40, 255))

    return board


class MathText:
    """Simple measured/wrapped text block (ASCII) used for solver messages."""

    def __init__(self, text, font, max_w):
        self.font = font
        words = text.split()
        lines, cur = [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if font.getbbox(trial)[2] <= max_w:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        self.lines = lines
        s = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        self.h = sum(s.textbbox((0, 0), ln, font=font)[3] for ln in lines)

    def draw(self, draw, x, y, color):
        for ln in self.lines:
            box = draw.textbbox((0, 0), ln, font=self.font)
            draw.text((x - box[0], y - box[1]), ln, fill=color, font=self.font)
            y += box[3]


def solve_outcome_text(is_solved, message):
    if is_solved:
        return "SOLVED"
    if not message:
        return "NOT SOLVED"
    return "NOT SOLVED: " + message


def _panel_detection(photo_dir, photo_path):
    return photo_dir / "grid.png"


def _panels_for_photo(photo_dir, photo_path):
    name = photo_dir.name
    puzzle_path = photo_dir / "puzzle.json"
    grid_img = _panel_detection(photo_dir, photo_path)
    overlay_img = photo_dir / "solved_overlay.png"

    if not puzzle_path.exists() or not grid_img.exists():
        return None

    data = json.loads(puzzle_path.read_text())
    width, height = data["width"], data["height"]
    rows, columns = data["rows"], data["columns"]
    goal = data.get("goal")
    message = None
    if not data.get("solved"):
        out = (photo_dir / "run.txt").read_text(errors="replace")
        sol_lines = [l for l in out.splitlines() if l.startswith("solver:")]
        message = sol_lines[0][len("solver: "):] if sol_lines else "unsolvable from decoded clues"

    recognition = render_board(width, height, rows, columns)
    solving = render_board(width, height, rows, columns, goal=goal,
                           message=None if data.get("solved") else message)

    return {
        "photo": Image.open(photo_path).convert("RGB"),
        "detection": Image.open(grid_img).convert("RGB"),
        "recognition": recognition,
        "solving": solving,
        "status": solve_outcome_text(bool(goal), message),
    }


def _scale_to_height(img, h):
    w = max(1, round(img.width * h / img.height))
    return img.resize((w, h), Image.LANCZOS)


def _make_collage(panels, name):
    panels2 = []
    for img in panels:
        img = _scale_to_height(img, PANEL_H)
        canvas = Image.new("RGB", (img.width, img.height + 46), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), name, fill=TITLE_TXT, font=load_font(26))
        canvas.paste(img, (0, 46))
        panels2.append(canvas)
    collage = Image.new("RGB", (sum(p.width for p in panels2), PANEL_H + 46), (255, 255, 255))
    x = 0
    for p in panels2:
        collage.paste(p, (x, 0))
        x += p.width
    return collage


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    results = Path(sys.argv[1])
    nonograms_dir = Path(sys.argv[2]) if len(sys.argv) == 3 else Path("nonograms")
    out_root = results / "rendered"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for photo_dir in sorted(p for p in results.iterdir() if p.is_dir()):
        if photo_dir.name == "rendered":
            continue
        name = photo_dir.name
        photo_path = nonograms_dir / (name + ".jpg")
        if not photo_path.exists():
            summary.append((name, "no source photo"))
            continue

        staged = False
        if "grid.png" in {p.name for p in photo_dir.iterdir()} or (photo_dir / "puzzle.json").exists():
            panels = _panels_for_photo(photo_dir, photo_path)
            if panels:
                out = out_root / name
                out.mkdir(parents=True, exist_ok=True)
                for key, img in (("detection", panels["detection"]),
                                 ("recognition", panels["recognition"]),
                                 ("solving", panels["solving"])):
                    img.save(out / f"{key}.png")
                _make_collage([
                    panels["detection"], panels["recognition"], panels["solving"]
                ], name).save(out / "result.png")
                (out / "status.txt").write_text(panels["status"] + "\n")
                staged = True

        if not staged:
            photo = Image.open(photo_path).convert("RGB")
            photo = _scale_to_height(photo, PANEL_H)
            canvas = Image.new("RGB", (photo.width, photo.height + 46), (255, 255, 255))
            d = ImageDraw.Draw(canvas)
            d.text((8, 6), name, fill=TITLE_TXT, font=load_font(26))
            canvas.paste(photo, (0, 46))
            d = ImageDraw.Draw(canvas, "RGBA")
            txt = "DETECTION FAILED - grid not found"
            f = load_font(34)
            box = d.textbbox((0, 0), txt, font=f)
            d.rectangle([10, 60, 20 + box[2], 60 + box[3]], fill=(255, 255, 255, 220))
            d.text((16 - box[0], 62 - box[1]), txt, fill=(200, 30, 30, 255), font=f)
            d = ImageDraw.Draw(canvas)
            out = out_root / name
            out.mkdir(parents=True, exist_ok=True)
            canvas.save(out / "result.png")
            (out / "status.txt").write_text("DETECTION FAILED\n")

        summary.append((name, (out_root / name / "status.txt").read_text().strip()))

    rel = out_root.relative_to(Path.cwd()) if our_path_under_cwd(out_root) else out_root
    html = ["<!doctype html><html><head><meta charset='utf-8'>",
            "<title>Nonogram pipeline results</title></head><body><h1>Pipeline results</h1>",
            "<table border=1 cellpadding=6><tr><th>photo</th><th>status</th><th>result</th></tr>"]
    for name, status in summary:
        img = f"{name}/result.png"
        html.append(f"<tr><td>{name}</td><td>{status}</td>"
                    f"<td><a href='{img}'><img src='{img}' width='1100'></a></td></tr>")
    html.append("</table></body></html>")
    (out_root / "index.html").write_text("\n".join(html))

    print("rendered into", out_root)
    for name, status in summary:
        print(f"  {name}: {status}")
    return 0


def our_path_under_cwd(p: Path) -> bool:
    try:
        p.resolve().relative_to(Path.cwd().resolve())
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    sys.exit(main())