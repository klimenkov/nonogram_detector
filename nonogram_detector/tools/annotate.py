#!/usr/bin/env python3
"""Browser annotation of warped nonogram clue cells.

serve  <cells_dir> [port]   -> review page at http://127.0.0.1:port (default 8899)
export <cells_dir>          -> materialize labels into digits_marked/<value>/
"""
import csv
import glob
import http.server
import json
import os
import re
import shutil
import sys

EMPTY_LABELS = {"-1", "e", "-", ""}

CORPUS = os.environ.get(
    "NG_DIGITS_MARKED") or os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "digits_marked"))


def photo_strip_key(name):
    # new deterministic: <photo>_<top|left>_<rrrr>_<cccc>.png
    m = re.match(r"(.+?)_(top|left)_(\d{4})_(\d{4})\.png$", name)
    if m:
        return m.group(1), m.group(2), "%04d_%04d" % (int(m.group(3)), int(m.group(4)))
    # legacy sequential: <photo>_<top|left>_<pos>.png
    m2 = re.match(r"(.+?)_(top|left)_(\d+)\.png$", name)
    if m2:
        return m2.group(1), m2.group(2), "idx_%s" % m2.group(3)
    return None


def load_old_labels(cells_dir):
    """Return {strip: {row_col: value}} from digits_marked for this photo.

    Legacy sequential files ONLY align onto (row,col) when every index for a
    strip is present and unique in a contiguous 0..N-1 set (per-dump order).
    The existing corpus violates this (indices are per-dump-run counters that
    overlap across runs), so most photos degrade to predictions-only on the
    first pass; deterministic `<photo>_<strip>_<rrrr>_<cccc>.png` files align
    always."""
    out = {}
    index_path = os.path.join(cells_dir, "index.json")
    idx = json.load(open(index_path)) if os.path.isfile(index_path) else None
    photo = idx["photo"] if idx else None

    if photo is None:
        return out
    for label in os.listdir(CORPUS):
        d = os.path.join(CORPUS, label)
        if not os.path.isdir(d):
            continue
        value = int(label) if label.isdigit() else -1
        for name in os.listdir(d):
            info = photo_strip_key(name)
            if not info or info[0] != photo:
                continue
            strip, key = info[1], info[2]
            out.setdefault(strip, {})[key] = value

    # Align legacy sequential indices if present and complete per strip.
    if not idx:
        return out
    by_strip = {}
    for c in idx["cells"]:
        by_strip.setdefault(c["strip"], []).append((c["pos"], c["row"], c["col"]))
    for strip, cells in by_strip.items():
        old = out.get(strip, {})
        keys = ["idx_%04d" % i for i in range(len(cells))]
        if old and all(k in old for k in keys):
            for (pos, row, col) in cells:
                val = old.get("idx_%04d" % pos)
                if val is not None:
                    out.setdefault(strip, {})["%04d_%04d" % (row, col)] = val
    return out


def shown_value(c, old_by_strip, labels):
    """(value, is_human, source, verified) per current labels/old/pred state."""
    key = "%04d_%04d" % (c["row"], c["col"])
    saved = labels.get(c["strip"], {}).get(key)
    old = old_by_strip.get(c["strip"], {}).get(key)
    if saved == "confirm":
        return (saved_old_or_pred(c, old), True, "human", True)
    if saved == "pred":
        return (c["predicted"], False, "accepted-prediction", False)
    if saved is not None:  # explicit typed digit
        return (saved, True, "human", True)
    if old is not None:
        return (old, False, "carried-over", True)
    return (c["predicted"], False, "accepted-prediction", False)


def saved_old_or_pred(c, old):
    return old if old is not None else c["predicted"]


def make_meta_card(meta, cells_dir):
    has_image = False
    img_name = "detection.jpg"
    if cells_dir:
        for name in ["detection.jpg", "detection.png", "grid.png", "grid.jpg"]:
            if os.path.isfile(os.path.join(cells_dir, name)):
                has_image = True
                img_name = name
                break

    if not meta and not has_image:
        return ""

    img_html = ""
    if has_image:
        img_html = (
            '<div class="meta-preview">'
            '<a href="/%s" target="_blank" title="Click to view full image in new tab">'
            '<img src="/%s" class="meta-img" alt="Detected Crossings Overlay">'
            '</a>'
            '<div class="meta-caption">'
            'Overlay with detected grid crossings (click to open full-size).<br>'
            '<span style="color:#d32f2f;font-weight:bold">■</span> Main grid crossings &bull; '
            '<span style="color:#2e7d32;font-weight:bold">■</span> Top clues &bull; '
            '<span style="color:#0288d1;font-weight:bold">■</span> Left clues'
            '</div>'
            '</div>' % (img_name, img_name)
        )

    rows = []
    if meta and meta.get("image_width") and meta.get("image_height"):
        rows.append(("Image Dimensions", "%d &times; %d px" % (meta["image_width"], meta["image_height"])))
    if meta and "main_grid" in meta:
        mg = meta["main_grid"]
        rows.append(("Main Grid", "%d &times; %d cells (%d &times; %d crossings)" % (
            mg.get("cells_width", 0), mg.get("cells_height", 0),
            mg.get("crossings_cols", 0), mg.get("crossings_rows", 0))))
    if meta and "top_grid" in meta:
        tg = meta["top_grid"]
        rows.append(("Top Clues", "%d cols &times; %d rows (%d &times; %d crossings)" % (
            tg.get("cells_width", 0), tg.get("cells_height", 0),
            tg.get("crossings_cols", 0), tg.get("crossings_rows", 0))))
    if meta and "left_grid" in meta:
        lg = meta["left_grid"]
        rows.append(("Left Clues", "%d cols &times; %d rows (%d &times; %d crossings)" % (
            lg.get("cells_width", 0), lg.get("cells_height", 0),
            lg.get("crossings_cols", 0), lg.get("crossings_rows", 0))))

    params = meta.get("params", {}) if meta else {}
    param_rows = []
    if "resize_max" in params:
        param_rows.append(("resize_max", str(params["resize_max"])))
    if "threshold_block_size" in params:
        param_rows.append(("threshold_block_size", str(params["threshold_block_size"])))
    if "threshold_c" in params:
        param_rows.append(("threshold_c", str(params["threshold_c"])))
    if "similarity_ratio_min" in params:
        param_rows.append(("similarity_ratio_min", str(params["similarity_ratio_min"])))

    table_rows = ""
    if rows:
        table_rows += '<tr><th colspan="2">Grid Dimensions</th></tr>'
        for k, v in rows:
            table_rows += '<tr><td><b>%s</b></td><td>%s</td></tr>' % (k, v)
    if param_rows:
        table_rows += '<tr><th colspan="2">Detection Parameters</th></tr>'
        for k, v in param_rows:
            table_rows += '<tr><td><code>%s</code></td><td>%s</td></tr>' % (k, v)

    table_html = ""
    if table_rows:
        table_html = '<div class="meta-info"><table class="meta-table">%s</table></div>' % table_rows

    return (
        '<details class="meta-details" open>'
        '<summary class="meta-summary"><b>Detected Grid & Parameters</b> (click to expand/collapse)</summary>'
        '<div class="meta-content">%s%s</div>'
        '</details>' % (img_html, table_html)
    )


def make_page(photo, cells, old_by_strip, labels, meta=None):
    groups = {}
    for c in cells:
        key = "%04d_%04d" % (c["row"], c["col"])
        saved = labels.get(c["strip"], {}).get(key)
        old = old_by_strip.get(c["strip"], {}).get(key)
        pred = c["predicted"]
        conf = c.get("whole_conf", 0.0)
        if saved == "confirm":
            val = saved_old_or_pred(c, old)
        elif saved == "pred":
            val = pred
        elif saved is not None:
            val = saved
        elif old is not None:
            val = old
        else:
            val = pred
        groups.setdefault(val, []).append((c, val, saved, old, pred, conf))

    # Sort groups numerically: 1, 2, 3, ... then -1 (empty) at the end
    sorted_vals = sorted(groups.keys(), key=lambda v: (1 if v == -1 else 0, v))

    total_cnt = len(cells)
    top_cnt = sum(1 for c in cells if c.get("strip") == "top")
    left_cnt = sum(1 for c in cells if c.get("strip") == "left")
    saved_cnt = sum(1 for c in cells if labels.get(c.get("strip"), {}).get("%04d_%04d" % (c.get("row", 0), c.get("col", 0))) is not None)

    jump_links = ""
    for v in sorted_vals:
        label_text = "Empty (-1)" if v == -1 else str(v)
        cnt = len(groups[v])
        jump_links += ('<a class="jump-btn" href="#grp-%s">%s <small>(%d)</small></a> '
                       % (v, label_text, cnt))

    body_html = ""
    for v in sorted_vals:
        label_title = "Empty (-1)" if v == -1 else "Clue: %s" % v
        items = groups[v]
        # Sort cells within group: 'top' strip first, then 'left', ordered by row, col
        items.sort(key=lambda item: (0 if item[0]["strip"] == "top" else 1, item[0]["row"], item[0]["col"]))
        body_html += ('<div class="group-section" id="grp-%s">'
                      '<h2>%s <span class="cnt">(%d cells)</span></h2>'
                      '<div class="strip">' % (v, label_title, len(items)))
        for (c, val, saved, old, pred, conf) in items:
            val_text = "" if val == -1 else str(val)
            cls = "cell"
            if saved == "confirm" or isinstance(saved, int):
                cls += " saved"
            elif old is not None and old != pred:
                cls += " conflict"
            elif conf is not None and 0.0 < conf < 0.6:
                cls += " lowconf"
            strip_abbr = "T" if c["strip"] == "top" else "L"
            coord_str = "%s %d,%d" % (strip_abbr, c["row"], c["col"])
            body_html += ('<div class="%s" data-strip="%s" data-row="%d" data-col="%d" '
                          'data-png="%s" data-pred="%d" data-old="%s" data-shown="%s">'
                          '<img src="/cells/%s">'
                          '<span class="coord">%s</span>'
                          '<span class="lbl">%s</span></div>'
                          % (cls, c["strip"], c["row"], c["col"], c["png"],
                             pred, old if old is not None else "",
                             val_text, c["png"], coord_str, val_text))
        body_html += "</div></div>"
    badge_html = '<span class="total-badge">%d cells total (%d top, %d left)</span>' % (total_cnt, top_cnt, left_cnt)
    meta_html = make_meta_card(meta, CELLS_DIR)
    return PAGE_TMPL % {
        "photo": photo,
        "badge_html": badge_html,
        "meta_html": meta_html,
        "body_html": body_html,
        "jump_links": jump_links,
        "saved_cnt": saved_cnt,
        "total_cnt": total_cnt,
    }


PAGE_TMPL = """<!doctype html><html><head><meta charset="utf-8"><title>%(photo)s annotation</title>
<style>
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;padding:12px 18px 60px;background:#f8fafc;color:#1e293b}
 .top-header{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:8px}
 h1{margin:0;font-size:20px;font-weight:700;display:flex;align-items:center;gap:10px}
 .total-badge{font-size:13px;font-weight:normal;color:#475569;background:#e2e8f0;border:1px solid #cbd5e1;padding:2px 8px;border-radius:12px}
 .help-card{font-size:12px;color:#64748b;margin-bottom:10px;background:#fff;padding:8px 12px;border-radius:6px;border:1px solid #e2e8f0;line-height:1.5}
 .help-card b{color:#334155}
 .help-card .tag-red{color:#dc2626;font-weight:600}
 .help-card .tag-amber{color:#d97706;font-weight:600}
 .help-card .tag-green{color:#16a34a;font-weight:600}

 /* STICKY CONTROL PANEL AT TOP */
 .sticky-panel{position:sticky;top:0;z-index:1000;background:rgba(255,255,255,.98);backdrop-filter:blur(8px);border-bottom:2px solid #0284c7;box-shadow:0 4px 14px rgba(0,0,0,.08);margin:0 -18px 14px;padding:8px 18px}
 .sticky-main{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
 .cell-preview-card{display:flex;align-items:center;gap:8px;background:#f1f5f9;border:1px solid #cbd5e1;border-radius:6px;padding:3px 8px}
 .preview-thumb{width:46px;height:46px;border-radius:4px;border:2px solid #0284c7;background:#fff;image-rendering:pixelated;display:block}
 .preview-info{display:flex;flex-direction:column;gap:1px}
 .coord-badge{font-size:12px;font-weight:700;color:#0f172a}
 .sub-info{font-size:11px;color:#64748b}
 .input-group{display:flex;align-items:center;gap:6px}
 .main-input{font-size:22px;font-weight:700;width:5ch;text-align:center;padding:3px 6px;border:2px solid #0284c7;border-radius:6px;outline:none;background:#fff;color:#0f172a;transition:box-shadow .15s}
 .main-input:focus{box-shadow:0 0 0 3px rgba(2,132,199,.35)}
 .btn{padding:6px 11px;font-size:13px;font-weight:600;border-radius:5px;border:1px solid transparent;cursor:pointer;transition:all .1s ease;display:inline-flex;align-items:center;gap:4px}
 .btn-save{background:#0284c7;color:#fff}
 .btn-save:hover{background:#0369a1}
 .btn-confirm{background:#16a34a;color:#fff}
 .btn-confirm:hover{background:#15803d}
 .btn-skip{background:#e2e8f0;color:#334155;border-color:#cbd5e1}
 .btn-skip:hover{background:#cbd5e1}
 .quick-chips{display:flex;align-items:center;gap:3px;flex-wrap:wrap}
 .chip{padding:4px 7px;font-size:12px;font-weight:700;border-radius:4px;border:1px solid #cbd5e1;background:#fff;color:#334155;cursor:pointer}
 .chip:hover{background:#0284c7;color:#fff;border-color:#0284c7}
 .chip-empty{background:#f1f5f9;color:#64748b}
 .chip-empty:hover{background:#64748b;color:#fff}
 .sticky-actions{margin-left:auto;display:flex;align-items:center;gap:8px}
 .stat-pill{padding:4px 8px;background:#e2e8f0;border-radius:5px;font-size:12px;color:#334155;border:1px solid #cbd5e1;font-weight:600}
 .btn-confirm-all{background:#f8fafc;color:#475569;border:1px solid #cbd5e1;padding:5px 9px;font-size:12px}
 .btn-confirm-all:hover{background:#e2e8f0;color:#0f172a}
 .jump-row{display:flex;align-items:center;flex-wrap:wrap;gap:4px;margin-top:6px;padding-top:5px;border-top:1px solid #e2e8f0;font-size:11px}
 .jump-label{font-weight:700;color:#475569;margin-right:4px}
 .jump-btn{padding:2px 6px;border:1px solid #cbd5e1;border-radius:4px;text-decoration:none;font-size:11px;color:#334155;background:#fff}
 .jump-btn:hover{background:#0284c7;color:#fff;border-color:#0284c7}
 .jump-btn small{opacity:.75}

 /* META CARD */
 .meta-details{margin:8px 0 12px;background:#fff;border:1px solid #d0d7de;border-radius:6px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.05)}
 .meta-summary{padding:8px 12px;background:#f6f8fa;cursor:pointer;font-size:13px;color:#24292f;border-bottom:1px solid #d0d7de;user-select:none;font-weight:600}
 .meta-summary:hover{background:#edf2f7}
 .meta-content{display:flex;flex-wrap:wrap;gap:16px;padding:12px;align-items:flex-start}
 .meta-preview{flex:0 0 auto;max-width:380px}
 .meta-img{width:100%%;max-width:380px;height:auto;border:1px solid #ccc;border-radius:4px;display:block;box-shadow:0 2px 5px rgba(0,0,0,.1)}
 .meta-caption{font-size:11px;color:#555;margin-top:4px;line-height:1.4}
 .meta-info{flex:1 1 300px;min-width:260px}
 .meta-table{width:100%%;border-collapse:collapse;font-size:12px}
 .meta-table th{background:#f1f5f9;text-align:left;padding:5px 8px;border:1px solid #cbd5e0;color:#334155;font-size:11px;text-transform:uppercase}
 .meta-table td{padding:4px 8px;border:1px solid #e2e8f0;color:#1e293b}
 .meta-table tr:nth-child(even){background:#f8fafc}

 /* CELLS GRID */
 h2{margin:16px 0 6px;font-size:16px;border-bottom:2px solid #e2e8f0;padding-bottom:4px;color:#334155}
 .cnt{font-size:13px;font-weight:normal;color:#64748b}
 .strip{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 14px}
 .cell{position:relative;border:3px solid #cbd5e1;cursor:pointer;border-radius:5px;background:#fff;transition:transform .1s,box-shadow .1s;scroll-margin-top:140px;scroll-margin-bottom:20px}
 .cell:hover{transform:scale(1.08);z-index:2}
 .cell img{width:64px;height:64px;display:block;image-rendering:pixelated;border-radius:2px}
 .cell .lbl{position:absolute;bottom:0;right:0;background:rgba(15,23,42,.85);color:#fff;font-size:13px;font-weight:700;padding:0 4px;border-radius:3px 0 3px 0}
 .cell .coord{position:absolute;top:0;left:0;background:rgba(255,255,255,.9);color:#334155;font-size:10px;padding:0 2px;border-radius:3px 0 3px 0;font-weight:600}
 .cell.conflict{border-color:#ef4444;box-shadow:0 0 0 2px #ef4444}
 .cell.lowconf{border-color:#f59e0b}
 .cell.saved{border-color:#22c55e}
 .cell.focused{border-color:#0284c7 !important;box-shadow:0 0 0 4px rgba(2,132,199,.5) !important;z-index:5}
</style></head><body>
<div class="top-header">
 <h1>%(photo)s %(badge_html)s</h1>
</div>
<div class="help-card">
 <b>Keyboard shortcuts:</b> Type digits <kbd>1</kbd>..<kbd>9</kbd> or <kbd>-</kbd> directly (always focused) &rarr; <kbd>Enter</kbd> to save & next.
 <kbd>Space</kbd> = confirm current/prediction & next. <kbd>Esc</kbd> / <kbd>Tab</kbd> = skip to next. <kbd>Shift+Tab</kbd> = previous.
 Legend: <span class="tag-red">Red</span> = old label disagrees. <span class="tag-amber">Amber</span> = low model confidence. <span class="tag-green">Green</span> = saved.
</div>
<div class="sticky-panel">
 <div class="sticky-main">
  <div class="cell-preview-card" title="Selected cell">
   <img id="cur-img" class="preview-thumb" src="" alt="cell" />
   <div class="preview-info">
    <div class="coord-badge"><span id="cur-strip">TOP</span> <span id="cur-coord">0, 0</span></div>
    <div class="sub-info">pred: <b id="cur-pred">-</b> | cur: <b id="cur-shown">-</b></div>
   </div>
  </div>
  <div class="input-group">
   <input id="inp" class="main-input" placeholder="val" autofocus autocomplete="off" />
   <button type="button" class="btn btn-save" id="btn-save" title="Save entered number and go next (Enter)">⏎ Save</button>
   <button type="button" class="btn btn-confirm" id="btn-confirm" title="Confirm current/predicted value (Space)">✓ Confirm</button>
   <button type="button" class="btn btn-skip" id="btn-skip" title="Skip to next without saving (Esc / Tab)">Skip</button>
  </div>
  <div class="quick-chips">
   <button type="button" class="chip" data-val="1">1</button>
   <button type="button" class="chip" data-val="2">2</button>
   <button type="button" class="chip" data-val="3">3</button>
   <button type="button" class="chip" data-val="4">4</button>
   <button type="button" class="chip" data-val="5">5</button>
   <button type="button" class="chip" data-val="6">6</button>
   <button type="button" class="chip" data-val="7">7</button>
   <button type="button" class="chip" data-val="8">8</button>
   <button type="button" class="chip" data-val="9">9</button>
   <button type="button" class="chip chip-empty" data-val="-1">Empty (-)</button>
  </div>
  <div class="sticky-actions">
   <span class="stat-pill">Saved: <b id="saved-cnt">%(saved_cnt)d</b> / %(total_cnt)d</span>
   <button type="button" class="btn btn-confirm-all" onclick="confirmAll()">Confirm remaining</button>
  </div>
 </div>
 <div class="jump-row">
  <span class="jump-label">Jump to clue:</span>
  %(jump_links)s
 </div>
</div>
%(meta_html)s
<div class="strip-container">%(body_html)s</div>
<script>
let cells = Array.from(document.querySelectorAll('.cell'));
let cur = 0;

function focusCell(i) {
 if (cells.length === 0) return;
 if (i < 0) i = 0;
 if (i >= cells.length) i = cells.length - 1;
 if (cells[cur]) cells[cur].classList.remove('focused');
 cur = i;
 let el = cells[i];
 el.classList.add('focused');

 let strip = el.dataset.strip;
 let row = el.dataset.row;
 let col = el.dataset.col;
 let png = el.dataset.png;
 let pred = el.dataset.pred;
 let old = el.dataset.old;
 let shown = el.dataset.shown;
 let v = shown !== '' ? shown : (old !== '' ? old : pred);

 document.getElementById('cur-img').src = '/cells/' + png;
 document.getElementById('cur-strip').textContent = strip.toUpperCase();
 document.getElementById('cur-coord').textContent = row + ', ' + col;
 document.getElementById('cur-pred').textContent = pred === '-1' ? 'empty' : pred;
 document.getElementById('cur-shown').textContent = v === '-1' ? 'empty' : v;

 let inp = document.getElementById('inp');
 inp.value = (v === '-1' || v === '') ? '' : v;
 inp.focus();
 inp.select();

 el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

cells.forEach((el, i) => {
 el.addEventListener('click', () => focusCell(i));
});

function saveValue(v, confirmOnly) {
 let el = cells[cur];
 if (!el) return;
 let body = { strip: el.dataset.strip, row: +el.dataset.row, col: +el.dataset.col, value: v };
 if (!el.classList.contains('saved')) {
  let sc = document.getElementById('saved-cnt');
  if (sc) sc.textContent = parseInt(sc.textContent || '0', 10) + 1;
 }
 if (confirmOnly) {
  fetch('/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...body, confirm: true }) });
 } else {
  fetch('/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
 }
 let finalVal = confirmOnly ? (el.dataset.old !== '' ? el.dataset.old : el.dataset.pred) : String(v);
 el.dataset.shown = finalVal;
 el.classList.add('saved');
 el.querySelector('.lbl').textContent = finalVal === '-1' ? '' : finalVal;
 cur = (cur + 1) %% cells.length;
 focusCell(cur);
}

document.getElementById('btn-save').addEventListener('click', () => {
 let inp = document.getElementById('inp');
 let t = inp.value.trim();
 saveValue(t !== '' ? t : '-1', false);
});

document.getElementById('btn-confirm').addEventListener('click', () => {
 saveValue(null, true);
});

document.getElementById('btn-skip').addEventListener('click', () => {
 focusCell((cur + 1) %% cells.length);
});

document.querySelectorAll('.chip').forEach(btn => {
 btn.addEventListener('click', e => {
  e.preventDefault();
  saveValue(btn.dataset.val, false);
 });
});

window.addEventListener('keydown', e => {
 if (e.altKey || e.ctrlKey || e.metaKey) return;
 let inp = document.getElementById('inp');
 let isInputFocused = (document.activeElement === inp);

 if (e.key === 'Enter') {
  e.preventDefault();
  let t = inp.value.trim();
  saveValue(t !== '' ? t : '-1', false);
 } else if (e.key === ' ') {
  e.preventDefault();
  saveValue(null, true);
 } else if (e.key === 'Escape') {
  e.preventDefault();
  focusCell((cur + 1) %% cells.length);
 } else if (e.key === 'Tab') {
  e.preventDefault();
  if (e.shiftKey) {
   focusCell((cur - 1 + cells.length) %% cells.length);
  } else {
   focusCell((cur + 1) %% cells.length);
  }
 } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
  if (!isInputFocused) {
   e.preventDefault();
   focusCell((cur + 1) %% cells.length);
  }
 } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
  if (!isInputFocused) {
   e.preventDefault();
   focusCell((cur - 1 + cells.length) %% cells.length);
  }
 } else if (/^[0-9\-]$/.test(e.key)) {
  if (!isInputFocused) {
   inp.focus();
   inp.value = e.key;
   e.preventDefault();
  }
 }
});

function confirmAll() {
 if (confirm('Confirm all remaining unverified cells as model predictions?')) {
  fetch('/confirm_all', { method: 'POST' }).then(() => location.reload());
 }
}

focusCell(0);
</script>
</body></html>"""


def save_labels(cells_dir, labels):
    with open(os.path.join(cells_dir, "labels.json"), "w") as f:
        json.dump(labels, f, indent=1)


def load_labels(cells_dir):
    p = os.path.join(cells_dir, "labels.json")
    return json.load(open(p)) if os.path.isfile(p) else {}


def load_index(cells_dir):
    idx_path = os.path.join(cells_dir, "index.json")
    if not os.path.isfile(idx_path):
        return os.path.basename(cells_dir), [], {}
    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    return idx.get("photo", os.path.basename(cells_dir)), idx.get("cells", []), idx.get("meta", {})


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json_ok(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            photo, cells, meta = load_index(CELLS_DIR)
            labels = load_labels(CELLS_DIR)
            old_labels = load_old_labels(CELLS_DIR)
            page = make_page(photo, cells, old_labels, labels, meta)
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/detection.jpg", "/detection.png", "/grid.png", "/grid.jpg"):
            filename = self.path.lstrip("/")
            img_path = os.path.join(CELLS_DIR, filename)
            if os.path.isfile(img_path):
                ctype = "image/png" if filename.endswith(".png") else "image/jpeg"
                data = open(img_path, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        m = re.match(r"^/cells/([^/]+\.png)$", self.path)
        if m:
            png = os.path.join(CELLS_DIR, m.group(1))
            png = os.path.normpath(png)
            if os.path.commonpath([os.path.normpath(CELLS_DIR), png]) != os.path.normpath(CELLS_DIR):
                self.send_error(403)
                return
            if os.path.isfile(png):
                data = open(png, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        self.send_error(404)

    def do_POST(self):
        ln = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(ln) if ln else b"{}"
        data = json.loads(raw)
        if self.path == "/save":
            strip, row, col = data["strip"], data["row"], data["col"]
            key = "%04d_%04d" % (row, col)
            if data.get("confirm"):
                LABELS.setdefault(strip, {})[key] = "confirm"
            else:
                v = normalize(data.get("value"))
                if v is not None:
                    LABELS.setdefault(strip, {})[key] = v
            save_labels(CELLS_DIR, LABELS)
            self._json_ok({"ok": True})
        elif self.path == "/confirm_all":
            # Keep remaining cells as accepted-prediction via a persistent
            # "pred" sentinel (so it survives a serve->export process restart);
            # export turns "pred" into (predicted, accepted-prediction, False).
            # Cells carrying an old corpus label are left alone: export carries
            # them over as verified human labels.
            for c in CELLS[1]:
                key = "%04d_%04d" % (c["row"], c["col"])
                stash = LABELS.setdefault(c["strip"], {})
                if key not in stash and OLD_LABELS.get(c["strip"], {}).get(key) is None:
                    stash[key] = "pred"
            save_labels(CELLS_DIR, LABELS)
            self._json_ok({"ok": True})
        else:
            self.send_error(404)


def normalize(text):
    t = (text or "").strip()
    if t.lower() in EMPTY_LABELS:
        return -1
    if t == "":
        return None
    if t.isdigit():
        iv = int(t)
        if iv >= 10 or iv in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            return iv
    return None


CELLS_DIR = None
CELLS = (None, [])
META = {}
OLD_LABELS = {}
LABELS = {}


def serve(cells_dir, port):
    global CELLS_DIR, CELLS, META, OLD_LABELS, LABELS
    CELLS_DIR = cells_dir
    photo, cells, meta = load_index(cells_dir)
    CELLS = (photo, cells)
    META = meta
    OLD_LABELS = load_old_labels(cells_dir)
    LABELS = load_labels(cells_dir)
    srv = http.server.HTTPServer(("127.0.0.1", port), Handler)
    print("review:", photo, "cells:", len(cells), "http://127.0.0.1:%d" % port)
    srv.serve_forever()


def export(cells_dir):
    photo, cells, _ = load_index(cells_dir)
    labels = load_labels(cells_dir)
    old = load_old_labels(cells_dir)
    rows = []
    for c in cells:
        key = "%04d_%04d" % (c["row"], c["col"])
        saved = labels.get(c["strip"], {}).get(key)
        old_val = old.get(c["strip"], {}).get(key)
        if saved == "confirm":
            val = saved_old_or_pred(c, old_val)
            source, verified = "human", True
        elif saved == "pred":
            val, source, verified = c["predicted"], "accepted-prediction", False
        elif saved is not None:
            val, source, verified = saved, "human", True
        elif old_val is not None:
            val, source, verified = old_val, "carried-over", True
        else:
            continue
        rows.append({"photo": photo, "strip": c["strip"], "pos": c["pos"],
                     "row": c["row"], "col": c["col"], "value": val,
                     "source": source, "verified": verified})
    # materialize
    for r in rows:
        label_dir = os.path.join(CORPUS, str(r["value"]) if r["value"] >= 0 else "-1")
        os.makedirs(label_dir, exist_ok=True)
        src = os.path.join(cells_dir, "%s_%s_%04d_%04d.png" % (
            photo, r["strip"], r["row"], r["col"]))
        dst = os.path.join(label_dir, "%s_%s_%04d_%04d.png" % (
            photo, r["strip"], r["row"], r["col"]))
        shutil.copyfile(src, dst)
    # remove that photo's cells from label folders whose label no longer matches
    # (keyed by resolved value, so a relabeled cell purges its stale copy) and
    # cells that left the manifest entirely (orphans, incl. legacy idx files)
    keep = {}
    for r in rows:
        keep[(r["strip"], "%04d_%04d" % (r["row"], r["col"]))] = \
            str(r["value"]) if r["value"] >= 0 else "-1"
    for label in os.listdir(CORPUS):
        d = os.path.join(CORPUS, label)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            info = photo_strip_key(name)
            if not info or info[0] != photo:
                continue
            if keep.get((info[1], info[2])) != label:
                os.remove(os.path.join(d, name))
    with open(os.path.join(cells_dir, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["photo", "strip", "pos", "row", "col",
                                          "value", "source", "verified"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("exported", len(rows), "cells; manifest ->",
          os.path.join(cells_dir, "manifest.csv"))


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("serve", "export"):
        print(__doc__)
        sys.exit(1)
    cmd, cells_dir = sys.argv[1], sys.argv[2]
    if not os.path.isdir(cells_dir):
        print("no such dir:", cells_dir)
        sys.exit(1)
    if cmd == "serve":
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8899
        serve(cells_dir, port)
    else:
        export(cells_dir)


if __name__ == "__main__":
    main()
