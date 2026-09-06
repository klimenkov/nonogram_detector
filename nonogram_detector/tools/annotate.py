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
    has_image = os.path.isfile(os.path.join(cells_dir, "detection.jpg")) if cells_dir else False
    if not meta and not has_image:
        return ""

    img_html = ""
    if has_image:
        img_html = (
            '<div class="meta-preview">'
            '<a href="/detection.jpg" target="_blank" title="Click to view full image in new tab">'
            '<img src="/detection.jpg" class="meta-img" alt="Detected Crossings Overlay">'
            '</a>'
            '<div class="meta-caption">'
            'Overlay with detected grid crossings (click to open full-size).<br>'
            '<span style="color:#d32f2f;font-weight:bold">■</span> Main grid crossings &bull; '
            '<span style="color:#2e7d32;font-weight:bold">■</span> Top clues &bull; '
            '<span style="color:#0288d1;font-weight:bold">■</span> Left clues'
            '</div>'
            '</div>'
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

    # Quick Jump bar at the top
    jump_bar = (
        '<div class="jump-bar">'
        '<span class="stat-pill"><b>Total cells:</b> %d (Top: %d, Left: %d)</span>'
        '<span class="stat-pill"><b>Saved:</b> <span id="saved-cnt">%d</span> / %d</span>'
        '<span style="font-weight:bold;margin:0 6px 0 10px">Jump to clue:</span>'
    ) % (total_cnt, top_cnt, left_cnt, saved_cnt, total_cnt)
    for v in sorted_vals:
        label_text = "Empty (-1)" if v == -1 else str(v)
        cnt = len(groups[v])
        jump_bar += ('<a class="jump-btn" href="#grp-%s">%s <small>(%d)</small></a> '
                     % (v, label_text, cnt))
    jump_bar += '</div>'

    body_html = jump_bar
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
    return PAGE_TMPL % (photo, photo, badge_html, meta_html, body_html)


PAGE_TMPL = """<!doctype html><html><head><meta charset="utf-8"><title>%s annotation</title>
<style>
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:16px;background:#fafafa;color:#222}
 h1{margin:0 0 10px;font-size:24px;display:flex;align-items:center;flex-wrap:wrap;gap:8px}
 h2{margin:18px 0 6px;font-size:18px;border-bottom:2px solid #ddd;padding-bottom:4px}
 .cnt{font-size:14px;font-weight:normal;color:#666}
 .strip{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 16px}
 .cell{position:relative;border:3px solid #bdbdbd;cursor:pointer;border-radius:4px;background:#fff;transition:transform 0.1s}
 .cell:hover{transform:scale(1.08);z-index:2}
 .cell img{width:64px;height:64px;display:block;image-rendering:pixelated}
 .cell .lbl{position:absolute;bottom:0;right:0;background:rgba(0,0,0,.75);color:#fff;font-size:13px;font-weight:bold;padding:0 4px;border-radius:2px 0 0 0}
 .cell .coord{position:absolute;top:0;left:0;background:rgba(255,255,255,.85);color:#333;font-size:10px;padding:0 2px;border-radius:0 0 2px 0}
 .cell.conflict{border-color:#e33;box-shadow:0 0 0 2px #e33}
 .cell.lowconf{border-color:#e90}
 .cell.saved{border-color:#2a2}
 .cell.focused{border-color:#007acc !important;box-shadow:0 0 0 3px #007acc !important;z-index:5}
 .jump-bar{position:sticky;top:0;background:rgba(250,250,250,.96);backdrop-filter:blur(4px);z-index:100;padding:10px 0;border-bottom:1px solid #ccc;margin-bottom:14px;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
 .jump-btn{padding:3px 8px;border:1px solid #bbb;border-radius:4px;text-decoration:none;font-size:12px;color:#222;background:#fff}
 .jump-btn:hover{background:#007acc;color:#fff;border-color:#007acc}
 .jump-btn small{font-size:10px;opacity:0.8}
 .stat-pill{padding:3px 9px;background:#e9edf2;border-radius:4px;font-size:12px;color:#2d3748;border:1px solid #cbd5e0;display:inline-flex;align-items:center;gap:4px}
 .total-badge{font-size:14px;font-weight:normal;color:#4a5568;background:#edf2f7;border:1px solid #cbd5e0;padding:3px 10px;border-radius:12px}
 .meta-details{margin:12px 0 16px;background:#fff;border:1px solid #d0d7de;border-radius:6px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.05)}
 .meta-summary{padding:10px 14px;background:#f6f8fa;cursor:pointer;font-size:14px;color:#24292f;border-bottom:1px solid #d0d7de;user-select:none;font-weight:600}
 .meta-summary:hover{background:#edf2f7}
 .meta-content{display:flex;flex-wrap:wrap;gap:20px;padding:14px;align-items:flex-start}
 .meta-preview{flex:0 0 auto;max-width:380px}
 .meta-img{width:100%%;max-width:380px;height:auto;border:1px solid #ccc;border-radius:4px;display:block;box-shadow:0 2px 5px rgba(0,0,0,0.1);transition:transform 0.15s}
 .meta-img:hover{transform:scale(1.02)}
 .meta-caption{font-size:11px;color:#555;margin-top:6px;line-height:1.5}
 .meta-info{flex:1 1 320px;min-width:280px}
 .meta-table{width:100%%;border-collapse:collapse;font-size:13px}
 .meta-table th{background:#f1f5f9;text-align:left;padding:6px 10px;border:1px solid #cbd5e0;color:#334155;font-size:11px;text-transform:uppercase;letter-spacing:0.5px}
 .meta-table td{padding:5px 10px;border:1px solid #e2e8f0;color:#1e293b}
 .meta-table tr:nth-child(even){background:#f8fafc}
 .bar{margin:14px 0} input{font-size:22px;width:9ch;padding:2px 6px}
 .help{color:#555;font-size:13px}
 button{padding:6px 12px;font-size:14px;cursor:pointer;border-radius:4px;border:1px solid #888;background:#eee}
 button:hover{background:#ddd}
</style></head><body>
<h1>%s %s</h1>
<div class="help">Grouped by clue value and sorted. Click any cell to edit. Type 1..9 / 10+ / '-' for empty, Enter saves+next, Space confirms shown label (human), Esc skips.
 Red = old label disagrees with model. Amber = low whole-cell confidence. Green = saved by you.</div>
<div style="margin:8px 0"><button onclick="confirmAll()">keep remaining as model prediction (unverified)</button></div>
%s
<div class="bar"><input id="inp" placeholder="value" autofocus><span class="help"> current cell: <b id="cur">-</b></span></div>
<div class="strip-container">%s</div>
<script>
let cells=Array.from(document.querySelectorAll('.cell'));
let cur=0;
let curInfo;
function focusCell(i){
 if(cells[cur]) cells[cur].classList.remove('focused');
 cur=i;
 let el=cells[i];
 el.classList.add('focused');
 curInfo={strip:el.dataset.strip,row:el.dataset.row,col:el.dataset.col,png:el.dataset.png,pred:el.dataset.pred,old:el.dataset.old,shown:el.dataset.shown};
 document.getElementById('cur').textContent=el.dataset.strip+' '+el.dataset.row+','+el.dataset.col;
 let v = el.dataset.shown!==''?el.dataset.shown:(el.dataset.old!==''?el.dataset.old:el.dataset.pred);
 document.getElementById('inp').value= v==='-1'||v=== ''?'':v;
 el.scrollIntoView({behavior:'smooth',block:'center'});
}
cells.forEach((el, i)=>{
 el.addEventListener('click', ()=>focusCell(i));
});
function saveValue(v,confirmOnly){
 let el=cells[cur];
 let body={strip:el.dataset.strip,row:+el.dataset.row,col:+el.dataset.col,value:v};
 if(!el.classList.contains('saved')){
  let sc=document.getElementById('saved-cnt');
  if(sc) sc.textContent = parseInt(sc.textContent||'0',10) + 1;
 }
 if(confirmOnly) fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...body,confirm:true})});
 else fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 el.dataset.shown = confirmOnly ? el.dataset.old : String(v);
 el.classList.add('saved');
 el.querySelector('.lbl').textContent = confirmOnly?(el.dataset.old!==''?el.dataset.old:el.dataset.pred):String(v);
 cur=(cur+1)%%cells.length;
 focusCell(cur);
}
document.getElementById('inp').addEventListener('keydown',e=>{
 if(e.key==='Enter'){let t=e.target.value.trim(); if(t!=='') saveValue(t,false);}
 else if(e.key===' '){e.preventDefault(); saveValue(null,true);}
 else if(e.key==='Escape'){cur=(cur+1)%%cells.length; focusCell(cur);}
});
function confirmAll(){fetch('/confirm_all',{method:'POST'}).then(()=>location.reload());}
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
    idx = json.load(open(os.path.join(cells_dir, "index.json")))
    return idx["photo"], idx["cells"], idx.get("meta", {})


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
        if self.path == "/detection.jpg":
            jpg = os.path.join(CELLS_DIR, "detection.jpg")
            if os.path.isfile(jpg):
                data = open(jpg, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
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
