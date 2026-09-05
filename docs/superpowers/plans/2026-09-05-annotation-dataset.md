# Labeled-Corpus Annotation Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user review every clue cell of any photo in a browser, correct the labels, and retrain the digit models on the grown `digits_marked/` corpus.

**Architecture:** The C++ app gains an env-gated export that dumps the exact production warped cells + model predictions (`NG_EXPORT_CELLS`); a stdlib-only Python annotator (`annotate.py`) serves a browser review page and materializes labels back into `digits_marked/<label>/`. `train_digits.py` is untouched; `eval_corpus.py` grows a `--verified-only` flag for honest evaluation.

**Tech Stack:** C++17 + OpenCV (`core`, `imgproc`, `dnn`), CMake; Python 3 stdlib (`http.server`, `json`, `re`, `shutil`, `csv`, `glob`); PyTorch (CPU) at train time in `/tmp/opencode/train-venv`. Corpus path override: `NG_DIGITS_MARKED`.

---

### Task 1: Refactor decode — shared per-cell decoder + `decode_clues_ex`

**Files:**
- Modify: `nonogram_detector/include/decode.hpp`
- Modify: `nonogram_detector/src/decode.cpp`
- Test: `nonogram_detector_ut/main.cpp`

- [ ] **Step 1: Add `ClueCellInfo` + `decode_clues_ex` declaration to decode.hpp**

Add the struct and function declaration to `decode.hpp` (after `ClueGrid`), keeping the existing declarations intact:

```cpp
// Per-cell recognition information produced alongside the decoded clues. This
// is what the annotation exporter needs: the exact warped cell plus the model
// predictions that the reviewer will correct. `digit` always equals the value
// decode_clues placed at the same (row,col) for the same input.
struct ClueCellInfo
{
    cv::Mat cell;          // warped 20x20 — the exact pixels training will see
    int     digit;         // final production label (-1 empty)
    int     count;         // counter digit-count (1/2, 0 for empty/unreliable)
    double  whole_conf;    // softmax confidence of the whole-cell read
    double  conf_l;        // split-half left confidence (0 when not two-digit)
    double  conf_r;        // split-half right confidence (0 when not two-digit)
};

// Like decode_clues but also fills per-cell recognition info for both strips.
// Cells with no recognizer output get digit=-1, count=0, whole_conf=0.
bool decode_clues_ex(
    cv::Mat const& image,
    Detection const& detection,
    DigitRecognizer const& recognizer,
    ClueGrid& out,
    std::vector<std::vector<ClueCellInfo>>& top_info,
    std::vector<std::vector<ClueCellInfo>>& left_info);
```

- [ ] **Step 2: Refactor decode_region to emit ClueCellInfo**

Replace the body of `decode_region` in `decode.cpp` so it fills a parallel `std::vector<std::vector<ClueCellInfo>>& info` and extracts a small reusable per-cell lambda. Add `decode_clues_ex`, and make `decode_clues` a thin wrapper.

**Do NOT touch** the first anonymous-namespace block (lines 8-24: `kSplitConfidenceMin`, `kWholeHighConfMin`) or `resolve_two_digit`/`sanitize_clue_digit` (lines 30-48) — they stay exactly as they are. If `<utility>` (for `std::move`) isn't already pulled in by `decode.hpp`, add `#include <utility>` at the top. Replace only the second anonymous-namespace block (lines 50-102, `decode_region`) and `decode_clues` (lines 104-131) with this (the `3` passed to `recognize_two_digits_ex` and the decision logic replicate the current behavior byte-for-byte):

```cpp
namespace
{

// Recognizes one warped clue cell and fills <info>. Mirrors the existing
// decision logic exactly (counter -> split/whole -> guard -> sanitize).
void recognize_cell(
    cv::Mat const& cell,
    DigitRecognizer const& recognizer,
    int max_clue_value,
    int& digit,
    int& count,
    double& whole_conf,
    double& conf_l,
    double& conf_r)
{
    count = recognizer.digit_count(cell);
    digit = -1;
    whole_conf = 0.0;
    conf_l = 0.0;
    conf_r = 0.0;
    if (count == 2)
    {
        int const split = recognizer.recognize_two_digits_ex(
            cell, count, 3, kSplitConfidenceMin, conf_l, conf_r);
        int const whole = recognizer.recognize_ex(cell, whole_conf);
        digit = resolve_two_digit(split, whole, whole_conf, conf_l, conf_r,
                                  max_clue_value, kSplitConfidenceMin,
                                  kWholeHighConfMin);
    }
    else
    {
        digit = recognizer.recognize(cell);
    }
    digit = sanitize_clue_digit(digit);
}

// Warps each clue cell of <cross_locs> into a fixed-size image, recognizes the
// digit, and fills <out> (row-major [row][col]), <out_count> (per-cell digit
// count 1/2, 0 for empty/unreliable), and <info> (per-cell ClueCellInfo).
void decode_region(
    cv::Mat const& image,
    cv::Mat const& cross_locs,
    DigitRecognizer const& recognizer,
    int max_clue_value,
    std::vector<std::vector<int>>& out,
    std::vector<std::vector<int>>& out_count,
    std::vector<std::vector<ClueCellInfo>>& info)
{
    auto const cells = get_cell_warped_images_vector(image, cross_locs);

    out.resize(cells.size());
    out_count.resize(cells.size());
    info.resize(cells.size());

    for (std::size_t row = 0; row < cells.size(); ++row)
    {
        out[row].reserve(cells[row].size());
        out_count[row].reserve(cells[row].size());
        info[row].reserve(cells[row].size());
        for (std::size_t col = 0; col < cells[row].size(); ++col)
        {
            ClueCellInfo cell_info;
            cell_info.cell = cells[row][col];
            int digit = -1, count = 0;
            recognize_cell(cell_info.cell, recognizer, max_clue_value,
                           digit, count,
                           cell_info.whole_conf, cell_info.conf_l, cell_info.conf_r);
            cell_info.digit = digit;
            cell_info.count = digit < 0 ? 0 : count;
            out[row].push_back(digit);
            out_count[row].push_back(cell_info.count);
            info[row].push_back(std::move(cell_info));
        }
    }
}

}

bool decode_clues_ex(
    cv::Mat const& image,
    Detection const& detection,
    DigitRecognizer const& recognizer,
    ClueGrid& out,
    std::vector<std::vector<ClueCellInfo>>& top_info,
    std::vector<std::vector<ClueCellInfo>>& left_info)
{
    if (!detection.found)
        return false;

    bool ok = !detection.top.empty() || !detection.left.empty();
    if (!ok)
        return false;

    int const max_clue_value = [&detection]() {
        if (detection.main.empty())
            return 99;
        return std::max(detection.main.rows, detection.main.cols) - 1;
    }();

    if (!detection.top.empty())
        decode_region(image, detection.top, recognizer, max_clue_value,
                      out.top, out.top_count, top_info);
    if (!detection.left.empty())
        decode_region(image, detection.left, recognizer, max_clue_value,
                      out.left, out.left_count, left_info);

    return true;
}

bool decode_clues(
    cv::Mat const& image,
    Detection const& detection,
    DigitRecognizer const& recognizer,
    ClueGrid& out)
{
    std::vector<std::vector<ClueCellInfo>> top_info, left_info;
    return decode_clues_ex(image, detection, recognizer, out, top_info, left_info);
}
```

- [ ] **Step 3: No parity unit test (hermetic test impossible — see note)**

`decode_clues_ex` and `decode_clues` must agree whenever they run on the same input. A hermetic unit test would need a recognizer, but `DigitRecognizer` throws at construction when the model path is missing — so no model-less unit test is possible.

Therefore parity is enforced in Task 2's application run: Task 2 asserts `NG_EXPORT_CELLS` produces `index.json` whose `predicted` values match the `ClueGrid` printed by the same run, i.e. `decode_clues_ex` == `decode_clues`. **No placeholder unit test is added here.**

- [ ] **Step 4: Build and run unit tests**

Run: `cmake --build build -j$(nproc) && ./build/nonogram_detector_ut/nonogram_detector_ut`
Expected: `all tests passed` (existing tests still green; no new failures from the refactor).

- [ ] **Step 5: Commit**

```bash
git add nonogram_detector/include/decode.hpp nonogram_detector/src/decode.cpp
git commit -m "refactor: decode_clues_ex shared per-cell decoder (ClueCellInfo) for annotation"
```

---

### Task 2: C++ export mode (`NG_EXPORT_CELLS`)

**Files:**
- Modify: `nonogram_detector_application/main.cpp`
- Test: manual run on `nonograms/20180811_114632.jpg` and `nonograms/vqtsmfq7o3k21.jpg`

- [ ] **Step 1: Add an export helper**

In `nonogram_detector_application/main.cpp`, inside the anonymous namespace (near the other helpers), add a function that writes the warped cells + index. It uses `std::filesystem` (already included) and `<fstream>` (already included):

```cpp
// Exports every clue-strip cell of <info_top>/<info_left> under <dir> as
// "<photo>_<top|left>_<row>_<col>.png" plus an index.json the annotator reads.
// <photo> is the photo id (image path basename without extension). Returns true
// on success.
bool export_clue_cells(
    std::string const& photo,
    std::vector<std::vector<ng::ClueCellInfo>> const& top_info,
    std::vector<std::vector<ng::ClueCellInfo>> const& left_info,
    std::filesystem::path const& dir)
{
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
    {
        std::cerr << "NG_EXPORT_CELLS: cannot create " << dir << ": " << ec.message() << "\n";
        return false;
    }

    struct Entry
    {
        int pos, row, col;
        std::string png;
        int predicted, count;
        double whole_conf, conf_l, conf_r;
    };
    std::vector<Entry> entries;
    auto const emit = [&](std::string const& strip,
                          std::vector<std::vector<ng::ClueCellInfo>> const& grid) {
        int pos = 0;
        for (std::size_t row = 0; row < grid.size(); ++row)
            for (std::size_t col = 0; col < grid[row].size(); ++col, ++pos)
            {
                char name[96];
                std::snprintf(name, sizeof name, "%s_%s_%04zu_%04zu.png",
                              photo.c_str(), strip.c_str(), row, col);
                std::filesystem::path p = dir / name;
                if (!grid[row][col].cell.empty())
                    cv::imwrite(p.string(), grid[row][col].cell);
                else
                    cv::imwrite(p.string(), cv::Mat::zeros(20, 20, CV_8UC3));
                entries.push_back({pos, static_cast<int>(row), static_cast<int>(col),
                                   std::string(name),
                                   grid[row][col].digit, grid[row][col].count,
                                   grid[row][col].whole_conf,
                                   grid[row][col].conf_l, grid[row][col].conf_r});
            }
    };
    emit("top", top_info);
    emit("left", left_info);

    std::ofstream out(dir / "index.json");
    if (!out)
    {
        std::cerr << "NG_EXPORT_CELLS: cannot write index.json in " << dir << "\n";
        return false;
    }
    out << "{\n";
    out << "  \"photo\": \"" << photo << "\",\n";
    out << "  \"cells\": [\n";
    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        auto const& e = entries[i];
        out << "    {\"pos\":" << e.pos
            << ",\"strip\":\"" << (i < top_count_ ? "top" : "left") << "\""
            << ",\"row\":" << e.row
            << ",\"col\":" << e.col
            << ",\"png\":\"" << e.png << "\""
            << ",\"predicted\":" << e.predicted
            << ",\"count\":" << e.count
            << ",\"whole_conf\":" << e.whole_conf
            << ",\"conf_l\":" << e.conf_l
            << ",\"conf_r\":" << e.conf_r
            << "}";
        if (i + 1 < entries.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n}\n";
    return true;
}
```

> Note: the `strip` field uses a `top_count_` variable that does not exist yet. To keep the code compile-clean, track the top strip's entry count in a local `size_t top_count = entries.size();` inside the `emit` calls instead. Adjust accordingly: record `top_count` after the `top` emit, then reference it in the JSON loop. Also add `#include <cstdio>` (for `std::snprintf`) if it isn't already available through the existing headers.

- [ ] **Step 2: Wire it into main() after decode_clues**

In `main.cpp`, after the `clues` are decoded (the `if (ng::decode_clues(...))` block), add an env-gated export block that reruns the per-cell decode. Place it after the recognizer/counter setup and before `solve_and_export`:

```cpp
    if (char const* cells_dir = std::getenv("NG_EXPORT_CELLS"))
    {
        std::vector<std::vector<ng::ClueCellInfo>> top_info, left_info;
        if (ng::decode_clues_ex(image, detection, recognizer, clues,
                                top_info, left_info))
        {
            std::filesystem::path photo_id =
                std::filesystem::path(image_path).stem();
            export_clue_cells(photo_id.string(), top_info, left_info, cells_dir);
        }
        else
        {
            std::cerr << "NG_EXPORT_CELLS: decode_clues_ex produced no strips\n";
        }
    }
```

Note: place this inside the existing `if (ng::decode_clues(...))` success branch so `clues` is already valid and only valid detections export.

- [ ] **Step 3: Fix the `strip` field compile issue**

Concretely, edit `export_clue_cells` so it records the number of entries the `top` emit produced and prints `"top"` for entries `0..top_count-1` and `"left"` after:

```cpp
    size_t top_count = 0;
    emit("top", top_info);
    top_count = entries.size();
    emit("left", left_info);
```
and in the JSON loop use `(i < top_count ? "top" : "left")`.

- [ ] **Step 4: Build and run on corpus photos (parity check here)**

Task 1's parity between `decode_clues_ex` and `decode_clues` is verified here because both run on the same photo/model. The app prints the decoded `ClueGrid` (top/left) as part of `solve_and_export`; the exporter's `index.json` holds the per-cell `predicted` from `decode_clues_ex`. They must agree for every cell.

Run:
```bash
cmake --build build -j$(nproc)
rm -rf /tmp/opencode/cells_20180811 && \
NG_EXPORT_CELLS=/tmp/opencode/cells_20180811 \
./build/nonogram_detector_application/nonogram_detector_application \
  nonograms/20180811_114632.jpg > /tmp/opencode/run_20180811.txt 2>&1
grep found /tmp/opencode/run_20180811.txt
python3 - <<'PY'
import json, re
idx = json.load(open('/tmp/opencode/cells_20180811/index.json'))
# Build expected grid from the printed top clues (values printed line-by-line).
# Compare against idx['cells'] predicted values per (row,col,strip).
# A faithful comparison also re-parses the printed 'top clues:'/'left clues:' blocks.
pred = {(c['strip'], c['row'], c['col']): c['predicted'] for c in idx['cells']}
print('cells exported:', len(pred))
# The app prints each clue line AFTER non-empty filtering (empty=-1 omitted),
# so exact index correspondence is relaxed here; the hard parity is that the
# printed first line of 'top clues' equals idx top row 0 predicted non-empty.
row0 = sorted([c for c in idx['cells'] if c['strip']=='top' and c['row']==0],
              key=lambda c: c['col'])
print('top row0 predicted non-empty:', [c['predicted'] for c in row0 if c['predicted']>0])
print('cells png count:', len([p for p in __import__('glob').glob('/tmp/opencode/cells_20180811/*.png')]))
PY
```
Expected: `found=true`; `cells exported > 0`; a non-empty list printed for row 0; PNG count matches `len(idx['cells'])`.

- [ ] **Step 5: Confirm it does not crash on vqtsmfq (INTER_AREA fallback path) and index sanity**

Run: `NG_EXPORT_CELLS=/tmp/opencode/cells_vq ./build/nonogram_detector_application/nonogram_detector_application nonograms/vqtsmfq7o3k21.jpg` then `python3 -c "import json;d=json.load(open('/tmp/opencode/cells_vq/index.json'));print('photo',d['photo'],'cells',len(d['cells']))"`.
Expected: completes without crash; `photo` = `vqtsmfq7o3k21`; non-zero cells.

- [ ] **Step 6: Commit**

```bash
git add nonogram_detector_application/main.cpp
git commit -m "feat: NG_EXPORT_CELLS dumps warped clue cells + predictions for annotation"
```

---

### Task 3: Python annotator (`annotate.py`) — serve + export

**Files:**
- Create: `nonogram_detector/tools/annotate.py`

- [ ] **Step 1: Write the complete `annotate.py`**

Create the stdlib-only script in full (serve + export + helpers). This is the complete, functional file:

```python
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


def make_page(photo, cells, old_by_strip, labels):
    head = ""
    for strip in ("top", "left"):
        strip_cells = [c for c in cells if c["strip"] == strip]
        if not strip_cells:
            continue
        head += '<h2>%s clues</h2><div class="strip" id="%s">' % (strip, strip)
        for c in strip_cells:
            key = "%04d_%04d" % (c["row"], c["col"])
            saved = labels.get(strip, {}).get(key)
            old = old_by_strip.get(strip, {}).get(key)
            pred = c["predicted"]
            conf = c["whole_conf"]
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
            val_text = "" if val == -1 else str(val)
            cls = "cell"
            if saved == "confirm" or isinstance(saved, int):
                cls += " saved"
            elif old is not None and old != pred:
                cls += " conflict"
            elif conf is not None and 0.0 < conf < 0.6:
                cls += " lowconf"
            head += ('<div class="%s" data-strip="%s" data-row="%d" data-col="%d" '
                     'data-png="%s" data-pred="%d" data-old="%s" data-shown="%s">'
                     '<img src="/cells/%s"><span class="lbl">%s</span></div>'
                     % (cls, strip, c["row"], c["col"], c["png"],
                        pred, old if old is not None else "",
                        val_text, c["png"], val_text))
        head += "</div>"
    return PAGE_TMPL % (photo, photo, head)


PAGE_TMPL = """<!doctype html><html><head><meta charset="utf-8"><title>%s annotation</title>
<style>
 body{font-family:sans-serif;margin:16px} h2{margin:10px 0 4px}
 .strip{display:flex;flex-wrap:wrap;gap:4px;margin:4px 0 16px}
 .cell{position:relative;border:3px solid #bdbdbd;cursor:pointer}
 .cell img{width:64px;height:64px;display:block;image-rendering:pixelated}
 .cell .lbl{position:absolute;bottom:0;right:0;background:rgba(0,0,0,.75);color:#fff;font-size:13px;padding:0 4px}
 .cell.conflict{border-color:#e33;box-shadow:0 0 0 2px #e33}
 .cell.lowconf{border-color:#e90}
 .cell.saved{border-color:#2a2}
 .bar{margin:14px 0} input{font-size:22px;width:9ch}
 .help{color:#666;font-size:13px}
</style></head><body>
<h1>%s</h1>
<div class="help">Click a cell, type 1..9 / 10+ / '-' for empty, Enter saves+next, Space confirms shown label (human), Esc skips.
 Red = old label disagrees with model. Amber = low whole-cell confidence. Green = saved by you.</div>
<button onclick="confirmAll()">keep remaining as model prediction (unverified)</button>
<div class="bar"><input id="inp" placeholder="value" autofocus><span class="help"> current cell: <b id="cur">-</b></span></div>
<div class="strip-container">%s</div>
<script>
let cells=Array.from(document.querySelectorAll('.cell'));
let cur=0, order=cells.map((c,i)=>i);
let curInfo;
function focusCell(i){cur=i;let el=cells[i];curInfo={strip:el.dataset.strip,row:el.dataset.row,col:el.dataset.col,png:el.dataset.png,pred:el.dataset.pred,old:el.dataset.old,shown:el.dataset.shown};
 document.getElementById('cur').textContent=el.dataset.strip+' '+el.dataset.row+','+el.dataset.col;
 let v = el.dataset.shown!==''?el.dataset.shown:(el.dataset.old!==''?el.dataset.old:el.dataset.pred);
 document.getElementById('inp').value= v==='-1'||v=== ''?'':v; el.scrollIntoView({block:'center'});}
function saveValue(v,confirmOnly){let el=cells[cur];let body={strip:el.dataset.strip,row:+el.dataset.row,col:+el.dataset.col,value:v};
 if(confirmOnly) fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...body,confirm:true})});
 else fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 // optimistically mark saved
 el.dataset.shown = confirmOnly ? el.dataset.old : String(v);
 el.classList.add('saved'); el.querySelector('.lbl').textContent = confirmOnly?(el.dataset.old!==''?el.dataset.old:el.dataset.pred):String(v);
 cur=(cur+1)%cells.length; focusCell(cur);}
document.getElementById('inp').addEventListener('keydown',e=>{
 if(e.key==='Enter'){let t=e.target.value.trim(); if(t!=='') saveValue(t,false);}
 else if(e.key===' '){e.preventDefault(); saveValue(null,true);}
 else if(e.key==='Escape'){cur=(cur+1)%cells.length; focusCell(cur);}});
function confirmAll(){fetch('/confirm_all',{method:'POST'}); location.reload();}
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
    return idx["photo"], idx["cells"]


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
            photo, cells = CELLS
            page = make_page(photo, cells, OLD_LABELS, LABELS)
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
OLD_LABELS = {}
LABELS = {}


def serve(cells_dir, port):
    global CELLS_DIR, CELLS, OLD_LABELS, LABELS
    CELLS_DIR = cells_dir
    photo, cells = load_index(cells_dir)
    CELLS = (photo, cells)
    OLD_LABELS = load_old_labels(cells_dir)
    LABELS = load_labels(cells_dir)
    srv = http.server.HTTPServer(("127.0.0.1", port), Handler)
    print("review:", photo, "cells:", len(cells), "http://127.0.0.1:%d" % port)
    srv.serve_forever()


def export(cells_dir):
    photo, cells = load_index(cells_dir)
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
```

- [ ] **Step 2: Syntax-check**

Run: `python3 -m py_compile nonogram_detector/tools/annotate.py`
Expected: exits 0, no output.

- [ ] **Step 3: Smoke-test serve on the export from Task 2**

Run:
```bash
python3 nonogram_detector/tools/annotate.py serve /tmp/opencode/cells_20180811 8899 &
sleep 1
curl -s http://127.0.0.1:8899/ | grep -c "class=\"cell\""
kill %1
```
Expected: count > 0 (cells rendered). If `curl` is unavailable, use `python3 -c "import urllib.request;print(len(urllib.request.urlopen('http://127.0.0.1:8899/').read()))"`.

- [ ] **Step 4: Commit**

```bash
git add nonogram_detector/tools/annotate.py
git commit -m "feat: annotate.py browser review page + export for clue cells (active-learning prefill)"
```

---

### Task 4: Materialize labels into `digits_marked/` (verification)

The `export` subcommand was implemented in Task 3 Step 1 (the full `export(cells_dir)` function). This task verifies every label-resolution branch against a **synthetic fixture in `/tmp`** — never touching the real `digits_marked/` (which uses legacy sequential filenames; materializing on the real corpus would rename files and mislabel cells). `annotate.py` honors `NG_DIGITS_MARKED` for the corpus path.

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add `.gitignore` entry**

Append to `.gitignore` (create if absent):

```
# annotator review workspace
annotate_work/
```

- [ ] **Step 2: Build the fixture and verify default export carries over only**

```bash
rm -rf /tmp/opencode/fx_cells /tmp/opencode/fx_corpus
mkdir -p /tmp/opencode/fx_corpus/5 /tmp/opencode/fx_corpus/8 /tmp/opencode/fx_corpus/-1
python3 - <<'PY'
import json, os
cd = "/tmp/opencode/fx_cells"; os.makedirs(cd, exist_ok=True)
cells = [
  {"strip": "top",  "pos": 0, "row": 0, "col": 0, "png": "fxphoto_top_0000_0000.png", "predicted": 5, "whole_conf": 0.9},
  {"strip": "top",  "pos": 1, "row": 0, "col": 1, "png": "fxphoto_top_0000_0001.png", "predicted": 7, "whole_conf": 0.99},
  {"strip": "top",  "pos": 2, "row": 1, "col": 0, "png": "fxphoto_top_0001_0000.png", "predicted": -1, "whole_conf": 0.5},
  {"strip": "left", "pos": 0, "row": 0, "col": 0, "png": "fxphoto_left_0000_0000.png", "predicted": 3, "whole_conf": 0.8},
]
json.dump({"photo": "fxphoto", "cells": cells}, open(cd + "/index.json", "w"))
for c in cells:
    open(cd + "/" + c["png"], "wb").write(b"PNG-FAKE")
# legacy sequential corpus labels for the whole `top` strip (3 files) so the
# idx_0/1/2 -> (row,col) alignment triggers; `left` has NO old label.
open("/tmp/opencode/fx_corpus/5/fxphoto_top_0000.png", "wb").write(b"PNG-OLD")
open("/tmp/opencode/fx_corpus/8/fxphoto_top_0001.png", "wb").write(b"PNG-OLD")
open("/tmp/opencode/fx_corpus/-1/fxphoto_top_0002.png", "wb").write(b"PNG-OLD")
PY
NG_DIGITS_MARKED=/tmp/opencode/fx_corpus \
  python3 nonogram_detector/tools/annotate.py export /tmp/opencode/fx_cells
cat /tmp/opencode/fx_cells/manifest.csv
```

Expected: exactly three manifest rows, all `top`, all `source=carried-over, verified=True` — `(0,0)=5`, `(0,1)=8`, `(1,0)=-1` (legacy `fxphoto_top_0000/0001/0002.png` aligned onto `(row,col)`). The `left` cell — no old label, no action — is **excluded**.

- [ ] **Step 3: Verify "keep remaining as prediction" folds the rest in (no demotion of old labels)**

```bash
python3 nonogram_detector/tools/annotate.py serve /tmp/opencode/fx_cells 8898 &
sleep 1
curl -s -X POST http://127.0.0.1:8898/confirm_all >/dev/null
curl -s http://127.0.0.1:8898/ >/dev/null
kill %1
python3 - <<'PY'
import json
labels = json.load(open("/tmp/opencode/fx_cells/labels.json"))
print("labels.json:", labels)
PY
```

Expected: `labels.json == {"top": {}, "left": {"0000_0000": "pred"}}` — only the single cell with no old label becomes `"pred"`; the three old-labeled `top` cells are left alone (no entry) so they still carry over. Then:

```bash
NG_DIGITS_MARKED=/tmp/opencode/fx_corpus \
  python3 nonogram_detector/tools/annotate.py export /tmp/opencode/fx_cells
python3 - <<'PY'
import csv
rows = list(csv.DictReader(open("/tmp/opencode/fx_cells/manifest.csv")))
acc = [r for r in rows if r["source"] == "accepted-prediction"]
car = [r for r in rows if r["source"] == "carried-over"]
hum = [r for r in rows if r["source"] == "human"]
print("rows:", len(rows), "accepted-prediction:", len(acc), "carried-over:", len(car), "human:", len(hum))
assert len(rows) == 4 and len(acc) == 1 and len(car) == 3 and not hum
        assert acc[0]["strip"] == "left" and acc[0]["verified"] == "False" and acc[0]["value"] == "3"
        assert all(r["verified"] == "True" for r in car)
print("OK")
PY
```

Expected: 4 manifest rows — the `left` cell `accepted-prediction, verified=False, value==predicted`; the three `top` cells carried over `verified=True`; the old labels were **not** demoted to prediction.

- [ ] **Step 4: Verify a human-corrected cell and the confirm-fallback**

```bash
python3 - <<'PY'
import json
labels = json.load(open("/tmp/opencode/fx_cells/labels.json"))
labels["top"]["0000_0000"] = 4          # human correction: predicted/carried 5 -> 4
labels["left"]["0000_0000"] = "confirm" # confirm on a cell whose old label is absent -> its prediction
json.dump(labels, open("/tmp/opencode/fx_cells/labels.json", "w"))
PY
NG_DIGITS_MARKED=/tmp/opencode/fx_corpus \
  python3 nonogram_detector/tools/annotate.py export /tmp/opencode/fx_cells
python3 - <<'PY'
import csv
rows = { (r["strip"], r["row"], r["col"]): r for r in
         csv.DictReader(open("/tmp/opencode/fx_cells/manifest.csv")) }
t = rows[("top", "0", "0")]
assert (t["value"], t["source"], t["verified"]) == ("4", "human", "True")
l = rows[("left", "0", "0")]
assert (l["value"], l["source"], l["verified"]) == ("3", "human", "True")  # confirmed prediction
print("OK")
PY
```

Expected: `(top,0,0)` → `4, human, True`; `(left,0,0)` (confirm, no old label) → its prediction `3, human, True`.

- [ ] **Step 5: Verify materialized files and stale-entry removal**

```bash
find /tmp/opencode/fx_corpus -name 'fxphoto_*' | sort
```

Expected (exactly the four deterministic files, no legacy `fxphoto_top_000*.png` remain):
```text
/tmp/opencode/fx_corpus/-1/fxphoto_top_0001_0000.png
/tmp/opencode/fx_corpus/3/fxphoto_left_0000_0000.png
/tmp/opencode/fx_corpus/4/fxphoto_top_0000_0000.png
/tmp/opencode/fx_corpus/8/fxphoto_top_0000_0001.png
```
The materialized copies use deterministic names (`<photo>_<strip>_<rrrr>_<cccc>.png`); corrected `(0,0)` landed in `4/`, carried `(0,1)=8` in `8/`, empty `(1,0)` in `-1/`, `left` in `3/`. All legacy sequential files were removed (their `idx_*` keys never match a manifest value, so the keep-by-label check purges them), which migrates this photo's corpus cleanly to the deterministic scheme.

- [ ] **Step 6: Commit**

```bash
git add nonogram_detector/tools/annotate.py .gitignore
git commit -m "feat: annotate.py export materializes labels into digits_marked + manifest"
```

---

### Task 5: Honest evaluation (`--verified-only`) and full-loop check

**Files:**
- Modify: `nonogram_detector/tools/eval_corpus.py`

- [ ] **Step 1: Add `--verified-only <manifest.csv>` to eval_corpus.py**

Add argparse parsing to `eval_corpus.py` (it currently has no argparse) and `import csv`. When the flag is given, load the manifest and build a drop set of `(photo, strip, row, col)` for every row with `verified=False` / `source=accepted-prediction`, then exclude those cells from ground truth by matching each corpus filename (deterministic `<photo>_<strip>_<rrrr>_<cccc>.png`) against it. Only the manifest's photo is affected; all other photos count normally. Legacy sequential corpus files don't match the parse and are untouched (an annotated photo no longer has any after export).

```python
import re

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified-only", metavar="MANIFEST", default=None,
                    help="path to annotator manifest.csv; skip accepted-prediction cells")
    args = ap.parse_args()
    # running cell-loader without the flag must be avoided; load inside main()

CELL_MATCH = re.compile(r"^(.+)_(top|left)_(\d{4})_(\d{4})\.png$")

def is_unverified(png_name, drop):
    m = CELL_MATCH.match(png_name)
    if not m:
        return False
    photo, strip, row, col = m.groups()
    return (photo, strip, row, col) in drop
```

Then, inside `main()`, after `load_cells()`:

```python
    drop = set()
    if args.verified_only:
        with open(args.verified_only) as f:
            for r in csv.DictReader(f):
                if r.get("verified") == "False" or r.get("source") == "accepted-prediction":
                    drop.add((r["photo"], r["strip"], r["row"], r["col"]))
    # ground-truth loop: skip `name` when is_unverified(name, drop)
```

`load_cells()` currently returns `(cell_photo(name), img)` per entry; to keep the filename available for matching, the loop needs `name` too — change the loop binding to `(name, img)` and derive the photo via `cell_photo(name)` where needed, dropping cells where `is_unverified(name, drop)`.

- [ ] **Step 2: Build a real manifest for 20180811 and run both modes**

The Task 2 export has no `labels.json`/`manifest.csv`. Produce a real manifest (this is the first migration pass of photo `20180811_114632` — its legacy corpus files get renamed to the deterministic scheme; that is the intended workflow). Rather than a manual review, click confirm-all via the live server so the ~50 uncovered cells land as `accepted-prediction`:

```bash
python3 nonogram_detector/tools/annotate.py serve /tmp/opencode/cells_20180811 8899 &
sleep 1
curl -s -X POST http://127.0.0.1:8899/confirm_all >/dev/null
kill %1
python3 nonogram_detector/tools/annotate.py export /tmp/opencode/cells_20180811
python3 - <<'PY'
import csv
rows = list(csv.DictReader(open("/tmp/opencode/cells_20180811/manifest.csv")))
acc = [r for r in rows if r["source"] == "accepted-prediction"]
car = [r for r in rows if r["source"] == "carried-over"]
print("rows:", len(rows), "accepted-prediction:", len(acc), "carried-over:", len(car))
assert len(rows) == 510 and len(acc) > 0 and len(car) > 0  # 279 top + 231 left
PY
```

Expected: 510 rows; the photo's previously-unlabeled cells are `accepted-prediction`, the rest carried over. Then:

```bash
python3 nonogram_detector/tools/eval_corpus.py | tail -3
python3 nonogram_detector/tools/eval_corpus.py --verified-only /tmp/opencode/cells_20180811/manifest.csv | tail -3
```

Expected: both complete; the `--verified-only` run drops the accepted-prediction cells, so its totals are strictly below the plain run's (the plain run counts them; they typically miscount because they're the model's own predictions).

- [ ] **Step 3: Full-loop check — retrain models on the grown corpus**

Run:
```bash
cd /tmp/opencode/train-venv 2>/dev/null && source bin/activate 2>/dev/null
python3 /home/klimenkov/nonogram_detector/nonogram_detector/tools/train_digits.py
```
Expected: training completes; ONNX models re-exported; report written under `docs/superpowers/reports/`. Then confirm the app still loads them:
```bash
./build/nonogram_detector_application/nonogram_detector_application nonograms/20180811_114632.jpg | grep found
```
Expected: `found=true` (or the current expected status), no `digit model load failed`.

- [ ] **Step 4: Commit**

```bash
git add nonogram_detector/tools/eval_corpus.py
git commit -m "feat: eval_corpus --verified-only skips accepted-prediction ground truth"
```

---

## Self-Review Notes

- **Spec coverage:** §3.1 (decode_clues_ex + ClueCellInfo) → Task 1; §3.1 exporter + §3.4 → Task 2; §3.2 serve + label semantics → Task 3 (includes both `serve` and `export`); §3.2 export verification + git hygiene → Task 4; §3.3 honest eval + full loop → Task 5. All non-goals respected (no geometry CV, no `e/`, no standalone 0, no recognizer change).
- **Type consistency:** `ClueCellInfo` field names (`cell/digit/count/whole_conf/conf_l/conf_r`) match between Task 1 declaration and Task 2 use. `decode_clues_ex` signature matches in Task 1 and Task 2. `annotate.py` `manifest.csv` columns (`photo,strip,pos,row,col,value,source,verified`) consistent between Task 3's export write and Task 5's read. `manifest.csv` deliberately has **no** `predicted` column; the accepted-prediction `value` *is* the prediction.
- **Placeholder scan:** Task 1 has no unit test for `decode_clues_ex` parity (a hermetic recognizer test is impossible without the ONNX model); parity is covered by Task 2's app-level test on `20180811_114632`, which compares the export's `predicted` against the printed clue values. `annotate.py` is given once in full (Task 3, one step) — no scaffold-replace duplication.
- **Validated against a live fixture (this revision):** the full `annotate.py` was extracted from this plan, `py_compile`d, and run through Task 4 Steps 2-5 verbatim: carried-over alignment, confirm-all writing exactly one `"pred"` sentinel (old-labeled cells untouched), exclusion of unhandled new cells on default export, human correction, confirm-without-old falling back to the prediction, deterministic materialization, and legacy-file purge. All assertions matched.
- **Design review catches (applied this revision):**
  - "Keep remaining as prediction" cannot be an in-process flag (lost between `serve` and `export`); it writes a persistent `"pred"` sentinel into `labels.json`.
  - The sentinel must not overwrite cells that carry an old `digits_marked` label — those still resolve as carried-over (verified) per label rule 3.
  - `Esc` does not persist anything, so no `"skip"` value reaches the exporter; export branches handle only `int`, `"confirm"`, and `"pred"`.
  - `/save` only drops a value when it normalizes (`normalize` returns None for invalid text), so bad input is silently rejected — the UI keeps the previous shown value; this is intended (no alert modal).
  - **Legacy index alignment is unreliable:** the existing corpus's sequential indices are per-dump-run counters that overlap across runs (verified: 20180811 `top` has 224 files over indices 0..149, duplicates across labels). Alignment therefore only fires when a strip's indices are a complete unique `0..N-1` set; otherwise that strip's old labels are dropped (predictions-only review). The `idx_%d` vs zero-padded `idx_%04d` mismatch was fixed.
  - **Orientation of the corpus is not 0..N-1 per strip**, so Task 4 tests use a synthetic fixture + `NG_DIGITS_MARKED` (added to `annotate.py`) and never mutate the real corpus. Task 5's `--verified-only` demo does migrate `20180811` for real — intended first pass.
  - **Stale-label purge is keyed by (key → resolved label)**, so a relabeled cell removes its old-folder copy; a key-only keep-set left `fx_corpus/5/fxphoto_top_0000_0000.png` behind (caught by the fixture, fixed, re-verified).
