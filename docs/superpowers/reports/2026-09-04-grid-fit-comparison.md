# Global Grid Smooth-Fit — Approach Comparison

Date: 2026-09-04
Spec: `docs/superpowers/specs/2026-09-04-grid-smooth-fit-design.md`
Goal: improve cross-location quality beyond the per-cross ink refinement (~0.5 px median)
by fitting a global smooth grid and snapping crossings onto it.

## Metric
- **Neighbor-fit residual** (reference-bias-free): for each interior crossing, distance
  to the mean of its 4 orthogonal neighbors. Reported as p50 / p90 / max. Source photo:
  `nonograms/20180811_114632.jpg`. Runs at resize_max = 1200.
- **Synthetic control**: recover an analytically-known curved+skewed grid from noisy
  crossings (validates the fit captures real curvature, not just over-smooths).
- Bar to beat: median ~0.5 px, p90 ~1.0 px (current per-cross baseline).

### ⚠ How to read the numbers (important honesty note)
The **neighbor-fit residual is NOT a clean accuracy measure.** It only measures internal
self-consistency (how far each dot sticks out from the mean of its 4 neighbours), so *any*
shared low-order surface trivially scores near-zero — it rewards smoothness, not correctness.
Evidence: A3 order=1 reports a "perfect" 0.000 on the photo yet is shown, by the synthetic
control, to collapse the genuinely-curved grid to a plane (775 px error). **Treat the big
photo "after" falls (e.g. main 0.576 -> 0.049) as an overstatement of real benefit.**
The trustworthy accuracy evidence is the **synthetic known-grid** below, which measures each
method's true error against a known answer.

Synthetic accuracy (curved + skew, 0.8 px injected noise, true mean grid error):
```
RAW   mean 0.299  (this is the injected noise the fits are meant to remove)
A1    mean 0.085
A2    mean 0.148
A3    mean 0.075   <- best
```
So the fits are provably ~2-4x **more accurate** than RAW on knowable data. On the real
photo the fills are only a 1-4 px correction (the photo's detections are already accurate),
so the benefit is visually subtle there — the fits' clear win is on noisier/misdetected input.


## Baseline (current)
Per-cross ink refinement only:
- main: p50 0.50, p90 1.00, max 1.98 (prior session `/tmp` note)
- Re-measured this session with the shared tool: main p50 0.576, p90 1.012, max 1.934;
  top p50 0.887, p90 2.650; left p50 0.843, p90 2.543.

## Approach 1 — Separable parametric row/col curves + perspective  ✅ (branch `trial/grid-fit-splines`, commit `3239584`)
Model: fit each grid line as a low-order polynomial in the opposite index, then fit each
coefficient band as a low-order polynomial across the line-family index (so curves vary
smoothly across the family). `grid_smooth_fit_approach1(locs, order, coeff_order)`.

### Real photo (order=2, coeff_order=2)
| region | before p50 / p90 / max | after p50 / p90 / max |
|--------|------------------------|-----------------------|
| main (n=551) | 0.576 / 1.012 / 1.934 | **0.053 / 0.076 / 0.092** |
| top  (n=116) | 0.887 / 2.650 / 3.498 | **0.416 / 0.448 / 0.459** |
| left (n=114) | 0.843 / 2.543 / 4.112 | **0.325 / 0.331 / 0.333** |

Order=3 with coeff_order=2 is worse on main (p50 0.096) — order=2 is the sweet spot.

### Synthetic
Curved+skewed 10x12 grid, 0.8 px noise: mean residual 0.299 → **0.090** (ratio 0.30).
With zero noise the model recovers the true grid **exactly** (all corners + row 5 exact),
confirming it follows genuine curvature rather than over-smoothing.

### Notes / pitfalls hit during implementation
- A ridge penalty on across-line coefficient second differences is **wrong for this**:
  its free endpoints get pulled toward the interior (shifting the whole grid) and it
  penalizes genuine low-frequency curvature (82x worse at ridge=2). Replaced with fitting
  each coefficient band as a low-order polynomial (degree = coeff_order) across the family.
  This objectively captures "smooth across family" without fighting curvature.

## Approach 2 — Independent per-line polynomial fit  ✅ (branch `trial/grid-fit-perline`)
Model: fit each horizontal line as a polynomial (degree `order`) in column index
(for x) and each vertical line as a polynomial in row index (for y), **independently** —
no shared/global model, no cross-family smoothing.

### Real photo (order=2)
| region | before p50 / p90 / max | after p50 / p90 / max |
|--------|------------------------|-----------------------|
| main (n=551) | 0.576 / 1.012 / 1.934 | 0.111 / 0.218 / 0.536 |
| top  (n=116) | 0.887 / 2.650 / 3.498 | 0.425 / 0.713 / 1.043 |
| left (n=114) | 0.843 / 2.543 / 4.112 | 0.381 / 0.705 / 2.432 |

Order=3 is worse on main (p50 0.131); order=2 is the per-line sweet spot.

### Synthetic
Same curved+skewed grid, 0.8 px noise: mean residual 0.299 → **0.148** (ratio 0.50),
vs approach 1's 0.090 (ratio 0.30). A2 reduces noise but is worse than A1 because it
cannot borrow strength across lines.

### Comparison read (A1 vs A2, both order=2)
- **main**: A1 wins clearly (p50 0.053 vs 0.111, ~2x; p90 0.076 vs 0.218; max 0.092 vs
  0.536). The shared model pays off most on the large, well-populated main grid.
- **top/left**: close at p50 (A1 0.416/0.325 vs A2 0.425/0.381), but A2's p90 is notably
  worse (0.713/0.705 vs 0.448/0.331). Fewer lines in clue regions shrink the gap.
- Both are large improvements over the current per-cross baseline (~0.5 px median).

## Approach 3 — Global 2-D polynomial surface  ✅ (branch `trial/grid-fit-surface`)
Model: fit `x(r,c)` and `y(r,c)` as one global bivariate polynomial (full total
degree, over the index domain mapped to [0,1]^2). Every crossing contributes to a
single shared surface; rows/columns coupled rigidly. `grid_smooth_fit_approach3(locs, order)`.

### Real photo (order=2 — see caveat)
| region | before p50 / p90 / max | after p50 / p90 / max |
|--------|------------------------|-----------------------|
| main (n=551) | 0.576 / 1.012 / 1.934 | **0.049 / 0.049 / 0.049** |
| top  (n=116) | 0.887 / 2.650 / 3.498 | 0.410 / 0.410 / 0.410 |
| left (n=114) | 0.843 / 2.543 / 4.112 | 0.320 / 0.320 / 0.320 |

### Synthetic — picks the right order (critical)
| order | zero-noise recovery | meaning |
|-------|--------------------|---------|
| 1 | **775 px error** | plane collapses the curved grid — gross over-smooth |
| 2 | 0.000 px | biquadratic exactly spans true form |
| 3,4 | 0.000 px | higher orders also recover |

With 0.8 px noise (order=2): 0.299 → **0.075** (ratio 0.25) — best of the three on the synthetic.

### ⚠ Important metric caveat (why the synthetic control matters)
On the real photo, **A3 order=1 reports a "perfect" 0.000 neighbor-residual**, yet the
synthetic shows it collapses the genuine curved/lens grid to a plane (775 px error).
The neighbor-fit residual only measures internal self-consistency, which any shared
low-order surface trivially satisfies — it cannot distinguish "correctly smooth" from
"over-smoothed to a blob." The synthetic-known-grid control is what exposes this.
**Conclusion: only measurable on real curvature; order=2 (biquadratic) is the smallest
order that captures real lens/sheet curvature without collapsing it. Treat order=1's
0.000 as an artifact, not a win.**

## Cross-approach summary (A1/A2/A3, main grid, order=2 unless noted)
| approach | synthetic resid (noise=0.8) | real main p50 | real main p90 | model |
|----------|------------------------------|---------------|---------------|-------|
| baseline | — | 0.576 | 1.012 | per-cross only |
| A1 separable + smooth bands | 0.090 (0.30x) | **0.053** | 0.076 | row/col curves, smooth coeff bands |
| A2 independent per-line | 0.148 (0.50x) | 0.111 | 0.218 | independent per-line polys |
| A3 global bivariate (order 2) | **0.075 (0.25x)** | **0.049** | 0.049 | one shared 2-D polynomial |

Recommendation (honest framing): On **provable accuracy** (synthetic known-grid), the order is
A3 > A1 > A2 > RAW — the fits genuinely reduce true error 2-4x and remove any systematic
bias. **A3 (global bivariate, order=2) is recommended** as the primary snap: simplest,
most accurate on the synthetic, and the most self-consistent on the photo. **A1** is the
close fallback if real photos show asymmetric/non-polynomial warp (its separable per-line
curves follow arbitrary smooth sheet shape without forcing one global polynomial). **A2**
(independent per-line) is the weakest and lives mainly as proof that the shared model pays.

Caveat on real-world impact: on an **already-clean photo** the fit is a subtle 1-4 px
correction and RAW can look equally good by eye — the neighbor-residual headline numbers
overstate the benefit (see the warning above). The fits earn their keep on noisier input,
thick/misdetected crosses, or systematic half-pixel bias. If deployment photos are always
this clean, weigh whether the small gain justifies the added coupling; if they can be noisy,
integrate A3 (or A1).

