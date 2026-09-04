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

## Approach 3 — Global 2-D polynomial surface  (pending, branch `trial/grid-fit-surface`)

## Cross-approach summary (to fill once 2 and 3 are done)
(TBD)
