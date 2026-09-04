# Global Grid Smooth/Regularize Fit — Design

## Problem

Cross locations are currently refined per-cross in two orthogonal stages:
a paraboloid fit on the template-match response (`refine_peak_loc`) and an
ink-centroid snap (`refine_cross_locs_ink`). Investigation on the real photo
(`nonograms/20180811_114632.jpg`) showed this per-cross approach is near its
practical limit:

- Main-grid residual vs. the neighbor-fitted grid: median ~0.5 px, p90 ~1.0 px,
  max ~2 px, with no systematic per-axis bias — i.e. mostly small random
  scatter plus a component that tracks the sheet's real curvature.
- The remaining apparent "outliers" near clue strips are largely curvature +
  bold-line effects, not digit contamination or non-orthogonality (both
  measured negligible).
- Widening the per-cross ink band to fix thick lines *regresses* real photos
  (p50 0.50 -> 0.71) because it captures adjacent/filled ink.

Conclusion: the per-cross refinement is checked; further gain needs a
**global** model that fits the whole grid as one regular surface and snaps
each cross into it, removing the residual scatter and curvature coherently.

## Goals / Requirements

- Fit a smooth, regular grid to the detected main-grid crosses and snap each
  cross to its fitted intersection.
- Improve whole-grid quality: lower neighbor-fit residual, visibly straighter
  grid lines, consistent intersections, cleaner overlays / `.non` export /
  cell warps.
- Compare **three** candidate fit models empirically on the real photo and
  adjacent synthetic tests, then pick a winner.
- Keep the existing detection pipeline, dimensions, sentinel (`-1,-1`), and
  success behavior intact.

## Non-Goals

- No change to per-cross ink refinement (stage 2) itself.
- No digit-contamination or non-orthogonality work (both shown negligible).
- No change to missing-cross augmentation beyond what the fit implicitly does.
- No production detector change until a winner is chosen from the comparison.

## The three approaches (one branch each)

All three fit the `(rows+1) x (cols+1)` matrix of crosses; each cross has a
grid index `(r, c)` and an observed `(x, y)`. Sentinel/missing crosses are
excluded from the fit.

### Approach 1 — Separable parametric row/col curves + perspective (start)

- Each horizontal line `r` is a low-order parametric curve `x = P_r(t)` over
  column index `t`, and each vertical line `c` a curve `y = Q_c(t)` over row
  index `t`; the two families are fit jointly so row-curves and column-curves
  are each smooth across their family index.
- A small global affine/perspective term handles overall rectification.
- Cross `(r,c)` = intersection of row-curve `r` and column-curve `c`.
- Strongest: handles smooth sheet curvature + lens distortion, robust to
  scattered individual crosses, produces consistent intersections.

### Approach 2 — Independent per-line polynomial fit

- Each row line and each column line fit independently with low-order
  polynomials; no shared global model.
- Simplest to reason about; baseline that shows what the shared/global model
  (Approach 1) actually buys. Independent fits can be inconsistent and
  under-robust on lines with few valid crosses.

### Approach 3 — Global 2-D polynomial surface

- Fit `x(r,c)` and `y(r,c)` as biquadratic/bicubic polynomials over the
  rectangular grid-index domain.
- Very compact, always smooth, few parameters. Couples rows/cols rigidly; real
  non-square curvature may need high order and can oscillate.

## Evaluation (shared metric)

- **Neighbor-fit residual** (reference-bias-free, the metric already trusted):
  per-cross deviation from the smooth grid predicted by its nearest neighbors,
  reported as p50 / p90 / max on the real photo, before vs. after snap.
- **Line straightness**: residual of each fitted grid line vs. its implied
  straight chord (curvature before fit, straightness after).
- **Consistency**: whether row/column fits meet at unique intersections.
- **Synthetic** control: an analytically-known curved/skewed grid recovers its
  true intersections, to validate each fit is not over/under-fitting.
- Current bar to beat: median ~0.5 px, p90 ~1.0 px. Report what each approach
  improves and what it leaves unchanged.

Guard against the earlier pitfall: do not use a darkness-centroid "true line
center" as absolute ground truth (it carries a systematic half-pixel bias); use
neighbor-fit residual and synthetic-known grids instead.

## Implementation plan (branch-per-approach)

- Branch off `opencode/big-pickle`: `trial/grid-fit-splines`,
  `trial/grid-fit-perline`, `trial/grid-fit-surface`.
- Each branch: implement as a **self-contained probe** (in `/tmp/opencode`,
  linking the existing detector lib) that loads the real photo, runs the
  detector, fits, snaps, and reports the metric — plus a small unit test in the
  repo where the model has a clean analytic test.
- Start with Approach 1; iterate to green on its synthetic validation before
  moving to the next branch.

## Deliverable

A comparison report (saved under `docs/superpowers/reports/2026-09-04-grid-fit-comparison.md`)
tabulating the metric for the baseline (current) and all three approaches on
the real photo and the synthetic control, with a recommendation and any
trade-offs (e.g. "Approach 1 best but Approach 3 simpler; recommend X").

## Out of scope for now

- Promoting the winning fit into the shipped `detect()` pipeline and unit-test
  suite until the report is reviewed and a winner chosen.
