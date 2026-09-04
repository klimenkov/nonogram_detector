#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

namespace ng
{

// Fits a smooth, regular grid to the detected crossing locations and snaps each
// location onto the fitted grid, returning a matrix of the same shape/type.
//
// Approach 1 (this trial branch): separable per-line curves whose line-family
// coefficients are themselves smooth across the family. Concretely, for each
// coordinate the model is built per axis by (a) fitting each grid line as a
// low-order polynomial in the opposite index and (b) fitting each coefficient
// band as a low-order polynomial in the line-family index, so the per-line
// curves vary smoothly across the family. This gives a smooth regular grid
// that follows real sheet curvature without over-fitting scattered individual
// crossings, and guarantees consistent intersections.
//
// <order> is the per-line polynomial degree; <coeff_order> is the degree of
// the across-family polynomial fit applied to each coefficient band (e.g. 2
// captures smooth quadratic lens/sheet curvature in the family direction).
cv::Mat grid_smooth_fit_approach1(cv::Mat const& cross_locs, int order, int coeff_order);

// Approach 2 (baseline for the comparison): fits each grid line independently
// as a low-order polynomial in the opposite index -- a polynomial in column
// index for x of each horizontal line, and a polynomial in row index for y of
// each vertical line. There is no shared/global model and no cross-family
// smoothing, so a line cannot borrow strength from its neighbours. <order> is
// the per-line polynomial degree.
cv::Mat grid_smooth_fit_approach2(cv::Mat const& cross_locs, int order);

// Approach 3: fits x(r,c) and y(r,c) directly as one global bivariate
// polynomial (full total degree <order>) over the rectangular index domain
// mapped to [0,1]^2. Every valid crossing contributes to a single shared
// surface, so rows and columns are coupled rigidly. Very compact and always
// smooth, but real non-square curvature may need a higher order to fit and can
// oscillate. <order> is the bivariate polynomial total degree.
cv::Mat grid_smooth_fit_approach3(cv::Mat const& cross_locs, int order);

}  // namespace ng
