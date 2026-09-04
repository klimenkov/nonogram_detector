#include "grid_smooth_fit.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/core.hpp>

namespace ng
{
namespace
{

constexpr float kSentinel = -1.0f;

// Fits a degree-<degree> polynomial in <t> to points (t, v) by least squares,
// returning the coefficients (constant first). Uses a scaled index u = t/T so
// powers stay well-conditioned for degree <= 4.
std::vector<double> fit_poly(std::vector<double> const& t,
                             std::vector<double> const& v,
                             int degree,
                             double scale)
{
    int const n = degree + 1;

    auto const base = [&](int k, double ti) {
        return std::pow(ti / scale, k);
    };

    std::vector<double> A(n * n, 0.0), rhs(n, 0.0);
    for (std::size_t i = 0; i < t.size(); ++i)
    {
        double const ui = base(1, t[i]);
        std::vector<double> pw(n, 1.0);
        for (int k = 1; k < n; ++k) pw[k] = pw[k - 1] * ui;
        for (int k = 0; k < n; ++k)
        {
            rhs[k] += v[i] * pw[k];
            for (int l = 0; l < n; ++l)
                A[k * n + l] += pw[k] * pw[l];
        }
    }

    // Solve A w = rhs by Gaussian elimination with partial pivoting.
    for (int col = 0; col < n; ++col)
    {
        int piv = col;
        for (int r = col + 1; r < n; ++r)
            if (std::fabs(A[r * n + col]) > std::fabs(A[piv * n + col]))
                piv = r;
        if (piv != col)
        {
            for (int k = 0; k < n; ++k) std::swap(A[col * n + k], A[piv * n + k]);
            std::swap(rhs[col], rhs[piv]);
        }
        for (int r = col + 1; r < n; ++r)
        {
            double const f = A[r * n + col] / A[col * n + col];
            if (A[col * n + col] == 0.0) continue;
            for (int k = col; k < n; ++k) A[r * n + k] -= f * A[col * n + k];
            rhs[r] -= f * rhs[col];
        }
    }
    std::vector<double> w(n, 0.0);
    for (int r = n - 1; r >= 0; --r)
    {
        double s = rhs[r];
        for (int k = r + 1; k < n; ++k) s -= A[r * n + k] * w[k];
        w[r] = s / A[r * n + r];
    }
    return w;
}

// Fits each coefficient band (one value per primary-index line) as a low-order
// polynomial in the family index, so the per-line curves vary smoothly across
// their family index. <family> is the primary index (0..P-1) at which the band
// value <val> was observed (one entry per fitting line, in primary order);
// <prim_scale> = P-1 keeps the polynomial basis well-conditioned. Returns the
// fitted band value for every primary index 0..P-1.
std::vector<double> fit_coeff_band(std::vector<double> const& family,
                                   std::vector<double> const& val,
                                   int degree,
                                   double prim_scale,
                                   int P)
{
    std::vector<double> band(P, 0.0);
    if (family.size() < 2 || degree < 0)
    {
        for (std::size_t i = 0; i < family.size(); ++i) band[family[i]] = val[i];
        return band;
    }
    int const deg = std::min(degree, static_cast<int>(family.size()) - 1);

    // Least-squares fit of val vs u = family/prim_scale, degree <deg>.
    int const n = deg + 1;
    std::vector<double> A(n * n, 0.0), rhs(n, 0.0);
    for (std::size_t i = 0; i < family.size(); ++i)
    {
        double const u = family[i] / prim_scale;
        std::vector<double> pw(n, 1.0);
        for (int k = 1; k < n; ++k) pw[k] = pw[k - 1] * u;
        for (int k = 0; k < n; ++k)
        {
            rhs[k] += val[i] * pw[k];
            for (int l = 0; l < n; ++l) A[k * n + l] += pw[k] * pw[l];
        }
    }
    for (int col = 0; col < n; ++col)
    {
        int piv = col;
        for (int r = col + 1; r < n; ++r)
            if (std::fabs(A[r * n + col]) > std::fabs(A[piv * n + col]))
                piv = r;
        if (piv != col)
        {
            for (int k = 0; k < n; ++k) std::swap(A[col * n + k], A[piv * n + k]);
            std::swap(rhs[col], rhs[piv]);
        }
        for (int r = col + 1; r < n; ++r)
        {
            double const f = A[r * n + col] / A[col * n + col];
            if (A[col * n + col] == 0.0) continue;
            for (int k = col; k < n; ++k) A[r * n + k] -= f * A[col * n + k];
            rhs[r] -= f * rhs[col];
        }
    }
    std::vector<double> w(n, 0.0);
    for (int r = n - 1; r >= 0; --r)
    {
        double s = rhs[r];
        for (int k = r + 1; k < n; ++k) s -= A[r * n + k] * w[k];
        w[r] = s / A[r * n + r];
    }

    for (int p = 0; p < P; ++p)
    {
        double const u = p / prim_scale;
        double s = 0.0, pw = 1.0;
        for (int k = 0; k < n; ++k)
        {
            s += w[k] * pw;
            pw *= u;
        }
        band[p] = s;
    }
    return band;
}

}  // namespace

cv::Mat grid_smooth_fit_approach1(cv::Mat const& cross_locs, int order, int coeff_order)
{
    cv::Mat out = cross_locs.clone();
    if (out.empty() || out.type() != CV_32FC2)
    {
        return out;
    }

    int const R = out.rows;      // number of horizontal lines (index 0..R-1)
    int const C = out.cols;      // number of vertical lines (index 0..C-1)
    if (R < 2 || C < 2)
    {
        return out;
    }

    // ---- x: per-row line x(c) fit, then smooth coefficients across rows ----
    for (int axis = 0; axis < 2; ++axis)
    {
        // Gather for each primary-index line the (secondary, value) samples,
        // ignoring sentinel crossings.
        // primary = row for x (fit per row across cols), and per column for y.
        int const P = (axis == 0) ? out.rows : out.cols;  // number of primary lines
        int const S = (axis == 0) ? out.cols : out.rows;  // secondary index per line

        std::vector<std::vector<double>> coeff(P);  // [primary][k]

        // For each primary line, build a design assuming the secondary runs
        // 0..S-1 (fewer if some are sentinel). Use a shared secondary index so
        // all lines are fit on the same grid (keeps coefficient meaning equal).
        for (int p = 0; p < P; ++p)
        {
            std::vector<double> t, v;
            std::vector<int> valid;
            for (int s = 0; s < S; ++s)
            {
                cv::Point2f const pt = (axis == 0)
                    ? out.at<cv::Point2f>(p, s)
                    : out.at<cv::Point2f>(s, p);
                if (pt.x == kSentinel || pt.y == kSentinel) continue;
                valid.push_back(s);
                t.push_back(static_cast<double>(s));
                v.push_back((axis == 0) ? pt.x : pt.y);
            }
            if (valid.empty())
            {
                continue;
            }
            int const deg = std::min(order, static_cast<int>(valid.size()) - 1);
            if (deg < 1)
            {
                continue;
            }
            coeff[p] = fit_poly(t, v, deg, static_cast<double>(S - 1));
        }

        // Smooth each coefficient band (over the primary index) -- only use
        // primary lines that have a fit.
        int const maxdeg = [&]() {
            int m = 0;
            for (int p = 0; p < P; ++p) m = std::max(m, static_cast<int>(coeff[p].size()));
            return m;
        }();
        for (int k = 0; k < maxdeg; ++k)
        {
            std::vector<double> family, bandval;
            for (int p = 0; p < P; ++p)
            {
                if (static_cast<int>(coeff[p].size()) > k)
                {
                    family.push_back(static_cast<double>(p));
                    bandval.push_back(coeff[p][k]);
                }
            }
            if (family.empty()) continue;
            double const prim_scale = P > 1 ? static_cast<double>(P - 1) : 1.0;
            std::vector<double> const sm =
                fit_coeff_band(family, bandval, coeff_order, prim_scale, P);
            for (int p = 0; p < P; ++p)
            {
                if (coeff[p].empty()) continue;
                if (static_cast<int>(coeff[p].size()) <= k) coeff[p].resize(k + 1, 0.0);
                coeff[p][k] = sm[p];
            }
        }

        // Evaluate the model at every crossing (primary p, secondary s).
        double const smax = S > 1 ? static_cast<double>(S - 1) : 1.0;
        for (int p = 0; p < P; ++p)
        {
            if (coeff[p].empty()) continue;
            for (int s = 0; s < S; ++s)
            {
                double u = s / smax;
                double val = 0.0, pw = 1.0;
                for (int k = 0; k < static_cast<int>(coeff[p].size()); ++k)
                {
                    val += coeff[p][k] * pw;
                    pw *= u;
                }
                if (axis == 0)
                {
                    cv::Point2f& pt = out.at<cv::Point2f>(p, s);
                    pt.x = static_cast<float>(val);
                }
                else
                {
                    cv::Point2f& pt = out.at<cv::Point2f>(s, p);
                    pt.y = static_cast<float>(val);
                }
            }
        }
    }

    return out;
}

cv::Mat grid_smooth_fit_approach2(cv::Mat const& cross_locs, int order)
{
    cv::Mat out = cross_locs.clone();
    if (out.empty() || out.type() != CV_32FC2)
    {
        return out;
    }

    int const R = out.rows;      // number of horizontal lines (index 0..R-1)
    int const C = out.cols;      // number of vertical lines (index 0..C-1)
    if (R < 2 || C < 2)
    {
        return out;
    }

    // Independent per-line fit: no shared/global model, no cross-family
    // smoothing. Each horizontal line is fit as a polynomial in column index
    // (for x) and each vertical line as a polynomial in row index (for y).
    for (int axis = 0; axis < 2; ++axis)
    {
        int const P = (axis == 0) ? out.rows : out.cols;
        int const S = (axis == 0) ? out.cols : out.rows;

        std::vector<std::vector<double>> coeff(P);
        for (int p = 0; p < P; ++p)
        {
            std::vector<double> t, v;
            for (int s = 0; s < S; ++s)
            {
                cv::Point2f const pt = (axis == 0)
                    ? out.at<cv::Point2f>(p, s)
                    : out.at<cv::Point2f>(s, p);
                if (pt.x == kSentinel || pt.y == kSentinel) continue;
                t.push_back(static_cast<double>(s));
                v.push_back((axis == 0) ? pt.x : pt.y);
            }
            if (t.size() < 2)
            {
                continue;
            }
            int const deg = std::min(order, static_cast<int>(t.size()) - 1);
            if (deg < 1)
            {
                continue;
            }
            coeff[p] = fit_poly(t, v, deg, static_cast<double>(S - 1));
        }

        double const smax = S > 1 ? static_cast<double>(S - 1) : 1.0;
        for (int p = 0; p < P; ++p)
        {
            if (coeff[p].empty()) continue;
            for (int s = 0; s < S; ++s)
            {
                double u = s / smax;
                double val = 0.0, pw = 1.0;
                for (int k = 0; k < static_cast<int>(coeff[p].size()); ++k)
                {
                    val += coeff[p][k] * pw;
                    pw *= u;
                }
                if (axis == 0)
                {
                    cv::Point2f& pt = out.at<cv::Point2f>(p, s);
                    pt.x = static_cast<float>(val);
                }
                else
                {
                    cv::Point2f& pt = out.at<cv::Point2f>(s, p);
                    pt.y = static_cast<float>(val);
                }
            }
        }
    }

    return out;
}

}  // namespace ng
