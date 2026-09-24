"""Accuracy measures and the trust verdict reported in step 18."""
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

from ..geometry.transforms import apply_h
from ..matching.fitting import robust_fit

TRUST_MIN_INLIERS = 15
TRUST_MIN_COVERAGE = 0.05
TRUST_MAX_HELDOUT_PX = 5.0
TRUST_MIN_NMI_RATIO = 1.5


def rmse(H, src, dst):
    return float(np.sqrt(np.mean(np.sum((apply_h(H, src) - dst) ** 2, axis=1))))


def random_heldout_rmse(src, dst, frame_shape, thresh, splits=5, train_frac=0.7):
    """Mean test RMSE over random 70/30 splits, each refit on its training points."""
    rng = np.random.default_rng(0)
    held = []
    for _ in range(splits):
        idx = rng.permutation(len(src))
        cut = int(train_frac * len(idx))
        tr, te = idx[:cut], idx[cut:]
        if len(tr) < 8 or len(te) < 4:
            continue
        H_tr, _, _ = robust_fit(src[tr], dst[tr], frame_shape, thresh=thresh, min_minor_axis_px=5.0)
        if H_tr is None:
            continue
        held.append(float(np.sqrt(np.mean(np.linalg.norm(apply_h(H_tr, src[te]) - dst[te], axis=1) ** 2))))
    return float(np.mean(held)) if held else None


def spatial_heldout(src, dst, frame_shape, thresh, n_bands=10, n_folds=5):
    """Held-out RMSE with test points grouped by position along the strip.

    Points fall into n_bands bands along the strip's long axis, band k going to fold
    k % n_folds, so each fold tests interleaved stretches of ground rather than random
    neighbours of training points. The refits skip the collinearity check, which rejects every
    subset of a thin strip. Returns (pooled RMSE or None, folds used).
    """
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    if len(src) < 12:
        return None, 0
    axis = 1 if frame_shape[0] >= frame_shape[1] else 0      # dst is (x, y)
    coord = dst[:, axis]
    edges = np.quantile(coord, np.linspace(0, 1, n_bands + 1))
    band = np.clip(np.searchsorted(edges, coord, side="right") - 1, 0, n_bands - 1)
    fold = band % n_folds
    errs, used = [], 0
    for k in range(n_folds):
        te = fold == k
        tr = ~te
        if tr.sum() < 8 or te.sum() < 3:
            continue
        H, _, _ = robust_fit(src[tr], dst[tr], frame_shape, thresh=thresh, min_minor_axis_px=0.0)
        if H is None:
            continue
        errs.append(np.linalg.norm(apply_h(H, src[te]) - dst[te], axis=1))
        used += 1
    if not errs:
        return None, 0
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e ** 2))), used


def _sampled_nmi(a, b):
    """NMI of 32-level quantised a and b over up to 200k pixels valid in both."""
    valid = (a > 0) & (b > 0)
    n = int(valid.sum())
    idx = np.random.default_rng(0).choice(n, size=min(200_000, n), replace=False)
    return float(normalized_mutual_info_score((a[valid] // 8)[idx], (b[valid] // 8)[idx]))


def nmi_with_baseline(warped, ref):
    """NMI of the aligned pair, and of the same pair deliberately misaligned by (37, 29) px.
    (None, None) if the overlap is under 1000 pixels."""
    if ((warped > 0) & (ref > 0)).sum() <= 1000:
        return None, None
    return _sampled_nmi(warped, ref), _sampled_nmi(np.roll(warped, (37, 29), axis=(0, 1)), ref)


def trust_verdict(n_inliers, coverage, heldout_px, nmi, nmi_base):
    """Trusted only if every check passes. Returns (trusted, checks)."""
    ratio = (nmi / nmi_base) if (nmi is not None and nmi_base) else None
    checks = [
        {"name": "Inliers", "value": int(n_inliers), "limit": f"≥ {TRUST_MIN_INLIERS}",
         "pass": n_inliers >= TRUST_MIN_INLIERS},
        {"name": "Coverage", "value": f"{coverage * 100:.1f}%", "limit": f"≥ {TRUST_MIN_COVERAGE * 100:.0f}%",
         "pass": coverage >= TRUST_MIN_COVERAGE},
        {"name": "Held-out RMSE", "value": None if heldout_px is None else f"{heldout_px:.2f} px",
         "limit": f"≤ {TRUST_MAX_HELDOUT_PX:g} px",
         "pass": heldout_px is not None and heldout_px <= TRUST_MAX_HELDOUT_PX},
        {"name": "NMI vs misaligned", "value": None if ratio is None else f"{ratio:.2f}×",
         "limit": f"≥ {TRUST_MIN_NMI_RATIO:g}×", "pass": ratio is not None and ratio >= TRUST_MIN_NMI_RATIO},
    ]
    return all(c["pass"] for c in checks), checks
