"""Guarded model fitting: similarity, then affine, then homography, each only if it keeps the inliers."""
import cv2
import numpy as np

from ..geometry.transforms import apply_h, to_h

_USAC = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)


def hull_coverage(pts, frame_shape):
    """Area of the points' convex hull as a fraction of the frame."""
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(pts)) / (frame_shape[0] * frame_shape[1]))


def is_degenerate(pts, min_minor_axis_px=5.0):
    """True if there are fewer than 3 points or their spread is thinner than min_minor_axis_px."""
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return True
    ev = np.linalg.eigvalsh(np.cov((pts - pts.mean(0)).T) + 1e-12 * np.eye(2))
    return bool(np.sqrt(max(ev.min(), 0.0)) < min_minor_axis_px)


def robust_fit(src, dst, frame_shape, thresh=3.0, min_minor_axis_px=5.0):
    """Richest model the correspondences support.

    Affine needs 20+ similarity inliers; a homography needs 40+ inliers covering 10% of the
    frame. A richer model replaces a simpler one only if it keeps at least as many inliers.
    Returns (H, inlier_mask, model name) or (None, None, None).
    """
    src = np.asarray(src, np.float32).reshape(-1, 1, 2)
    dst = np.asarray(dst, np.float32).reshape(-1, 1, 2)
    if len(src) < 4:
        return None, None, None
    best = (None, None, None)
    M, m = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=thresh,
                                       maxIters=20000, confidence=0.999)
    if M is not None and m is not None:
        best = (to_h(M), m.ravel().astype(bool), "similarity")
    if best[1] is not None and best[1].sum() >= 20:
        M, m = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=thresh,
                                    maxIters=20000, confidence=0.999)
        if M is not None and m is not None and m.ravel().sum() >= best[1].sum():
            best = (to_h(M), m.ravel().astype(bool), "affine")
    if best[1] is not None and best[1].sum() >= 40:
        if hull_coverage(dst.reshape(-1, 2)[best[1]], frame_shape) >= 0.10:
            H, m = cv2.findHomography(src, dst, _USAC, thresh, maxIters=50000, confidence=0.9995)
            if H is not None and m is not None and m.ravel().sum() >= best[1].sum():
                best = (H, m.ravel().astype(bool), "homography")
    H, mask, name = best
    if H is None:
        return None, None, None
    if is_degenerate(dst.reshape(-1, 2)[mask], min_minor_axis_px=min_minor_axis_px):
        return None, None, None
    return H, mask, name


def residual_scale(src, dst, frame_shape, t0=100.0, iters=3):
    """RANSAC threshold from the MAD of residuals: start loose, refit with 2.5 sigma, repeat."""
    t = t0
    for _ in range(iters):
        H, _, _ = robust_fit(src, dst, frame_shape, thresh=t, min_minor_axis_px=0.0)
        if H is None:
            return None
        r = np.linalg.norm(apply_h(H, src) - dst, axis=1)
        sigma = 1.4826 * float(np.median(np.abs(r - np.median(r))))
        t = float(max(3.0, 2.5 * max(sigma, 1e-3)))
    return t
