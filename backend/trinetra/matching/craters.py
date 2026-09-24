"""Crater-anchored correspondences at the coarse level."""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.feature import blob_log

from .dense import parabolic_peak
from ..geometry.transforms import apply_h, warp
from ..imaging.radiometry import grad_mag_u8, phase_randomize, zscore_grad


def detect_craters(img_u8, mask, min_r=3.0, max_r=30.0, max_n=600):
    """Crater candidates (x, y, r) from LoG blobs on smoothed gradient magnitude.

    Rim and walls form a ring of strong gradient whatever the sun angle, so the same crater is
    found in both images. Candidates are ranked by local contrast.
    """
    g = cv2.GaussianBlur(grad_mag_u8(img_u8).astype(np.float32), (0, 0), 1.5) / 255.0
    g[~mask] = 0.0
    blobs = blob_log(g, min_sigma=min_r / np.sqrt(2), max_sigma=max_r / np.sqrt(2),
                     num_sigma=8, threshold=0.05, overlap=0.5)
    if len(blobs) == 0:
        return np.zeros((0, 3), np.float32)
    ys, xs, r = blobs[:, 0], blobs[:, 1], blobs[:, 2] * np.sqrt(2)
    h, w = mask.shape
    keep = []
    for i in range(len(blobs)):
        x, y, rr = int(xs[i]), int(ys[i]), r[i]
        m = int(np.ceil(2 * rr)) + 2
        if x - m < 0 or y - m < 0 or x + m >= w or y + m >= h:
            continue
        if not mask[y - m:y + m, x - m:x + m].all():
            continue
        keep.append((xs[i], ys[i], rr, float(g[y - m:y + m, x - m:x + m].std())))
    if not keep:
        return np.zeros((0, 3), np.float32)
    keep.sort(key=lambda t: -t[3])
    return np.asarray([k[:3] for k in keep[:max_n]], np.float32)


def crater_matches(ohrc_c, ref_c, H_coarse, min_ncc=0.3, min_margin=2.0):
    """Craters detected in the prior-warped OHRC and in the reference, paired by position and
    size, then refined by gradient ZNCC with a parabolic sub-pixel step.

    Returns (pts_ohrc, pts_ref, scores, stats), points in coarse pixels.
    """
    stats = {"craters_ohrc": 0, "craters_ref": 0, "paired": 0, "matched": 0}
    empty = (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, np.float32), stats)
    h, w = ref_c.shape[:2]
    Hc = np.asarray(H_coarse, np.float64)
    warped = warp(ohrc_c, Hc, (h, w))
    valid = warp(np.full(ohrc_c.shape, 255, np.uint8), Hc, (h, w), cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8)) > 0
    if valid.sum() < 2000:
        return empty

    cw = detect_craters(warped, valid)
    cr = detect_craters(ref_c, valid)
    stats["craters_ohrc"], stats["craters_ref"] = int(len(cw)), int(len(cr))
    if len(cw) < 3 or len(cr) < 3:
        return empty

    # Pair each reference crater with the nearest OHRC crater of similar size.
    tree = cKDTree(cr[:, :2])
    best_for_ref = {}
    for i, (x, y, r) in enumerate(cw):
        d, j = tree.query([x, y], k=min(5, len(cr)), distance_upper_bound=max(12.0, 2.5 * r))
        for dj, jj in zip(np.atleast_1d(d), np.atleast_1d(j)):
            if not np.isfinite(dj):
                break
            if 0.67 <= cr[jj, 2] / r <= 1.5:
                if jj not in best_for_ref or dj < best_for_ref[jj][1]:
                    best_for_ref[jj] = (i, dj)
                break
    pairs = [(i, j) for j, (i, _d) in best_for_ref.items()]
    stats["paired"] = len(pairs)
    if len(pairs) < 3:
        return empty

    a_z, b_z = zscore_grad(warped), zscore_grad(ref_c)
    b_null = zscore_grad(phase_randomize(ref_c, seed=0))
    pw, pr, sc = [], [], []
    for i, j in pairs:
        x, y, r = cw[i]
        xr, yr = cr[j, 0], cr[j, 1]
        s = int(max(12, 2 * r))
        x0, y0 = int(round(x)) - s, int(round(y)) - s
        if x0 < 0 or y0 < 0 or x0 + 2 * s > w or y0 + 2 * s > h:
            continue
        if not valid[y0:y0 + 2 * s, x0:x0 + 2 * s].all():
            continue
        tpl = np.ascontiguousarray(a_z[y0:y0 + 2 * s, x0:x0 + 2 * s])
        m = s // 2 + 6
        sx0, sy0 = int(round(xr)) - s - m, int(round(yr)) - s - m
        sx1, sy1 = sx0 + 2 * s + 2 * m, sy0 + 2 * s + 2 * m
        if sx0 < 0 or sy0 < 0 or sx1 > w or sy1 > h:
            continue
        try:
            res = cv2.matchTemplate(b_z[sy0:sy1, sx0:sx1], tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            null = cv2.minMaxLoc(cv2.matchTemplate(b_null[sy0:sy1, sx0:sx1], tpl, cv2.TM_CCOEFF_NORMED))[1]
        except cv2.error:
            continue
        if mx < min_ncc or mx / max(null, 1e-6) < min_margin:
            continue
        mx0, my0 = sx0 + loc[0], sy0 + loc[1]
        sdx, sdy = parabolic_peak(res, loc)
        pw.append([x0 + s, y0 + s])
        pr.append([mx0 + s + sdx, my0 + s + sdy])
        sc.append(float(mx))
    if len(pw) < 3:
        return empty
    pw, pr, sc = np.asarray(pw, np.float64), np.asarray(pr, np.float64), np.asarray(sc, np.float32)

    # The prior is already close, so a genuine correction is smooth; a wrong pairing sticks out.
    d = pr - pw
    med = np.median(d, axis=0)
    dev = np.linalg.norm(d - med, axis=1)
    tol = max(3.0, 3.0 * 1.4826 * float(np.median(dev)))
    keep = dev <= tol
    if keep.sum() < 3:
        return empty
    pw, pr, sc = pw[keep], pr[keep], sc[keep]
    stats["matched"] = int(len(pw))
    pts_ohrc = apply_h(np.linalg.inv(Hc), pw)
    return pts_ohrc.astype(np.float32), pr.astype(np.float32), sc, stats
