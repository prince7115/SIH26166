"""Dense ZNCC matching of a pre-warped tile pair on a grid of large patches."""
import cv2
import numpy as np

from . import empty_matches
from ..imaging.radiometry import phase_randomize, zscore_grad, zscore_intensity

FEATURES = {"grad": zscore_grad, "intensity": zscore_intensity}


def match_dense(warped_u8, ref_u8, valid_mask, subdiv=3, min_margin=2.0,
                feature="grad", invert=False, subpixel="phase"):
    """Patch correspondences between a warped OHRC tile and its reference tile.

    The valid area is cut into subdiv x subdiv cells; the central 85% of each is located in the
    reference by ZNCC and kept only if its peak beats a phase-randomised null by min_margin.

    feature   "grad" (gradient magnitude) or "intensity" (brightness). Gradient magnitude cannot
              tell a lit slope from a shadowed one; brightness can, and invert=True negates the
              OHRC template to match ground whose lit and shadowed faces swapped.
    subpixel  "phase": windowed phase correlation between the matched patches (the fine stage);
              "parabola": parabola through the correlation peak (the tile rescues).

    Returns (points_warped, points_ref, scores), (x, y) in tile pixels.
    """
    h, w = ref_u8.shape[:2]
    ys, xs = np.where(valid_mask)
    if len(ys) < 400:
        return empty_matches()
    Y0, Y1, X0, X1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    ph, pw = (Y1 - Y0) // subdiv, (X1 - X0) // subdiv
    if ph < 64 or pw < 64:
        return empty_matches()

    feat = FEATURES[feature]
    b_z = feat(ref_u8)
    a_z = -feat(warped_u8) if invert else feat(warped_u8)
    nulls = None
    ka, kb, sc = [], [], []
    for iy in range(subdiv):
        for ix in range(subdiv):
            th, tw = int(ph * 0.85), int(pw * 0.85)
            ty = Y0 + iy * ph + (ph - th) // 2
            tx = X0 + ix * pw + (pw - tw) // 2
            if valid_mask[ty:ty + th, tx:tx + tw].mean() < 0.9:
                continue
            tpl = a_z[ty:ty + th, tx:tx + tw]
            if tpl.shape[0] > b_z.shape[0] or tpl.shape[1] > b_z.shape[1]:
                continue
            try:
                res = cv2.matchTemplate(b_z, tpl, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(res)
                if nulls is None:
                    nulls = [feat(phase_randomize(ref_u8, seed=q)) for q in range(2)]
                null = max(cv2.minMaxLoc(cv2.matchTemplate(n, tpl, cv2.TM_CCOEFF_NORMED))[1] for n in nulls)
            except cv2.error:
                continue
            if mx / max(null, 1e-6) < min_margin:
                continue

            cx, cy = tx + tw / 2.0, ty + th / 2.0
            if subpixel == "phase":
                dx, dy = loc[0] - tx, loc[1] - ty
                sy0, sx0 = int(ty + dy), int(tx + dx)
                if sy0 < 0 or sx0 < 0 or sy0 + th > h or sx0 + tw > w:
                    continue
                sdx, sdy = _phase_offset(a_z[ty:ty + th, tx:tx + tw], b_z[sy0:sy0 + th, sx0:sx0 + tw], th, tw)
                kb.append([cx + dx + sdx, cy + dy + sdy])
            else:
                sdx, sdy = parabolic_peak(res, loc)
                dx, dy = loc[0] - tx + sdx, loc[1] - ty + sdy
                kb.append([cx + dx, cy + dy])
            ka.append([cx, cy])
            sc.append(float(min(1.0, mx)))
    if not ka:
        return empty_matches()
    return np.asarray(ka, np.float32), np.asarray(kb, np.float32), np.asarray(sc, np.float32)


def _phase_offset(a, b, h, w):
    """Sub-pixel shift of b against a by Hann-windowed phase correlation; 0 if above 2 px."""
    try:
        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        (sdx, sdy), _ = cv2.phaseCorrelate(np.ascontiguousarray(a) * win, np.ascontiguousarray(b) * win)
        if abs(sdx) > 2 or abs(sdy) > 2:
            sdx = sdy = 0.0
    except cv2.error:
        sdx = sdy = 0.0
    return sdx, sdy


def parabolic_peak(res, loc):
    """Sub-pixel offset of a correlation peak from a parabola through its neighbours.

    On smooth, downsampled tiles this proved exact where windowed phase correlation returned
    corrections of the wrong sign, which is why the rescues use it.
    """
    x, y = loc
    h, w = res.shape

    def _vertex(m1, c, p1):
        den = m1 - 2.0 * c + p1
        return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (m1 - p1) / den, -0.5, 0.5))
    sx = _vertex(res[y, x - 1], res[y, x], res[y, x + 1]) if 0 < x < w - 1 else 0.0
    sy = _vertex(res[y - 1, x], res[y, x], res[y + 1, x]) if 0 < y < h - 1 else 0.0
    return sx, sy
