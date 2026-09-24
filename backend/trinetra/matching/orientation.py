"""Orientation diagnostics: flip/scale search by ZNCC and a log-polar rotation estimate."""
import cv2
import numpy as np

from ..imaging.radiometry import phase_randomize, zscore_grad
from ..imaging.resample import pad_to_square


def _best_ncc(A_z, A_img, B, scales):
    best = (-1.0, None, None)
    for s in scales:
        h, w = int(round(B.shape[0] * s)), int(round(B.shape[1] * s))
        if h < 24 or w < 16 or h > 4000 or w > 4000:
            continue
        Bs = cv2.resize(B, (w, h), interpolation=cv2.INTER_CUBIC)
        if h <= A_img.shape[0] and w <= A_img.shape[1]:
            search_z, tpl_z = A_z, zscore_grad(Bs)
            search_shape, tpl_shape = A_img.shape, (h, w)
        elif A_img.shape[0] <= h and A_img.shape[1] <= w:
            search_z, tpl_z = zscore_grad(Bs), A_z
            search_shape, tpl_shape = (h, w), A_img.shape
        else:
            continue
        if (tpl_shape[0] * tpl_shape[1]) / float(search_shape[0] * search_shape[1]) < 0.35:
            continue
        try:
            res = cv2.matchTemplate(search_z, tpl_z, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (float(mx), round(float(s), 3), loc)
    return best


def ncc_orientation_search(a_u8, b_u8, scales=None, max_side=400):
    """Best ZNCC of b against a for each flip over a range of scales.

    Returns (rows, noise_floor): rows are (score, flip name, cv2 flip code, scale, location),
    best first; the floor is the best score reached by phase-randomised copies of b.
    """
    scales = np.arange(0.80, 1.26, 0.02) if scales is None else scales
    k = min(1.0, max_side / max(max(a_u8.shape[:2]), max(b_u8.shape[:2])))

    def shrink(x):
        return cv2.resize(x, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1.0 else x
    A, B = shrink(a_u8), shrink(b_u8)
    A_z = zscore_grad(A)
    rows = []
    for name, code, Bv in (("no flip", None, B), ("horizontal", 1, B[:, ::-1].copy()),
                           ("vertical", 0, B[::-1, :].copy()), ("both", -1, B[::-1, ::-1].copy())):
        mx, sc, loc = _best_ncc(A_z, A, Bv, scales)
        if sc is not None:
            rows.append((mx, name, code, sc, loc))
    rows.sort(key=lambda t: -t[0])
    floor = max([_best_ncc(A_z, A, phase_randomize(B, seed=q), scales)[0] for q in range(3)] or [1e-6])
    return rows, max(float(floor), 1e-6)


def estimate_rotation_deg(img_a, img_b, canvas=512):
    """Rotation between two images from log-polar phase correlation of their FFT magnitudes.
    Returns (degrees, response); the sign convention comes from rotation_sign."""
    a = cv2.resize(pad_to_square(np.asarray(img_a, np.float32))[0], (canvas, canvas), interpolation=cv2.INTER_AREA)
    b = cv2.resize(pad_to_square(np.asarray(img_b, np.float32))[0], (canvas, canvas), interpolation=cv2.INTER_AREA)
    win = cv2.createHanningWindow((canvas, canvas), cv2.CV_32F)
    a, b = a * win, b * win

    def log_polar_mag(img):
        mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img)))).astype(np.float32)
        return cv2.warpPolar(mag, (canvas, canvas), (canvas / 2.0, canvas / 2.0), canvas / 2.0,
                             cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR)
    (_, shift_y), response = cv2.phaseCorrelate(log_polar_mag(a), log_polar_mag(b))
    return float(shift_y * (360.0 / canvas)), float(response)


def rotation_sign(img_u8, test_deg=7.0):
    """Sign convention of estimate_rotation_deg, found by recovering a known rotation of img.
    Returns (sign, recovered degrees, the rotated test image)."""
    h, w = img_u8.shape
    rotated = cv2.warpAffine(img_u8, cv2.getRotationMatrix2D((w / 2, h / 2), test_deg, 1.0), (w, h))
    recovered, _ = estimate_rotation_deg(img_u8, rotated)
    sign = -1.0 if abs(recovered - test_deg) <= abs(recovered + test_deg) else 1.0
    return sign, recovered, rotated
