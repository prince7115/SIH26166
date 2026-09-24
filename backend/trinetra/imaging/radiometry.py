"""Contrast stretching, local contrast and the normalised features used for correlation."""
import cv2
import numpy as np
from skimage.exposure import match_histograms


def to_uint8(img, lo_p=1.0, hi_p=99.0, mask=None):
    """Percentile stretch to 8 bit; percentiles come from the finite pixels inside mask."""
    a = np.asarray(img, dtype=np.float32)
    sel = np.isfinite(a)
    if mask is not None:
        sel &= np.asarray(mask, dtype=bool)
    finite = a[sel]
    if finite.size < 8:
        finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(finite, lo_p), np.percentile(finite, hi_p)
    if hi <= lo:
        hi = lo + 1.0
    a = np.nan_to_num(a, nan=lo)
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def stretch_to_uint8(a, lo_pct=2.0, hi_pct=98.0, sample_rows=2000):
    """Percentile stretch of a (possibly 16-bit, memory-mapped) raster to 8 bit.

    Percentiles come from a row sample with zeros (strip padding) excluded, and the output is
    written in blocks so a large memmap never has to fit in RAM as float.
    """
    if a.dtype == np.uint8:
        return np.asarray(a)
    step = max(1, a.shape[0] // sample_rows)
    s = np.asarray(a[::step]).ravel()
    s = s[s > 0]
    if s.size == 0:
        return np.zeros(a.shape, np.uint8)
    lo, hi = np.percentile(s, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    out = np.empty(a.shape, np.uint8)
    for r0 in range(0, a.shape[0], 4096):
        r1 = min(a.shape[0], r0 + 4096)
        blk = (np.asarray(a[r0:r1], np.float32) - lo) * (255.0 / (hi - lo))
        out[r0:r1] = np.clip(blk, 0, 255).astype(np.uint8)
    return out


def apply_clahe(img_u8, clip_limit, tile=(8, 8)):
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile).apply(img_u8)


def grad_mag_u8(img_u8, sigma=1.2):
    f = np.asarray(img_u8, dtype=np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), sigma)
    return to_uint8(g, 1, 99)


def texture_score(img_u8):
    """Laplacian variance; 0 for images too small to judge."""
    a = np.asarray(img_u8, np.float32)
    if a.size < 64:
        return 0.0
    return float(cv2.Laplacian(a, cv2.CV_32F, ksize=3).var())


def prep_pair(ohrc_raw, ref_raw, mask=None):
    """Plain 8-bit stretches of both images, plus CLAHE versions with the reference
    histogram-matched to the OHRC (the input the keypoint matcher sees)."""
    o_pc, n_pc = to_uint8(ohrc_raw, mask=mask), to_uint8(ref_raw)
    o_sg = apply_clahe(o_pc, 2.0)
    n_sg = apply_clahe(n_pc, 4.0)
    n_sg = np.clip(match_histograms(n_sg.astype(np.float32), o_sg.astype(np.float32)), 0, 255).astype(np.uint8)
    return o_pc, n_pc, o_sg, n_sg


def zscore(x):
    x = np.asarray(x, np.float32)
    return (x - x.mean()) / (x.std() + 1e-9)


def zscore_grad(img_u8):
    """Standardised gradient magnitude: the feature most ZNCC matching runs on."""
    return zscore(grad_mag_u8(img_u8))


def zscore_intensity(img_u8):
    """Standardised, lightly blurred brightness."""
    return zscore(cv2.GaussianBlur(np.asarray(img_u8, np.float32), (0, 0), 1.0))


def phase_randomize(img, seed=0):
    """Same amplitude spectrum with random phases: an image with the texture statistics of img
    but no structure, used as the null for correlation peaks."""
    F = np.fft.fft2(np.asarray(img, np.float32))
    ph = np.random.default_rng(seed).uniform(-np.pi, np.pi, F.shape)
    return to_uint8(np.real(np.fft.ifft2(np.abs(F) * np.exp(1j * ph))))
