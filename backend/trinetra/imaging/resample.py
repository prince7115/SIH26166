"""Resampling between native and working ground sample distances (GSD, metres per pixel)."""
import cv2
import numpy as np


def level_view(raster, native_gsd, target_gsd):
    """Whole raster at target_gsd: integer decimation first, then a resize."""
    f = native_gsd / target_gsd
    step = max(1, int(1.0 / f / 2)) if f < 1 else 1
    small = np.asarray(raster[::step, ::step])
    f2 = (native_gsd * step) / target_gsd
    if abs(f2 - 1.0) < 1e-6:
        return small
    return cv2.resize(small, None, fx=f2, fy=f2,
                      interpolation=cv2.INTER_AREA if f2 < 1 else cv2.INTER_LINEAR)


def level_view_antialiased(raster, native_gsd, target_gsd, band_rows=4096):
    """Whole uint8 raster at target_gsd, area-averaged band by band.

    level_view's decimation aliases fine texture into noise; harmless at NAC's ~3x scale gap,
    destructive at TMC's ~19x.
    """
    f = native_gsd / target_gsd
    h, w = raster.shape[:2]
    if abs(f - 1.0) < 1e-6:
        return np.asarray(raster)
    out_h, out_w = max(1, int(round(h * f))), max(1, int(round(w * f)))
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR
    if raster.dtype != np.uint8:
        raise TypeError(f"level_view_antialiased needs uint8, got {raster.dtype}.")
    out = np.zeros((out_h, out_w), np.uint8)
    for r0 in range(0, h, band_rows):
        r1 = min(h, r0 + band_rows)
        y0, y1 = int(round(r0 * f)), min(out_h, int(round(r1 * f)))
        if y1 <= y0:
            continue
        out[y0:y1] = cv2.resize(np.asarray(raster[r0:r1]), (out_w, y1 - y0), interpolation=interp)
    return out


def level_tile(raster, native_gsd, target_gsd, r0, c0, h, w):
    """An h x w tile at target_gsd whose top-left is (r0, c0) in target pixels.

    Returns (tile, (x0, y0) of the tile in target pixels), or (None, None) if it is too small.
    """
    s = target_gsd / native_gsd
    R0, C0 = max(0, int(np.floor(r0 * s))), max(0, int(np.floor(c0 * s)))
    R1 = min(raster.shape[0], int(np.ceil((r0 + h) * s)))
    C1 = min(raster.shape[1], int(np.ceil((c0 + w) * s)))
    if R1 - R0 < 8 * s or C1 - C0 < 8 * s:
        return None, None
    patch = np.asarray(raster[R0:R1, C0:C1])
    oh, ow = max(1, int(round((R1 - R0) / s))), max(1, int(round((C1 - C0) / s)))
    return cv2.resize(patch, (ow, oh), interpolation=cv2.INTER_AREA), (C0 / s, R0 / s)


def level_shape(crop_shape, native_gsd, target_gsd):
    k = native_gsd / target_gsd
    return (int(round(crop_shape[0] * k)), int(round(crop_shape[1] * k)))


def pixels_at(shape, native_gsd, target_gsd):
    """Pixel count of a native-resolution shape once resampled to target_gsd."""
    return (shape[0] * native_gsd / target_gsd) * (shape[1] * native_gsd / target_gsd)


def pad_to_square(img):
    """Zero-pad to a centred square. Returns (square, x_offset, y_offset)."""
    h, w = img.shape[:2]
    n = max(h, w)
    if n == h == w:
        return img, 0, 0
    out = np.zeros((n, n), img.dtype)
    ox, oy = (n - w) // 2, (n - h) // 2
    out[oy:oy + h, ox:ox + w] = img
    return out, ox, oy
