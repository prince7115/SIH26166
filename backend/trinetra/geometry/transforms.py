"""3x3 homogeneous transforms on (x, y) pixels. Homographies map OHRC pixels to reference pixels."""
import cv2
import numpy as np


def mat_translate(tx, ty):
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def mat_scale(sx, sy=None):
    sy = sx if sy is None else sy
    return np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])


def to_h(M23):
    return np.vstack([np.asarray(M23, dtype=np.float64), [0.0, 0.0, 1.0]])


def apply_h(H, pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    p = np.hstack([pts, np.ones((len(pts), 1))])
    q = (np.asarray(H, dtype=np.float64) @ p.T).T
    return q[:, :2] / q[:, 2:3]


def H_at_level(H, gsd_from, gsd_to):
    """The same ground transform expressed between images resampled from gsd_from to gsd_to."""
    k = gsd_from / gsd_to
    return mat_scale(k) @ np.asarray(H, dtype=np.float64) @ mat_scale(1.0 / k)


def warp(img, H, shape, flags=cv2.INTER_LINEAR):
    """img warped by H into a frame of the given (rows, cols) shape, zeros outside."""
    return cv2.warpPerspective(img, np.asarray(H).astype(np.float64), (shape[1], shape[0]),
                               flags=flags, borderValue=0)
