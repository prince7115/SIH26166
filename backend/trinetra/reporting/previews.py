"""Per-step preview images.

Coarse frames are long thin strips; shrunk whole to 800 px they become slivers in which markers
vanish and a few-pixel misalignment is invisible. So each preview is cropped to the region that
matters, resized, and annotated afterwards. The website scrolls along a long strip.
"""
import cv2
import numpy as np

from ..imaging.radiometry import to_uint8

PREVIEW_SIDE = 800
STRIP_SHORT_SIDE = 360   # a long strip keeps at least this many px across ...
STRIP_LONG_MAX = 3200    # ... unless that would make it longer than this


def _crop_box(mask, margin=0.06):
    ys, xs = np.where(mask)
    h, w = mask.shape[:2]
    if len(ys) == 0:
        return 0, h, 0, w
    my, mx = int((ys.max() - ys.min()) * margin) + 8, int((xs.max() - xs.min()) * margin) + 8
    return max(0, ys.min() - my), min(h, ys.max() + my + 1), max(0, xs.min() - mx), min(w, xs.max() + mx + 1)


def _resize(img):
    """Preview-sized copy of img and the scale applied."""
    h, w = img.shape[:2]
    k = PREVIEW_SIDE / max(h, w)
    if min(h, w) * k < STRIP_SHORT_SIDE:
        k = min(STRIP_SHORT_SIDE / min(h, w), STRIP_LONG_MAX / max(h, w))
    interp = cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (max(1, int(round(w * k))), max(1, int(round(h * k)))), interpolation=interp), k


def _caption(bgr, text):
    bar = np.full((28, bgr.shape[1], 3), 20, np.uint8)
    cv2.putText(bar, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 225, 225), 1, cv2.LINE_AA)
    return np.vstack([bar, bgr])


def overlay_preview(warped, ref, caption):
    """False-colour overlay on the OHRC footprint: OHRC magenta, reference green. Aligned ground
    reads grey; misalignment shows as coloured fringes."""
    r0, r1, c0, c1 = _crop_box(warped > 0)
    w, rf = warped[r0:r1, c0:c1], ref[r0:r1, c0:c1]
    has = w > 0
    bgr = np.dstack([np.where(has, w, rf), rf, np.where(has, w, rf)]).astype(np.uint8)
    return _caption(_resize(bgr)[0], caption)


def points_preview(base, groups, caption, radius=3):
    """Points drawn after resizing so they stay visible, cropped to where the points are.

    groups: [(points_xy_in_base, bgr), ...]; a bgr may be an (N, 3) array of per-point colours.
    """
    pts_all = [np.asarray(p, np.float64).reshape(-1, 2) for p, _ in groups if len(p)]
    if pts_all:
        allp = np.vstack(pts_all)
        mask = np.zeros(base.shape[:2], bool)
        xs = np.clip(allp[:, 0].astype(int), 0, mask.shape[1] - 1)
        ys = np.clip(allp[:, 1].astype(int), 0, mask.shape[0] - 1)
        mask[ys, xs] = True
        r0, r1, c0, c1 = _crop_box(mask)
    else:
        r0, r1, c0, c1 = 0, base.shape[0], 0, base.shape[1]
    img, k = _resize(cv2.cvtColor(to_uint8(base[r0:r1, c0:c1]), cv2.COLOR_GRAY2BGR))
    for pts, color in groups:
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        cols = np.asarray(color).reshape(-1, 3) if np.ndim(color) == 2 else None
        for i, (x, y) in enumerate(pts):
            c = tuple(int(v) for v in (cols[i] if cols is not None else color))
            cv2.circle(img, (int((x - c0) * k), int((y - r0) * k)), radius, c, -1, cv2.LINE_AA)
    return _caption(img, caption)


def side_by_side(a, b, height=256):
    """Two rasters stretched to 8 bit, scaled to a common height and placed side by side."""
    def _fit(x):
        x = to_uint8(x)
        if x.shape[0] <= 4:
            return x
        return cv2.resize(x, (max(1, int(height * x.shape[1] / max(x.shape[0], 1))), height),
                          interpolation=cv2.INTER_AREA)
    a, b = _fit(a), _fit(b)
    if a.shape[0] != b.shape[0]:
        h = max(a.shape[0], b.shape[0])
        a = np.pad(a, ((0, h - a.shape[0]), (0, 0)))
        b = np.pad(b, ((0, h - b.shape[0]), (0, 0)))
    return np.hstack([a, b])


def match_lines(a, b, pts_a, pts_b, max_lines=100):
    """a and b side by side with the first max_lines correspondences joined."""
    h = max(a.shape[0], b.shape[0])
    a_pad = np.zeros((h, a.shape[1]), dtype=a.dtype)
    a_pad[:a.shape[0], :] = a
    b_pad = np.zeros((h, b.shape[1]), dtype=b.dtype)
    b_pad[:b.shape[0], :] = b
    vis = cv2.cvtColor(np.hstack([a_pad, b_pad]), cv2.COLOR_GRAY2BGR)
    offset = a.shape[1]
    for i in range(min(max_lines, len(pts_a))):
        p0 = (int(pts_a[i, 0]), int(pts_a[i, 1]))
        p1 = (int(pts_b[i, 0]) + offset, int(pts_b[i, 1]))
        cv2.line(vis, p0, p1, (0, 255, 255), 1)
        cv2.circle(vis, p0, 3, (0, 0, 255), -1)
        cv2.circle(vis, p1, 3, (0, 255, 0), -1)
    return vis
