"""Per-tile consensus gate and the optional rescues for tiles whose dense match failed."""
import cv2
import numpy as np

from .dense import match_dense


def tile_consensus(ka, kb, sc, px):
    """The points of a tile must agree on one shift: drop those more than px from the median
    shift, and reject the tile if the median deviation itself exceeds px. None if rejected."""
    if len(ka) < 2:
        return None
    d = kb - ka
    med = np.median(d, axis=0)
    dev = np.linalg.norm(d - med, axis=1)
    if float(np.median(dev)) > px:
        return None
    agree = dev <= px
    if int(agree.sum()) < 2:
        return None
    return ka[agree], kb[agree], sc[agree]


def suppress_shadows(img_u8, valid, rel=0.45, max_frac=0.45, sigma=12.0):
    """Cast shadows replaced by a smooth fill from the surrounding lit ground.

    Shadow edges move with the sun, so they are the gradients most likely to match the wrong
    place. Normalised convolution fills each shadow from nearby lit pixels, removing the edge
    without inventing structure.
    """
    v = img_u8[valid]
    if v.size < 100:
        return img_u8
    thr = rel * float(np.median(v))
    shadow = (img_u8 < thr) & valid
    frac = shadow.sum() / max(valid.sum(), 1)
    if frac < 0.01:
        return img_u8
    if frac > max_frac:
        shadow = (img_u8 < np.quantile(v, max_frac)) & valid
    shadow = cv2.dilate(shadow.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    lit = (valid & ~shadow).astype(np.float32)
    f = img_u8.astype(np.float32)
    num = cv2.GaussianBlur(f * lit, (0, 0), sigma)
    den = cv2.GaussianBlur(lit, (0, 0), sigma)
    fill = np.where(den > 1e-3, num / np.maximum(den, 1e-3), float(np.median(v)))
    out = f.copy()
    out[shadow] = fill[shadow]
    return np.clip(out, 0, 255).astype(np.uint8)


def _subdiv_for(mask):
    """Largest patch grid (3 or 2) whose cells are at least 64 px, or 0."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0
    side = min(ys.max() - ys.min(), xs.max() - xs.min())
    return 3 if side // 3 >= 64 else 2 if side // 2 >= 64 else 0


def _match_rescaled(a, b, mask, s, consensus_px):
    """Dense match with both images shrunk by s; points returned in the original tile frame."""
    h, w = b.shape[:2]
    size = (max(1, int(round(w / s))), max(1, int(round(h / s))))
    a_s = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    b_s = cv2.resize(b, size, interpolation=cv2.INTER_AREA)
    m_s = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    sub = _subdiv_for(m_s)
    if sub == 0:
        return None
    out = tile_consensus(*match_dense(a_s, b_s, m_s, subdiv=sub, subpixel="parabola"), consensus_px / s)
    if out is None:
        return None
    ka, kb, sc = out
    return (ka + 0.5) * s - 0.5, (kb + 0.5) * s - 0.5, sc


def rescue_tile(o_pc, n_pc, vmask, opts, consensus_px):
    """Enabled rescues in order (shadow, rescale, polarity); the first to pass the consensus
    gate wins. Returns ((ka, kb, sc), method) or (None, None)."""
    a, b = o_pc, n_pc
    tag = ""
    if opts.get("shadow_mask"):
        a, b = suppress_shadows(o_pc, vmask), suppress_shadows(n_pc, vmask)
        out = tile_consensus(*match_dense(a, b, vmask, subpixel="parabola"), consensus_px)
        if out is not None:
            return out, "shadow"
        tag = " + shadow"
    if opts.get("tile_rescale"):
        for s in (1.5, 2.0):
            out = _match_rescaled(a, b, vmask, s, consensus_px)
            if out is not None:
                return out, f"rescale ×{s:g}{tag}"
    if opts.get("tile_polarity"):
        for inv in (False, True):
            out = tile_consensus(*match_dense(a, b, vmask, feature="intensity", invert=inv, subpixel="parabola"),
                                 consensus_px)
            if out is not None:
                return out, ("intensity inverted" if inv else "intensity") + tag
    return None, None
