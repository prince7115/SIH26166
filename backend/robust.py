"""Robustness add-ons for the registration engine.

Everything here is opt-in through run options (DEFAULT_OPTIONS). With the defaults the engine
behaves exactly as before, except ECC, which is now gated: it is kept only when it improves
the fine-level alignment score, so its worst case equals "ECC off".

  * sun geometry + difficulty score        (report only)
  * gated ECC                              (ported from sih_final_v5_ohrc_tmc_1.ipynb, cell 15)
  * spatial held-out RMSE + trust verdict  (evaluation only)
  * tile rescues for tiles that failed     (shadow suppression, multi-scale, intensity polarity)
  * crater-anchored matches                (pooled with the dense matches before RANSAC)
  * retry ladder                           (reruns steps 12-14 with rescues switched on)
"""
import os, json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable

import numpy as np
import cv2

from .common import (NS, to_uint8, grad_mag_u8, _zgrad, _phase_randomize, level_tile, apply_h,
                     mat_translate, H_at_level, to_h, robust_fit, _EMPTY)


# ============================================================
# OPTIONS
# ============================================================
DEFAULT_OPTIONS = {
    "ecc_mode": "gated",        # gated | legacy | off
    "matcher": "superglue",     # superglue | lightglue   (coarse stage, steps 8 and 11)
    "shadow_mask": False,       # rescue: suppress shadows in failed tiles
    "tile_rescale": False,      # rescue: retry failed tiles at 1.5x / 2x coarser scale
    "tile_polarity": False,     # rescue: intensity ZNCC, normal and inverted polarity
    "crater_matching": False,   # extra crater-anchored matches pooled before step 13
    "auto_retry": False,        # rerun steps 12-14 with rescues if the result is not trusted
}
ECC_MODES = ("gated", "legacy", "off")
MATCHERS = ("superglue", "lightglue")
RESCUE_KEYS = ("shadow_mask", "tile_rescale", "tile_polarity")

# Each retry switches on more rescues than the one before; at most two retries.
RETRY_LADDER = [
    ("shadow suppression + multi-scale", {"shadow_mask": True, "tile_rescale": True}),
    ("+ intensity polarity + crater matches", {"tile_polarity": True, "crater_matching": True}),
]


def resolve_options(profile_options=None, run_options=None):
    """DEFAULT_OPTIONS <- pipeline profile <- this run. Raises ValueError on bad values."""
    opts = dict(DEFAULT_OPTIONS)
    for src in (profile_options or {}, run_options or {}):
        for k, v in src.items():
            if k not in DEFAULT_OPTIONS:
                raise ValueError(f"Unknown option '{k}'")
            opts[k] = v
    if opts["ecc_mode"] not in ECC_MODES:
        raise ValueError(f"ecc_mode must be one of {ECC_MODES}")
    if opts["matcher"] not in MATCHERS:
        raise ValueError(f"matcher must be one of {MATCHERS}")
    for k in DEFAULT_OPTIONS:
        if isinstance(DEFAULT_OPTIONS[k], bool):
            opts[k] = bool(opts[k])
    return opts


def any_rescue(opts):
    return any(opts.get(k) for k in RESCUE_KEYS)


# ============================================================
# SUN GEOMETRY + DIFFICULTY
# ============================================================
AZIM_CONVENTIONS = ("north_cw", "image_cw", "unknown")


def make_sun(elev=None, incidence=None, azim=None, conv=None, source=None):
    """Normalised sun record. Elevation wins over incidence (elevation = 90 - incidence)."""
    def _f(v):
        try: return None if v is None or v == "" else float(v)
        except (TypeError, ValueError): return None
    elev, incidence, azim = _f(elev), _f(incidence), _f(azim)
    if elev is None and incidence is not None:
        elev = 90.0 - incidence
    if azim is not None:
        azim = azim % 360.0
        conv = conv if conv in AZIM_CONVENTIONS else "unknown"
    else:
        conv = None
    if elev is None and azim is None:
        return None
    return {"elev_deg": elev, "azim_deg": azim, "azim_convention": conv, "source": source}


def read_sun_json(folder):
    """data/raw/<kind>/<pair>/sun.json — values copied from the product's web page.

    {"incidence_deg": 81.6, "azimuth_deg": 123.4, "azimuth_convention": "north_cw"}
    ("elevation_deg" may be given instead of "incidence_deg")."""
    p = os.path.join(folder, "sun.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    return make_sun(elev=d.get("elevation_deg"), incidence=d.get("incidence_deg"),
                    azim=d.get("azimuth_deg"), conv=d.get("azimuth_convention"), source="sun.json")


def read_label_sun(xml_path):
    """Sun elevation/azimuth from an ISRO PDS4 label (OHRC, TMC-2).
    The label's azimuth is taken as clockwise from north."""
    root = ET.parse(xml_path).getroot()
    def _v(tag):
        el = root.find(f".//isda:{tag}", NS)
        return el.text.strip() if el is not None and el.text else None
    return make_sun(elev=_v("sun_elevation"), azim=_v("sun_azimuth"), conv="north_cw", source="label")


def apply_sun_override(base, override):
    """Values typed on the website replace the file/label values field by field."""
    if not override:
        return base
    merged = dict(base or {"elev_deg": None, "azim_deg": None, "azim_convention": None, "source": None})
    changed = False
    for k in ("elev_deg", "azim_deg", "azim_convention"):
        if override.get(k) not in (None, ""):
            merged[k] = override[k]
            changed = True
    if not changed:
        return base
    return make_sun(elev=merged["elev_deg"], azim=merged["azim_deg"],
                    conv=merged["azim_convention"], source="website")


def assess_difficulty(ohrc_sun, ref_sun, scale_ratio, tex_ohrc, tex_ref, ref_name="reference"):
    """Heuristic pair difficulty from lighting, scale and texture. Report only."""
    score, factors = 0, []
    def add(points, text):
        nonlocal score
        score += points
        factors.append(f"{text} (+{points})" if points else text)

    eo = (ohrc_sun or {}).get("elev_deg")
    er = (ref_sun or {}).get("elev_deg")
    elev_gap = None
    if eo is not None and er is not None:
        elev_gap = abs(eo - er)
        if elev_gap >= 15: add(3, f"sun elevation gap {elev_gap:.1f}°: very different shadow lengths")
        elif elev_gap >= 7: add(1, f"sun elevation gap {elev_gap:.1f}°")
        else: add(0, f"sun elevation gap {elev_gap:.1f}°: similar")
        low = min(eo, er)
        if low < 15: add(1, f"low sun ({low:.1f}°): long shadows")
    else:
        missing = [n for n, v in (("OHRC", eo), (ref_name, er)) if v is None]
        add(0, f"sun elevation unknown for {', '.join(missing)}: not scored")

    ao, ar = (ohrc_sun or {}).get("azim_deg"), (ref_sun or {}).get("azim_deg")
    co, cr = (ohrc_sun or {}).get("azim_convention"), (ref_sun or {}).get("azim_convention")
    azim_gap = None
    if ao is not None and ar is not None and co == cr == "north_cw":
        azim_gap = abs((ao - ar + 180.0) % 360.0 - 180.0)
        if azim_gap >= 30: add(2, f"sun azimuth gap {azim_gap:.1f}°: shadows point different ways")
        elif azim_gap >= 10: add(1, f"sun azimuth gap {azim_gap:.1f}°")
        else: add(0, f"sun azimuth gap {azim_gap:.1f}°: shadows point the same way")
    else:
        add(0, "sun azimuth not compared (missing, or conventions differ / unknown)")

    if scale_ratio >= 25: add(2, f"scale ratio {scale_ratio:.1f}×")
    elif scale_ratio >= 10: add(1, f"scale ratio {scale_ratio:.1f}×")

    tex = min(tex_ohrc, tex_ref)
    if tex < 30: add(2, f"very low texture ({tex:.0f})")
    elif tex < 80: add(1, f"low texture ({tex:.0f})")

    level = "easy" if score <= 1 else "medium" if score <= 3 else "hard"
    return {"level": level, "score": score, "factors": factors,
            "elev_gap_deg": elev_gap, "azim_gap_deg": azim_gap,
            "texture_ohrc": round(float(tex_ohrc), 1), "texture_ref": round(float(tex_ref), 1),
            "scale_ratio": round(float(scale_ratio), 3)}


# ============================================================
# GATED ECC  (notebook cell 15)
# ============================================================
@dataclass
class FineFrame:
    """What ECC and its scoring need to read tiles at any level."""
    ohrc_crop: object
    ref_crop: object
    ohrc_gsd: float
    ref_gsd: float
    fine_gsd: float
    coarse_gsd: float
    ohrc_fine_shape: tuple
    level_view: Callable


def _ecc_refine(fr, H_in, gsd, model_name, iters=200, eps=1e-6):
    o_g = grad_mag_u8(to_uint8(fr.level_view(fr.ohrc_crop, fr.ohrc_gsd, gsd)))
    n_g = grad_mag_u8(to_uint8(fr.level_view(fr.ref_crop, fr.ref_gsd, gsd)))
    W = H_at_level(H_in, fr.fine_gsd, gsd).astype(np.float32)
    motion = cv2.MOTION_HOMOGRAPHY if model_name == 'homography' else cv2.MOTION_AFFINE
    warp = W.astype(np.float32) if motion == cv2.MOTION_HOMOGRAPHY else W[:2].astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    # warpMatrix maps TEMPLATE -> INPUT: OHRC -> reference, our convention; no inversion.
    cc, warp = cv2.findTransformECC(o_g, n_g, warp, motion, crit, None, 5)
    W_ref = warp if motion == cv2.MOTION_HOMOGRAPHY else to_h(warp)
    return H_at_level(W_ref, gsd, fr.fine_gsd), float(cc)


def fine_alignment_score(fr, H, n_tiles=6, size=640):
    """Mean ECC correlation over FINE-level tiles, reference rendered into each OHRC tile."""
    rows = np.linspace(0, max(1, fr.ohrc_fine_shape[0] - size), max(2, n_tiles)).astype(int)
    cols = np.linspace(0, max(1, fr.ohrc_fine_shape[1] - size), 2).astype(int)
    vals = []
    for r0 in rows:
        for c0 in cols:
            o_tile, o_org = level_tile(fr.ohrc_crop, fr.ohrc_gsd, fr.fine_gsd, int(r0), int(c0), size, size)
            if o_tile is None: continue
            ox, oy = o_org; oh, ow = o_tile.shape[:2]
            corners = np.array([[ox, oy], [ox + ow, oy], [ox, oy + oh], [ox + ow, oy + oh]])
            proj = apply_h(H, corners)
            n_r0 = int(np.floor(proj[:, 1].min())) - 8
            n_c0 = int(np.floor(proj[:, 0].min())) - 8
            n_h = int(np.ceil(proj[:, 1].max() - proj[:, 1].min())) + 16
            n_w = int(np.ceil(proj[:, 0].max() - proj[:, 0].min())) + 16
            if n_h < 32 or n_w < 32: continue
            n_tile, n_org = level_tile(fr.ref_crop, fr.ref_gsd, fr.fine_gsd,
                                       max(0, n_r0), max(0, n_c0), n_h, n_w)
            if n_tile is None: continue
            nx, ny = n_org
            H_tile = mat_translate(-nx, -ny) @ H @ mat_translate(ox, oy)
            n_in_o = cv2.warpPerspective(to_uint8(n_tile), H_tile.astype(np.float32), (ow, oh),
                                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
            m = (n_in_o > 0).astype(np.uint8) * 255
            if m.mean() / 255.0 < 0.5: continue
            try:
                vals.append(float(cv2.computeECC(grad_mag_u8(to_uint8(o_tile)), grad_mag_u8(n_in_o), m)))
            except cv2.error:
                continue
            if len(vals) >= n_tiles: break
        if len(vals) >= n_tiles: break
    return float(np.mean(vals)) if vals else None


def legacy_ecc(fr, H_fine, model_name):
    """The old behaviour: one ECC pass at the coarse level, kept whenever it is finite."""
    try:
        H_try, cc = _ecc_refine(fr, H_fine, fr.coarse_gsd, model_name)
    except cv2.error:
        return H_fine, False, {"mode": "legacy", "reason": "ECC did not converge"}
    if np.all(np.isfinite(H_try)):
        return H_try, True, {"mode": "legacy", "cc": cc}
    return H_fine, False, {"mode": "legacy", "reason": "ECC returned a non-finite warp"}


def gated_ecc(fr, H_fine, model_name, px_budget=6_000_000):
    """ECC at a ladder of levels (coarse -> fine), each candidate kept only if it raises the
    fine-level alignment score. A refinement coarser than the current solution cannot make it
    worse any more; worst case is the tiled fit unchanged."""
    info = {"mode": "gated", "tried": []}
    score_before = fine_alignment_score(fr, H_fine)
    info["score_before"] = score_before
    if score_before is None:
        info["reason"] = "no fine tiles to score on; ECC skipped"
        return H_fine, False, info

    def _px_at(shape, native, g):
        return (shape[0] * native / g) * (shape[1] * native / g)
    ladder, g = [fr.coarse_gsd], fr.coarse_gsd / 2.0
    while g > fr.fine_gsd * 0.99 and max(_px_at(fr.ohrc_crop.shape, fr.ohrc_gsd, g),
                                         _px_at(fr.ref_crop.shape, fr.ref_gsd, g)) <= px_budget:
        ladder.append(g); g /= 2.0

    best_H, best_score, best_g = H_fine, score_before, None
    for g in ladder:
        try:
            H_try, _cc = _ecc_refine(fr, H_fine, g, model_name)
        except cv2.error:
            info["tried"].append({"gsd_m": round(g, 3), "score": None, "kept": False, "note": "did not converge"})
            continue
        if not np.all(np.isfinite(H_try)):
            continue
        s = fine_alignment_score(fr, H_try)
        kept = s is not None and s > best_score
        info["tried"].append({"gsd_m": round(g, 3), "score": s, "kept": kept})
        if kept:
            best_H, best_score, best_g = H_try, s, g
    info["score_after"] = best_score
    info["kept_gsd_m"] = None if best_g is None else round(best_g, 3)
    return best_H, best_g is not None, info


# ============================================================
# EVALUATION
# ============================================================
TRUST_MIN_INLIERS = 15
TRUST_MIN_COVERAGE = 0.05
TRUST_MAX_HELDOUT_PX = 5.0
TRUST_MIN_NMI_RATIO = 1.5


def spatial_heldout(src, dst, frame_shape, thresh, n_bands=10, n_folds=5):
    """Held-out RMSE with the test points grouped by position along the strip.

    Points are cut into n_bands bands along the strip's long axis; band k goes to fold
    k % n_folds, so every fold tests interleaved stretches of ground rather than random
    neighbours of training points. The refits skip the 'points bunched along a line' check,
    which is meant for final models and rejects every subset of a thin strip.
    Returns (pooled RMSE or None, folds used)."""
    src = np.asarray(src, np.float64); dst = np.asarray(dst, np.float64)
    if len(src) < 12:
        return None, 0
    axis = 1 if frame_shape[0] >= frame_shape[1] else 0      # dst is (x, y)
    coord = dst[:, axis]
    edges = np.quantile(coord, np.linspace(0, 1, n_bands + 1))
    band = np.clip(np.searchsorted(edges, coord, side="right") - 1, 0, n_bands - 1)
    fold = band % n_folds
    errs, used = [], 0
    for k in range(n_folds):
        te = fold == k; tr = ~te
        if tr.sum() < 8 or te.sum() < 3: continue
        H, _, _ = robust_fit(src[tr], dst[tr], frame_shape, thresh=thresh, min_minor_axis_px=0.0)
        if H is None: continue
        errs.append(np.linalg.norm(apply_h(H, src[te]) - dst[te], axis=1))
        used += 1
    if not errs:
        return None, 0
    e = np.concatenate(errs)
    return float(np.sqrt(np.mean(e ** 2))), used


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


def pick_best(attempts):
    """Trusted first, then lowest held-out RMSE, then most inliers."""
    def key(a):
        ev = a["eval"]
        h = ev["rmse_px_heldout_spatial"]
        return (not ev["trusted"], h if h is not None else float("inf"), -a["fit"]["n_inliers"])
    return min(range(len(attempts)), key=lambda i: key(attempts[i]))


# ============================================================
# TILE RESCUES  (only called for tiles whose normal match failed)
# ============================================================
def tile_consensus(ka, kb, sc, px):
    """Same per-tile gate as the engine: points must agree on one shift within px."""
    if len(ka) < 2:
        return None
    d = kb - ka; med = np.median(d, axis=0); dev = np.linalg.norm(d - med, axis=1)
    if float(np.median(dev)) > px:
        return None
    agree = dev <= px
    if int(agree.sum()) < 2:
        return None
    return ka[agree], kb[agree], sc[agree]


def suppress_shadows(img_u8, valid, rel=0.45, max_frac=0.45, sigma=12.0):
    """Replace cast shadows with a smooth fill from the surrounding lit ground.

    Shadow edges move with the sun, so they are the gradients most likely to be matched to the
    wrong place. Normalised convolution fills each shadow from nearby lit pixels, which removes
    the edge without inventing structure."""
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
    ys, xs = np.where(mask)
    if len(ys) == 0: return 0
    side = min(ys.max() - ys.min(), xs.max() - xs.min())
    return 3 if side // 3 >= 64 else 2 if side // 2 >= 64 else 0


def _parabolic_peak(res, loc):
    """Sub-pixel offset of a correlation peak from a parabola through its neighbours.
    (Phase correlation on windowed gradient patches was measured to return corrections of
    the wrong sign on smooth, downsampled tiles; the parabola was exact in the same test.)"""
    x, y = loc; h, w = res.shape
    def _p(m1, c, p1):
        den = m1 - 2.0 * c + p1
        return 0.0 if abs(den) < 1e-12 else float(np.clip(0.5 * (m1 - p1) / den, -0.5, 0.5))
    sx = _p(res[y, x - 1], res[y, x], res[y, x + 1]) if 0 < x < w - 1 else 0.0
    sy = _p(res[y - 1, x], res[y, x], res[y + 1, x]) if 0 < y < h - 1 else 0.0
    return sx, sy


def _zint(x):
    f = cv2.GaussianBlur(np.asarray(x, np.float32), (0, 0), 1.0)
    return (f - f.mean()) / (f.std() + 1e-9)


def dense_match(warped_u8, ref_u8, valid_mask, feature="grad", invert=False, subdiv=3, min_margin=2.0):
    """The engine's dense tile matcher (subdiv x subdiv ZNCC patches, phase-randomised null),
    with a choice of feature and a parabolic sub-pixel step.

    feature="grad"      gradient magnitude, as the engine uses;
    feature="intensity" brightness. Gradient magnitude cannot tell a lit slope from a shadowed
                        one; brightness can, and with invert=True the OHRC template is negated so
                        it matches ground whose lit and shadowed faces swapped."""
    h, w = ref_u8.shape[:2]
    ys, xs = np.where(valid_mask)
    if len(ys) < 400: return _EMPTY
    Y0, Y1, X0, X1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    feat = _zgrad if feature == "grad" else _zint
    b_z = feat(ref_u8)
    a_z = (-1.0 if invert else 1.0) * feat(warped_u8)
    nulls = [feat(_phase_randomize(ref_u8, seed=q)) for q in range(2)]
    ka, kb, sc = [], [], []
    for iy in range(subdiv):
        for ix in range(subdiv):
            ph = (Y1 - Y0) // subdiv; pw = (X1 - X0) // subdiv
            if ph < 64 or pw < 64: continue
            ty = Y0 + iy * ph; tx = X0 + ix * pw
            th = int(ph * 0.85); tw_ = int(pw * 0.85)
            ty += (ph - th) // 2; tx += (pw - tw_) // 2
            if valid_mask[ty:ty + th, tx:tx + tw_].mean() < 0.9: continue
            tpl = np.ascontiguousarray(a_z[ty:ty + th, tx:tx + tw_])
            if tpl.shape[0] > b_z.shape[0] or tpl.shape[1] > b_z.shape[1]: continue
            try:
                res = cv2.matchTemplate(b_z, tpl, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(res)
                null = max(cv2.minMaxLoc(cv2.matchTemplate(n, tpl, cv2.TM_CCOEFF_NORMED))[1] for n in nulls)
            except cv2.error: continue
            if mx / max(null, 1e-6) < min_margin: continue
            sdx, sdy = _parabolic_peak(res, loc)
            dx, dy = loc[0] - tx + sdx, loc[1] - ty + sdy
            cx, cy = tx + tw_ / 2.0, ty + th / 2.0
            ka.append([cx, cy]); kb.append([cx + dx, cy + dy])
            sc.append(float(min(1.0, mx)))
    if not ka: return _EMPTY
    return (np.asarray(ka, np.float32), np.asarray(kb, np.float32), np.asarray(sc, np.float32))


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
    out = tile_consensus(*dense_match(a_s, b_s, m_s, subdiv=sub), consensus_px / s)
    if out is None:
        return None
    ka, kb, sc = out
    return (ka + 0.5) * s - 0.5, (kb + 0.5) * s - 0.5, sc


def rescue_tile(o_pc, n_pc, vmask, opts, consensus_px):
    """Try the enabled rescues in order; first one that passes the consensus gate wins.
    Returns ((ka, kb, sc), method) or (None, None)."""
    a, b = o_pc, n_pc
    tag = ""
    if opts.get("shadow_mask"):
        a, b = suppress_shadows(o_pc, vmask), suppress_shadows(n_pc, vmask)
        out = tile_consensus(*dense_match(a, b, vmask), consensus_px)
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
            out = tile_consensus(*dense_match(a, b, vmask, feature="intensity", invert=inv), consensus_px)
            if out is not None:
                return out, ("intensity inverted" if inv else "intensity") + tag
    return None, None


# ============================================================
# CRATER-ANCHORED MATCHES
# ============================================================
def detect_craters(img_u8, mask, min_r=3.0, max_r=30.0, max_n=600):
    """Crater candidates as (x, y, r) from LoG blobs on smoothed gradient magnitude.

    A crater's rim and walls form a ring of strong gradient whatever the sun angle, so a
    blob detector on gradient magnitude finds its centre from either image."""
    from skimage.feature import blob_log
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
        if x - m < 0 or y - m < 0 or x + m >= w or y + m >= h: continue
        if not mask[y - m:y + m, x - m:x + m].all(): continue
        keep.append((xs[i], ys[i], rr, float(g[y - m:y + m, x - m:x + m].std())))
    if not keep:
        return np.zeros((0, 3), np.float32)
    keep.sort(key=lambda t: -t[3])
    return np.asarray([k[:3] for k in keep[:max_n]], np.float32)


def crater_matches(ohrc_c, ref_c, H_coarse, min_ncc=0.3, min_margin=2.0):
    """Crater-anchored correspondences at the coarse level.

    The OHRC is warped into the reference frame with the coarse prior; craters are detected in
    both, paired by position and size, and each pair is refined by ZNCC on gradient magnitude
    with a phase-correlation sub-pixel step. Returns (pts_ohrc_coarse, pts_ref_coarse, scores,
    stats), all in coarse pixels."""
    stats = {"craters_ohrc": 0, "craters_ref": 0, "paired": 0, "matched": 0}
    empty = (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32), np.zeros(0, np.float32), stats)
    h, w = ref_c.shape[:2]
    Hc = np.asarray(H_coarse, np.float64)
    warped = cv2.warpPerspective(ohrc_c, Hc, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    valid = cv2.warpPerspective(np.full(ohrc_c.shape, 255, np.uint8), Hc, (w, h),
                                flags=cv2.INTER_NEAREST, borderValue=0)
    valid = cv2.erode(valid, np.ones((9, 9), np.uint8)) > 0
    if valid.sum() < 2000:
        return empty

    cw = detect_craters(warped, valid)
    cr = detect_craters(ref_c, valid)
    stats["craters_ohrc"], stats["craters_ref"] = int(len(cw)), int(len(cr))
    if len(cw) < 3 or len(cr) < 3:
        return empty

    from scipy.spatial import cKDTree
    tree = cKDTree(cr[:, :2])
    best_for_ref = {}
    for i, (x, y, r) in enumerate(cw):
        d, j = tree.query([x, y], k=min(5, len(cr)), distance_upper_bound=max(12.0, 2.5 * r))
        for dj, jj in zip(np.atleast_1d(d), np.atleast_1d(j)):
            if not np.isfinite(dj): break
            if 0.67 <= cr[jj, 2] / r <= 1.5:
                if jj not in best_for_ref or dj < best_for_ref[jj][1]:
                    best_for_ref[jj] = (i, dj)
                break
    pairs = [(i, j) for j, (i, _d) in best_for_ref.items()]
    stats["paired"] = len(pairs)
    if len(pairs) < 3:
        return empty

    a_z, b_z = _zgrad(warped), _zgrad(ref_c)
    b_null = _zgrad(_phase_randomize(ref_c, seed=0))
    pw, pr, sc = [], [], []
    for i, j in pairs:
        x, y, r = cw[i]; xr, yr = cr[j, 0], cr[j, 1]
        s = int(max(12, 2 * r))
        x0, y0 = int(round(x)) - s, int(round(y)) - s
        if x0 < 0 or y0 < 0 or x0 + 2 * s > w or y0 + 2 * s > h: continue
        if not valid[y0:y0 + 2 * s, x0:x0 + 2 * s].all(): continue
        tpl = np.ascontiguousarray(a_z[y0:y0 + 2 * s, x0:x0 + 2 * s])
        m = s // 2 + 6
        sx0, sy0 = int(round(xr)) - s - m, int(round(yr)) - s - m
        sx1, sy1 = sx0 + 2 * s + 2 * m, sy0 + 2 * s + 2 * m
        if sx0 < 0 or sy0 < 0 or sx1 > w or sy1 > h: continue
        try:
            res = cv2.matchTemplate(b_z[sy0:sy1, sx0:sx1], tpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            null = cv2.minMaxLoc(cv2.matchTemplate(b_null[sy0:sy1, sx0:sx1], tpl, cv2.TM_CCOEFF_NORMED))[1]
        except cv2.error:
            continue
        if mx < min_ncc or mx / max(null, 1e-6) < min_margin: continue
        mx0, my0 = sx0 + loc[0], sy0 + loc[1]
        sdx, sdy = _parabolic_peak(res, loc)
        pw.append([x0 + s, y0 + s]); pr.append([mx0 + s + sdx, my0 + s + sdy]); sc.append(float(mx))
    if len(pw) < 3:
        return empty
    pw, pr, sc = np.asarray(pw, np.float64), np.asarray(pr, np.float64), np.asarray(sc, np.float32)

    # Keep pairs that agree on the local shift (the prior is already close, so a genuine
    # correction is smooth; a wrong pairing sticks out).
    d = pr - pw; med = np.median(d, axis=0); dev = np.linalg.norm(d - med, axis=1)
    tol = max(3.0, 3.0 * 1.4826 * float(np.median(dev)))
    keep = dev <= tol
    if keep.sum() < 3:
        return empty
    pw, pr, sc = pw[keep], pr[keep], sc[keep]
    stats["matched"] = int(len(pw))
    pts_ohrc = apply_h(np.linalg.inv(Hc), pw)
    return pts_ohrc.astype(np.float32), pr.astype(np.float32), sc, stats
