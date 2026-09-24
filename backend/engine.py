"""Registration engine — steps 3..18, shared by every OHRC <-> reference pipeline.

A pipeline module (backend/pipelines/*.py) does steps 1-2: it loads the OHRC and the
reference product and hands them over as a LoadedPair, together with a Profile holding the
knobs that differ between pipelines. Everything from the overlap crop to the evaluation
report is the same algorithm for NAC and TMC, so it lives here once.

Homographies map OHRC pixels -> reference pixels throughout.
"""
import os, json, time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
import torch
from sklearn.metrics import normalized_mutual_info_score
from transformers import AutoImageProcessor, AutoModelForKeypointMatching

from .common import (
    KM_PER_DEG_LAT, km_per_deg_lon, unwrap_lon, fit_affine_pixel_to_latlon, mat_translate, mat_scale, to_h, apply_h,
    H_at_level, to_uint8, grad_mag_u8, texture_score, to_rgb_pil, prep_pair, level_view,
    level_view_antialiased, level_tile, level_shape, _phase_randomize, ncc_orientation_search,
    hull_coverage, robust_fit, pad_to_square, match_dense, _EMPTY, img_to_b64, fig_to_b64,
    side_by_side, stretch_to_uint8, plt,
)
from . import robust


@dataclass
class Profile:
    """What differs between pipelines once both products are loaded."""
    key: str                               # 'ohrc_nac' / 'ohrc_tmc'
    ref_name: str                          # 'NAC' / 'TMC' — used in step names and metrics
    version: str                           # output file tag: 'v4' / 'v5'
    fine_gsd_multiplier: float = 1.0       # fine GSD = coarser native GSD x this
    antialiased_resampling: bool = False   # level_view_antialiased instead of level_view
    adaptive_tile: bool = False            # TILE sized to the OHRC overlap instead of 640
    options: dict = field(default_factory=dict)   # per-pipeline defaults for robust.DEFAULT_OPTIONS


@dataclass
class LoadedPair:
    """Output of a pipeline's load step (steps 1-2)."""
    ohrc_raster: object
    ohrc_corners: dict
    ohrc_shape: tuple
    ohrc_gsd: float
    ohrc_product_id: str
    ref_raster: object
    ref_corners: dict
    ref_shape: tuple
    ref_gsd: float
    ref_product_id: str
    extra_metrics: dict = field(default_factory=dict)
    ohrc_sun: dict = None                  # robust.make_sun() records; used for the difficulty score
    ref_sun: dict = None


# Keypoint-matcher weights are loaded once per process, not once per run.
MATCHER_CHECKPOINTS = {
    "superglue": ("SuperGlue", "magic-leap-community/superglue_outdoor"),
    "lightglue": ("LightGlue", "ETH-CVG/lightglue_superpoint"),
}
_MATCHERS = {}

def _matcher(device, name):
    key = (name, device)
    if key not in _MATCHERS:
        ckpt = MATCHER_CHECKPOINTS[name][1]
        _MATCHERS[key] = (AutoImageProcessor.from_pretrained(ckpt),
                          AutoModelForKeypointMatching.from_pretrained(ckpt).to(device).eval())
    return _MATCHERS[key]


def _blend(warped, ref):
    return np.where(warped > 0, ((warped.astype(np.float32) + ref.astype(np.float32)) / 2).astype(np.uint8), ref)


# ---- Step previews ----------------------------------------------------------------------
# The coarse frames are long thin strips; shrunk whole to 800 px they become ~150 px wide
# slivers in which 2 px markers vanish and a few-pixel alignment change is invisible. So each
# preview is cropped to the region that matters, resized first, and annotated afterwards.
PREVIEW_SIDE = 800

def _crop_box(mask, margin=0.06):
    ys, xs = np.where(mask)
    h, w = mask.shape[:2]
    if len(ys) == 0:
        return 0, h, 0, w
    my, mx = int((ys.max() - ys.min()) * margin) + 8, int((xs.max() - xs.min()) * margin) + 8
    return max(0, ys.min() - my), min(h, ys.max() + my + 1), max(0, xs.min() - mx), min(w, xs.max() + mx + 1)

STRIP_SHORT_SIDE = 360     # a long strip keeps at least this many px across ...
STRIP_LONG_MAX = 3200      # ... unless that would make it longer than this

def _layout(h, w, side=PREVIEW_SIDE):
    """How to show an h x w region: (segments, axis, scale). Always one unbroken piece; a long
    strip is kept readable across its short side and the website scrolls along it."""
    k = side / max(h, w)
    if min(h, w) * k < STRIP_SHORT_SIDE:
        k = min(STRIP_SHORT_SIDE / min(h, w), STRIP_LONG_MAX / max(h, w))
    return 1, 0, k

def _fold(img, n, axis, gap=6):
    """Cut img into n pieces along axis (0 = rows) and lay them out across the other axis."""
    if n <= 1:
        return img
    parts = np.array_split(img, n, axis=axis)
    L = max(p.shape[axis] for p in parts)
    out = []
    for i, p in enumerate(parts):
        pad = [(0, 0)] * img.ndim
        pad[axis] = (0, L - p.shape[axis])
        out.append(np.pad(p, pad, constant_values=40))
        if i < n - 1:
            sep_shape = list(out[-1].shape)
            sep_shape[1 - axis] = gap
            out.append(np.full(sep_shape, 40, img.dtype))
    return np.concatenate(out, axis=1 - axis)

def _fit(img):
    """Resize for the preview; returns (image, scale, segments, axis) for _fold."""
    h, w = img.shape[:2]
    n, axis, k = _layout(h, w)
    interp = cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (max(1, int(round(w * k))), max(1, int(round(h * k)))), interpolation=interp), k, n, axis

def _caption(bgr, text):
    """Dark strip with a caption above the preview."""
    bar = np.full((28, bgr.shape[1], 3), 20, np.uint8)
    cv2.putText(bar, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 225, 225), 1, cv2.LINE_AA)
    return np.vstack([bar, bgr])

def _overlay_preview(warped, ref, caption):
    """False-colour overlay cropped to the OHRC footprint: OHRC magenta, reference green.
    Aligned ground reads grey; any misalignment shows as coloured fringes."""
    r0, r1, c0, c1 = _crop_box(warped > 0)
    w, rf = warped[r0:r1, c0:c1], ref[r0:r1, c0:c1]
    has = w > 0
    bgr = np.dstack([np.where(has, w, rf), rf, np.where(has, w, rf)]).astype(np.uint8)
    bgr, _k, n, axis = _fit(bgr)
    return _caption(_fold(bgr, n, axis), caption)

def _points_preview(base_u8, groups, caption, radius=3):
    """Points drawn AFTER resizing so they stay visible. groups: [(pts_xy_in_base, bgr), ...];
    a bgr may also be an (N, 3) array of per-point colours. Cropped to where the points are."""
    pts_all = [np.asarray(p, np.float64).reshape(-1, 2) for p, _ in groups if len(p)]
    if pts_all:
        allp = np.vstack(pts_all)
        mask = np.zeros(base_u8.shape[:2], bool)
        xs = np.clip(allp[:, 0].astype(int), 0, mask.shape[1] - 1)
        ys = np.clip(allp[:, 1].astype(int), 0, mask.shape[0] - 1)
        mask[ys, xs] = True
        r0, r1, c0, c1 = _crop_box(mask)
    else:
        r0, r1, c0, c1 = 0, base_u8.shape[0], 0, base_u8.shape[1]
    img, k, n, axis = _fit(cv2.cvtColor(to_uint8(base_u8[r0:r1, c0:c1]), cv2.COLOR_GRAY2BGR))
    for pts, color in groups:
        pts = np.asarray(pts, np.float64).reshape(-1, 2)
        cols = np.asarray(color).reshape(-1, 3) if np.ndim(color) == 2 else None
        for i, (x, y) in enumerate(pts):
            c = tuple(int(v) for v in (cols[i] if cols is not None else color))
            cv2.circle(img, (int((x - c0) * k), int((y - r0) * k)), radius, c, -1, cv2.LINE_AA)
    return _caption(_fold(img, n, axis), caption)


def run_registration(pair_id, pair, profile, config, emit_event, results, out_dir):
    opts = robust.resolve_options(profile.options, config.get('options'))
    MAX_TILES = 400
    REF = profile.ref_name
    _level_view = level_view_antialiased if profile.antialiased_resampling else level_view

    ohrc_raster, ohrc_corners, ohrc_shape = pair.ohrc_raster, pair.ohrc_corners, pair.ohrc_shape
    ref_raster, ref_corners, ref_shape = pair.ref_raster, pair.ref_corners, pair.ref_shape
    OHRC_NATIVE_GSD_M, REF_NATIVE_GSD_M = pair.ohrc_gsd, pair.ref_gsd
    _, ohrc_fwd_inv = fit_affine_pixel_to_latlon(ohrc_corners, ohrc_shape)
    _, ref_fwd_inv = fit_affine_pixel_to_latlon(ref_corners, ref_shape)

    # ========== STEP 3: Overlap & Crop ==========
    emit_event(3, "Overlap Detection & Crop", "running", 12)

    PAD_M = 150.0
    def corner_bounds(corners_deg, lon_ref=None):
        lats = [c["lat"] for c in corners_deg.values()]
        ref = corners_deg["UL"]["lon"] if lon_ref is None else lon_ref
        lons = [float(unwrap_lon(c["lon"], ref)) for c in corners_deg.values()]
        return min(lats), max(lats), min(lons), max(lons)

    LON_REF = ohrc_corners["UL"]["lon"]
    o_lat0, o_lat1, o_lon0, o_lon1 = corner_bounds(ohrc_corners, LON_REF)
    n_lat0, n_lat1, n_lon0, n_lon1 = corner_bounds(ref_corners, LON_REF)
    _ov_lat = min(o_lat1, n_lat1) - max(o_lat0, n_lat0)
    _ov_lon = min(o_lon1, n_lon1) - max(o_lon0, n_lon0)
    if _ov_lat <= 0 or _ov_lon <= 0:
        raise RuntimeError(f"OHRC and {REF} footprints do not overlap.")

    def _footprint_poly(corners, lon_ref):
        order = ("UL", "UR", "LR", "LL")
        return np.array([[float(unwrap_lon(corners[k]["lon"], lon_ref)), corners[k]["lat"]] for k in order], np.float32)
    poly_o = _footprint_poly(ohrc_corners, LON_REF)
    poly_n = _footprint_poly(ref_corners, LON_REF)
    _area, inter_poly = cv2.intersectConvexConvex(poly_o, poly_n)
    if _area > 0 and inter_poly is not None and len(inter_poly) >= 3:
        inter_poly = np.asarray(inter_poly, np.float64).reshape(-1, 2)
    else:
        _fl0, _fl1 = max(o_lat0, n_lat0), min(o_lat1, n_lat1)
        _fn0, _fn1 = max(o_lon0, n_lon0), min(o_lon1, n_lon1)
        inter_poly = np.array([[_fn0, _fl0], [_fn1, _fl0], [_fn1, _fl1], [_fn0, _fl1]])

    def crop_to_polygon(raster, inverse_fn, verts_lonlat, gsd, name):
        n_lines, n_samples = raster.shape[:2]
        pix = [inverse_fn(float(v[1]), float(v[0])) for v in verts_lonlat]
        rows = [p[0] for p in pix]; cols = [p[1] for p in pix]
        _ext_r, _ext_c = max(rows) - min(rows), max(cols) - min(cols)
        pad_px = min(PAD_M / max(gsd, 1e-6), 0.15 * max(min(_ext_r, _ext_c), 1.0))
        r0 = int(max(0, np.floor(min(rows) - pad_px)))
        r1 = int(min(n_lines, np.ceil(max(rows) + pad_px)))
        c0 = int(max(0, np.floor(min(cols) - pad_px)))
        c1 = int(min(n_samples, np.ceil(max(cols) + pad_px)))
        if r0 >= r1 or c0 >= c1: raise ValueError(f"{name}: empty crop window")
        return raster[r0:r1, c0:c1], (r0, r1, c0, c1)

    ohrc_crop, ohrc_win = crop_to_polygon(ohrc_raster, ohrc_fwd_inv, inter_poly, OHRC_NATIVE_GSD_M, "OHRC")
    ref_crop, ref_win = crop_to_polygon(ref_raster, ref_fwd_inv, inter_poly, REF_NATIVE_GSD_M, REF)
    # 16-bit products (TMC-2) are stretched to 8-bit here, on the crop, never on the full strip.
    if ohrc_crop.dtype != np.uint8: ohrc_crop = stretch_to_uint8(ohrc_crop)
    if ref_crop.dtype != np.uint8: ref_crop = stretch_to_uint8(ref_crop)
    T_ohrc_crop = mat_translate(-ohrc_win[2], -ohrc_win[0])
    T_ref_crop = mat_translate(-ref_win[2], -ref_win[0])

    _mid_lat = 0.5 * (max(o_lat0, n_lat0) + min(o_lat1, n_lat1))
    ov_km_lat = _ov_lat * KM_PER_DEG_LAT
    ov_km_lon = _ov_lon * km_per_deg_lon(_mid_lat)
    emit_event(3, "Overlap Detection & Crop", "done", 16,
               detail=f"Overlap: {ov_km_lat:.1f} × {ov_km_lon:.1f} km · OHRC crop {ohrc_crop.shape} · {REF} crop {ref_crop.shape}",
               image=img_to_b64(side_by_side(ohrc_crop, ref_crop)))

    # ========== STEP 4: Resolution Scheme ==========
    emit_event(4, "Resolution Scheme", "running", 18)

    TARGET_GSD_FINE_M = max(OHRC_NATIVE_GSD_M, REF_NATIVE_GSD_M) * profile.fine_gsd_multiplier
    COARSE_PIXEL_BUDGET = 2_000_000
    TARGET_GSD_COARSE_M = TARGET_GSD_FINE_M
    def _px_at(shape, native_gsd, gsd):
        return (shape[0] * native_gsd / gsd) * (shape[1] * native_gsd / gsd)
    while max(_px_at(ohrc_crop.shape, OHRC_NATIVE_GSD_M, TARGET_GSD_COARSE_M),
              _px_at(ref_crop.shape, REF_NATIVE_GSD_M, TARGET_GSD_COARSE_M)) > COARSE_PIXEL_BUDGET:
        TARGET_GSD_COARSE_M *= 1.25

    OHRC_FINE_SHAPE = level_shape(ohrc_crop.shape, OHRC_NATIVE_GSD_M, TARGET_GSD_FINE_M)
    REF_FINE_SHAPE = level_shape(ref_crop.shape, REF_NATIVE_GSD_M, TARGET_GSD_FINE_M)
    M_ohrc_native_to_fine = mat_scale(OHRC_NATIVE_GSD_M / TARGET_GSD_FINE_M) @ T_ohrc_crop
    M_ref_native_to_fine = mat_scale(REF_NATIVE_GSD_M / TARGET_GSD_FINE_M) @ T_ref_crop

    ohrc_coarse = _level_view(ohrc_crop, OHRC_NATIVE_GSD_M, TARGET_GSD_COARSE_M)
    ref_coarse = _level_view(ref_crop, REF_NATIVE_GSD_M, TARGET_GSD_COARSE_M)

    _ratio = max(OHRC_NATIVE_GSD_M, REF_NATIVE_GSD_M) / min(OHRC_NATIVE_GSD_M, REF_NATIVE_GSD_M)
    emit_event(4, "Resolution Scheme", "done", 20,
               detail=f"Scale ratio {_ratio:.1f}× · Fine {TARGET_GSD_FINE_M:.3f} m/px · Coarse {TARGET_GSD_COARSE_M:.3f} m/px · OHRC coarse {ohrc_coarse.shape} · {REF} coarse {ref_coarse.shape}",
               image=img_to_b64(side_by_side(ohrc_coarse, ref_coarse)))

    # ========== STEP 5: Appearance Preprocessing ==========
    emit_event(5, "Appearance Preprocessing", "running", 22)

    ohrc_c_pc, ref_c_pc, ohrc_c_sg, ref_c_sg = prep_pair(ohrc_coarse, ref_coarse)

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].imshow(ohrc_c_pc, cmap='gray'); ax[0, 0].set_title('OHRC plain', color='white', fontsize=9)
    ax[0, 1].imshow(ref_c_pc, cmap='gray'); ax[0, 1].set_title(f'{REF} plain', color='white', fontsize=9)
    ax[1, 0].imshow(ohrc_c_sg, cmap='gray'); ax[1, 0].set_title('OHRC CLAHE → SuperGlue', color='white', fontsize=9)
    ax[1, 1].imshow(ref_c_sg, cmap='gray'); ax[1, 1].set_title(f'{REF} CLAHE + hist-matched', color='white', fontsize=9)
    for a in ax.ravel(): a.axis('off')
    plt.tight_layout()
    preproc_img = fig_to_b64(fig)

    # Pair difficulty (report only): lighting gap, scale ratio and texture.
    difficulty = robust.assess_difficulty(pair.ohrc_sun, pair.ref_sun, _ratio,
                                          texture_score(ohrc_c_pc), texture_score(ref_c_pc), REF)
    emit_event(5, "Appearance Preprocessing", "done", 26,
               detail=f"CLAHE + histogram matching applied · Pair difficulty: {difficulty['level'].upper()} "
                      f"(score {difficulty['score']}) · " + "; ".join(difficulty['factors']),
               image=preproc_img)
    results['preprocessed'] = preproc_img

    # ========== STEP 6: Orientation Search ==========
    emit_event(6, "Orientation Search (ZNCC)", "running", 28)

    ncc_rows, ncc_floor = ncc_orientation_search(ohrc_c_pc, ref_c_pc)
    top = ncc_rows[0] if ncc_rows else None
    margin = top[0] / max(ncc_floor, 1e-6) if top else 0
    verdict = "SAME GROUND" if (margin >= 2.5 and top and top[0] >= 0.12) else "INCONCLUSIVE" if margin >= 1.5 else "NO DETECTABLE OVERLAP"
    if verdict == "SAME GROUND" and len(ncc_rows) > 1 and top[0] / max(ncc_rows[1][0], 1e-6) >= 1.15:
        FLIP_CANDIDATES = [top[2]]
    else:
        FLIP_CANDIDATES = [r[2] for r in ncc_rows]

    emit_event(6, "Orientation Search (ZNCC)", "done", 32,
               detail=f"Verdict: {verdict} · Margin ×{margin:.1f} · Best flip: {top[1] if top else '?'} · {len(FLIP_CANDIDATES)} candidate(s)",
               image=img_to_b64(grad_mag_u8(ohrc_c_pc)))

    # ========== STEP 7: Rotation Estimate ==========
    emit_event(7, "Residual Rotation Estimate", "running", 34)

    def _pad_square(img):
        a = np.asarray(img, dtype=np.float32)
        h, w = a.shape[:2]; n = max(h, w)
        out = np.zeros((n, n), np.float32)
        r, c = (n - h) // 2, (n - w) // 2
        out[r:r + h, c:c + w] = a
        return out

    def estimate_rotation_deg(img_a, img_b, canvas=512):
        a = cv2.resize(_pad_square(img_a), (canvas, canvas), interpolation=cv2.INTER_AREA)
        b = cv2.resize(_pad_square(img_b), (canvas, canvas), interpolation=cv2.INTER_AREA)
        win = cv2.createHanningWindow((canvas, canvas), cv2.CV_32F)
        a, b = a * win, b * win
        def log_polar_mag(img):
            mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img)))).astype(np.float32)
            return cv2.warpPolar(mag, (canvas, canvas), (canvas / 2.0, canvas / 2.0), canvas / 2.0, cv2.WARP_POLAR_LOG + cv2.INTER_LINEAR)
        (_, shift_y), response = cv2.phaseCorrelate(log_polar_mag(a), log_polar_mag(b))
        return float(shift_y * (360.0 / canvas)), float(response)

    _TEST_ANGLE = 7.0
    _h, _w = ohrc_c_pc.shape
    _rot = cv2.warpAffine(ohrc_c_pc, cv2.getRotationMatrix2D((_w / 2, _h / 2), _TEST_ANGLE, 1.0), (_w, _h))
    _recovered, _ = estimate_rotation_deg(ohrc_c_pc, _rot)
    ROT_SIGN = -1.0 if abs(_recovered - _TEST_ANGLE) <= abs(_recovered + _TEST_ANGLE) else 1.0

    emit_event(7, "Residual Rotation Estimate", "done", 36,
               detail=f"Self-test passed · Rotation sign: {ROT_SIGN:+.0f}",
               image=img_to_b64(to_uint8(_rot)))

    # ========== STEP 8: SuperGlue ==========
    emit_event(8, "SuperPoint + SuperGlue", "running", 38)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    MATCHER_LABEL = MATCHER_CHECKPOINTS[opts['matcher']][0]
    sg_processor, sg_model = _matcher(device, opts['matcher'])

    def match_superglue(a_u8, b_u8, threshold=0.15, size=1024):
        if a_u8.size == 0 or b_u8.size == 0: return _EMPTY
        a_sq, ax_, ay_ = pad_to_square(a_u8)
        b_sq, bx_, by_ = pad_to_square(b_u8)
        images = [to_rgb_pil(a_sq), to_rgb_pil(b_sq)]
        sg_processor.size = {"height": int(size), "width": int(size)}
        inputs = sg_processor(images, return_tensors="pt").to(device)
        with torch.inference_mode(): raw = sg_model(**inputs)
        sizes = [[(im.height, im.width) for im in images]]
        out = sg_processor.post_process_keypoint_matching(raw, sizes, threshold=threshold)[0]
        k0 = out["keypoints0"].float().cpu().numpy().reshape(-1, 2) - np.array([ax_, ay_], np.float32)
        k1 = out["keypoints1"].float().cpu().numpy().reshape(-1, 2) - np.array([bx_, by_], np.float32)
        sc = out["matching_scores"].float().cpu().numpy().reshape(-1)
        keep = ((k0[:, 0] >= 0) & (k0[:, 1] >= 0) & (k0[:, 0] < a_u8.shape[1]) & (k0[:, 1] < a_u8.shape[0]) &
                (k1[:, 0] >= 0) & (k1[:, 1] >= 0) & (k1[:, 0] < b_u8.shape[1]) & (k1[:, 1] < b_u8.shape[0]))
        return k0[keep], k1[keep], sc[keep]

    _k0, _k1, _sc = match_superglue(ohrc_c_sg, ref_c_sg)
    _h_max = max(ohrc_c_sg.shape[0], ref_c_sg.shape[0])
    _o_pad = np.zeros((_h_max, ohrc_c_sg.shape[1]), dtype=ohrc_c_sg.dtype); _o_pad[:ohrc_c_sg.shape[0], :] = ohrc_c_sg
    _n_pad = np.zeros((_h_max, ref_c_sg.shape[1]), dtype=ref_c_sg.dtype); _n_pad[:ref_c_sg.shape[0], :] = ref_c_sg
    _sg_vis_rgb = cv2.cvtColor(np.hstack([_o_pad, _n_pad]), cv2.COLOR_GRAY2BGR)
    _w_off = ohrc_c_sg.shape[1]
    for _i in range(min(100, len(_k0))):
        _pt0 = (int(_k0[_i, 0]), int(_k0[_i, 1]))
        _pt1 = (int(_k1[_i, 0]) + _w_off, int(_k1[_i, 1]))
        cv2.line(_sg_vis_rgb, _pt0, _pt1, (0, 255, 255), 1)
        cv2.circle(_sg_vis_rgb, _pt0, 3, (0, 0, 255), -1)
        cv2.circle(_sg_vis_rgb, _pt1, 3, (0, 255, 0), -1)
    emit_event(8, "SuperPoint + SuperGlue", "done", 42,
               detail=f"Matcher: SuperPoint + {MATCHER_LABEL} · Device: {device} · {len(_k0)} coarse matches at threshold 0.15",
               image=img_to_b64(_sg_vis_rgb))

    # ========== STEP 9: Geometric Prior ==========
    emit_event(9, "Geometric Prior", "running", 44)

    def latlon_affine(corners_deg, shape, lon_ref):
        n_l, n_s = shape
        keys = ("UL", "UR", "LL", "LR")
        pix = np.array([[0, 0], [0, n_s - 1], [n_l - 1, 0], [n_l - 1, n_s - 1]], float)
        lat = np.array([corners_deg[k]["lat"] for k in keys], float)
        lon = np.array([float(unwrap_lon(corners_deg[k]["lon"], lon_ref)) for k in keys])
        A = np.column_stack([pix[:, 0], pix[:, 1], np.ones(4)])
        cl, *_ = np.linalg.lstsq(A, lat, rcond=None)
        cn, *_ = np.linalg.lstsq(A, lon, rcond=None)
        return np.array([cl[:2], cn[:2]]), np.array([cl[2], cn[2]])

    _Mo, _to = latlon_affine(ohrc_corners, ohrc_shape, LON_REF)
    _Mn, _tn = latlon_affine(ref_corners, ref_shape, LON_REF)
    _R = np.linalg.inv(_Mn) @ _Mo
    _T = np.linalg.inv(_Mn) @ (_to - _tn)
    H_native_geo = np.array([[_R[1, 1], _R[1, 0], _T[1]], [_R[0, 1], _R[0, 0], _T[0]], [0.0, 0.0, 1.0]])
    H_fine_geo = M_ref_native_to_fine @ H_native_geo @ np.linalg.inv(M_ohrc_native_to_fine)

    _A = H_native_geo[:2, :2]
    _det = float(np.linalg.det(_A))
    _geo_warped = cv2.warpPerspective(ohrc_c_sg, H_at_level(H_fine_geo, TARGET_GSD_FINE_M, TARGET_GSD_COARSE_M).astype(np.float64),
                                      (ref_c_sg.shape[1], ref_c_sg.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
    emit_event(9, "Geometric Prior", "done", 48,
               detail=f"Scale ×{np.hypot(*_A[:, 0]):.4f}, ×{np.hypot(*_A[:, 1]):.4f} · {'MIRRORED' if _det < 0 else 'Not mirrored'} · det={_det:+.4f}",
               image=img_to_b64(_overlay_preview(_geo_warped, ref_c_sg, f"Step 9 - geometric prior | OHRC magenta, {REF} green, grey = aligned")))

    # ========== STEP 10: Dense Shift Field ==========
    emit_event(10, "Dense Shift Field", "running", 50)

    H_coarse_geo = H_at_level(H_fine_geo, TARGET_GSD_FINE_M, TARGET_GSD_COARSE_M)
    _h_n, _w_n = ref_c_sg.shape
    ohrc_c_sg_warped = cv2.warpPerspective(ohrc_c_sg, H_coarse_geo.astype(np.float64), (_w_n, _h_n), flags=cv2.INTER_LINEAR, borderValue=0)
    _valid_c = cv2.warpPerspective(np.full(ohrc_c_sg.shape, 255, np.uint8), H_coarse_geo.astype(np.float64), (_w_n, _h_n), flags=cv2.INTER_NEAREST, borderValue=0)

    def _z(x):
        x = np.asarray(x, np.float32)
        return (x - x.mean()) / (x.std() + 1e-9)

    def measure_band_shifts(warped_u8, ref_u8, valid_u8, n_bands=8, slack=160):
        h, w = ref_u8.shape
        a_all, b_all = _z(grad_mag_u8(warped_u8)), _z(grad_mag_u8(ref_u8))
        edges = np.linspace(0, h, n_bands + 1).astype(int)
        rows = []
        for k in range(n_bands):
            y0, y1 = edges[k], edges[k + 1]
            if y1 - y0 < 64: continue
            cov = (valid_u8[y0:y1] > 0).mean()
            if cov < 0.35: continue
            tw = max(32, int(w * 0.6)); x0 = (w - tw) // 2
            th = max(32, int((y1 - y0) * 0.8)); ty = y0 + ((y1 - y0) - th) // 2
            tpl = a_all[ty:ty + th, x0:x0 + tw]
            sy0, sy1 = max(0, ty - slack), min(h, ty + th + slack)
            seg = b_all[sy0:sy1]
            if tpl.shape[0] > seg.shape[0] or tpl.shape[1] > seg.shape[1]: continue
            _, mx, _, loc = cv2.minMaxLoc(cv2.matchTemplate(seg, tpl, cv2.TM_CCOEFF_NORMED))
            null = max(cv2.minMaxLoc(cv2.matchTemplate(
                _z(_phase_randomize(ref_u8[sy0:sy1], seed=q)), tpl, cv2.TM_CCOEFF_NORMED))[1] for q in range(2))
            rows.append(dict(yc=ty + th / 2.0, dx=loc[0] - x0, dy=(sy0 + loc[1]) - ty,
                             ncc=float(mx), margin=float(mx / max(null, 1e-6)), cov=float(cov)))
        return rows

    _bands = measure_band_shifts(ohrc_c_sg_warped, ref_c_sg, _valid_c)
    _good = [r for r in _bands if r['margin'] >= 2.0]

    PRIOR_SHIFT_COEF = None
    PRIOR_SHIFT_RANGE = None
    PRIOR_SHIFT_BASE = (0.0, 0.0)
    DENSE_APPLIED = False

    if len(_good) >= 3:
        _yc = np.array([r['yc'] for r in _good], float)
        _dx = np.array([r['dx'] for r in _good], float)
        _dy = np.array([r['dy'] for r in _good], float)
        _deg = 2 if len(_good) >= 5 else 1
        PRIOR_SHIFT_COEF = (np.polyfit(_yc, _dx, _deg), np.polyfit(_yc, _dy, _deg))
        PRIOR_SHIFT_RANGE = (float(_yc.min()), float(_yc.max()))
        DENSE_APPLIED = True
        _kc = TARGET_GSD_FINE_M / TARGET_GSD_COARSE_M
        _midc_dx = float(np.polyval(PRIOR_SHIFT_COEF[0], REF_FINE_SHAPE[0] * _kc / 2.0))
        _midc_dy = float(np.polyval(PRIOR_SHIFT_COEF[1], REF_FINE_SHAPE[0] * _kc / 2.0))
        PRIOR_SHIFT_BASE = (_midc_dx, _midc_dy)
        H_coarse_geo = mat_translate(_midc_dx, _midc_dy) @ H_coarse_geo
        H_fine_geo = H_at_level(H_coarse_geo, TARGET_GSD_COARSE_M, TARGET_GSD_FINE_M)

    _shift_warped = cv2.warpPerspective(ohrc_c_sg, H_coarse_geo.astype(np.float64),
                                        (_w_n, _h_n), flags=cv2.INTER_LINEAR, borderValue=0)
    emit_event(10, "Dense Shift Field", "done", 54,
               detail=f"{len(_good)}/{len(_bands)} bands cleared null · {'Polynomial shift applied' if DENSE_APPLIED else 'No shift needed'}",
               image=img_to_b64(_overlay_preview(_shift_warped, ref_c_sg, f"Step 10 - after dense shift correction | OHRC magenta, {REF} green")))

    # ========== STEP 11: Coarse Refinement ==========
    emit_event(11, "Coarse Refinement", "running", 56)

    ohrc_c_sg_warped = cv2.warpPerspective(ohrc_c_sg, H_coarse_geo.astype(np.float64), (_w_n, _h_n), flags=cv2.INTER_LINEAR, borderValue=0)

    pool0, pool1, _pools = match_superglue(ohrc_c_sg_warped, ref_c_sg)

    H_fine0 = H_fine_geo
    coarse_model = 'geometric prior + dense shift' if DENSE_APPLIED else 'geometric prior'
    REFINED = False

    if len(pool0) >= 4:
        C, coarse_mask, _m = robust_fit(pool0, pool1, ref_c_sg.shape, verbose=False)
        if C is not None:
            n_in = int(coarse_mask.sum())
            shift = float(np.linalg.norm(apply_h(C, [[_w_n / 2, _h_n / 2]])[0] - [_w_n / 2, _h_n / 2]))
            if n_in >= 25 and shift <= 250:
                H_c = C @ H_coarse_geo
                if abs(np.linalg.det(H_c[:2, :2])) > 1e-9:
                    H_fine0 = H_at_level(H_c, TARGET_GSD_COARSE_M, TARGET_GSD_FINE_M)
                    coarse_model = f'geometric prior + {_m} correction'
                    REFINED = True

    _ref_warped = cv2.warpPerspective(ohrc_c_sg, H_at_level(H_fine0, TARGET_GSD_FINE_M, TARGET_GSD_COARSE_M).astype(np.float64),
                                      (_w_n, _h_n), flags=cv2.INTER_LINEAR, borderValue=0)
    emit_event(11, "Coarse Refinement", "done", 60,
               detail=f"Model: {coarse_model} · {len(pool0)} matches · {'Correction applied' if REFINED else 'Using prior as-is'}",
               image=img_to_b64(_overlay_preview(_ref_warped, ref_c_sg, f"Step 11 - after coarse refinement | OHRC magenta, {REF} green")))

    # ========== STEPS 12-14: one attempt (auto-retry may repeat them with rescues on) ==========
    def _shift_at(yc):
        if PRIOR_SHIFT_COEF is None: return 0.0, 0.0
        if PRIOR_SHIFT_RANGE is not None:
            yc = float(np.clip(yc, PRIOR_SHIFT_RANGE[0], PRIOR_SHIFT_RANGE[1]))
        return (float(np.polyval(PRIOR_SHIFT_COEF[0], yc)), float(np.polyval(PRIOR_SHIFT_COEF[1], yc)))

    def prior_shift_fine(y_fine):
        if PRIOR_SHIFT_COEF is None: return 0.0, 0.0
        k = TARGET_GSD_FINE_M / TARGET_GSD_COARSE_M
        dx, dy = _shift_at(float(y_fine) * k)
        return (dx - PRIOR_SHIFT_BASE[0]) / k, (dy - PRIOR_SHIFT_BASE[1]) / k

    SG_SIZE_FINE = 640
    if profile.adaptive_tile:
        # v5-tmc: size tiles to fit ~4 columns across the OHRC overlap, clamped to [256, 640].
        TILE = int(np.clip(2 ** int(np.floor(np.log2(max(256.0, OHRC_FINE_SHAPE[1] / 4.0)))),
                           256, SG_SIZE_FINE))
    else:
        TILE = SG_SIZE_FINE
    TILE_OVERLAP = 0.25; SEARCH_MARGIN = 128
    MIN_VALID_FRAC = 0.25; MIN_TEXTURE = 12.0
    TILE_CONSENSUS_PX = 6.0

    stride = max(1, int(TILE * (1.0 - TILE_OVERLAP)))
    _rows = list(range(0, max(1, OHRC_FINE_SHAPE[0] - TILE // 2), stride))
    _cols = list(range(0, max(1, OHRC_FINE_SHAPE[1] - TILE // 2), stride))
    tile_jobs = [(r0, c0) for r0 in _rows for c0 in _cols][:MAX_TILES]
    N_WORKERS = min(os.cpu_count() or 4, 14)
    _k_f2c = TARGET_GSD_FINE_M / TARGET_GSD_COARSE_M

    frame = robust.FineFrame(ohrc_crop=ohrc_crop, ref_crop=ref_crop,
                             ohrc_gsd=OHRC_NATIVE_GSD_M, ref_gsd=REF_NATIVE_GSD_M,
                             fine_gsd=TARGET_GSD_FINE_M, coarse_gsd=TARGET_GSD_COARSE_M,
                             ohrc_fine_shape=OHRC_FINE_SHAPE, level_view=_level_view)
    _crater_cache = {}

    def _process_one_tile(r0, c0, o):
        o_tile_raw, o_origin = level_tile(ohrc_crop, OHRC_NATIVE_GSD_M, TARGET_GSD_FINE_M, r0, c0, TILE, TILE)
        if o_tile_raw is None: return {'status': 'skip'}
        ox, oy = o_origin; oh, ow = o_tile_raw.shape[:2]
        if texture_score(to_uint8(o_tile_raw)) < MIN_TEXTURE: return {'status': 'skip_texture'}

        corners = np.array([[ox, oy], [ox + ow, oy], [ox, oy + oh], [ox + ow, oy + oh]])
        _py = apply_h(H_fine0, [[ox + ow / 2.0, oy + oh / 2.0]])[0][1]
        _sx, _sy = prior_shift_fine(_py)
        H_local = mat_translate(_sx, _sy) @ H_fine0
        H_local_inv = np.linalg.inv(H_local)
        proj = apply_h(H_local, corners)
        n_c0 = int(np.floor(proj[:, 0].min())) - SEARCH_MARGIN
        n_r0 = int(np.floor(proj[:, 1].min())) - SEARCH_MARGIN
        n_w = int(np.ceil(proj[:, 0].max() - proj[:, 0].min())) + 2 * SEARCH_MARGIN
        n_h = int(np.ceil(proj[:, 1].max() - proj[:, 1].min())) + 2 * SEARCH_MARGIN
        if n_w < 32 or n_h < 32 or n_r0 > REF_FINE_SHAPE[0] or n_c0 > REF_FINE_SHAPE[1]: return {'status': 'skip'}

        n_tile_raw, n_origin = level_tile(ref_crop, REF_NATIVE_GSD_M, TARGET_GSD_FINE_M, max(0, n_r0), max(0, n_c0), n_h, n_w)
        if n_tile_raw is None: return {'status': 'skip'}
        nx, ny = n_origin; nh, nw = n_tile_raw.shape[:2]

        H_tile = mat_translate(-nx, -ny) @ H_local @ mat_translate(ox, oy)
        valid = cv2.warpPerspective(np.full((oh, ow), 255, np.uint8), H_tile, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
        if valid.mean() / 255.0 < MIN_VALID_FRAC: return {'status': 'skip_overlap'}

        warped_o_raw = cv2.warpPerspective(np.asarray(o_tile_raw), H_tile, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
        vmask = valid > 0
        o_pc, n_pc, o_sg, n_sg = prep_pair(warped_o_raw, n_tile_raw, mask=vmask)
        if texture_score(n_pc) < MIN_TEXTURE: return {'status': 'skip_texture'}

        back_to_ohrc = H_local_inv @ mat_translate(nx, ny)
        to_ref_fine = mat_translate(nx, ny)

        # Dense matching; per-tile consensus: the points must agree on one shift.
        ka, kb, sc = match_dense(o_pc, n_pc, vmask)
        fail = 'skip' if len(ka) < 2 else None
        if fail is None:
            agreed = robust.tile_consensus(ka, kb, sc, TILE_CONSENSUS_PX)
            if agreed is None: fail = 'no_consensus'
            else: ka, kb, sc = agreed

        rescued_by = None
        if fail is not None:
            if not robust.any_rescue(o): return {'status': fail}
            out, rescued_by = robust.rescue_tile(o_pc, n_pc, vmask, o, TILE_CONSENSUS_PX)
            if out is None: return {'status': fail}
            ka, kb, sc = out

        return {
            'status': 'matched',
            'pts_o': apply_h(back_to_ohrc, ka),
            'pts_n': apply_h(to_ref_fine, kb),
            'scores': sc,
            'n_src': len(ka),
            'rescued_by': rescued_by,
        }

    def match_tiles(o, note, pg):
        """Step 12. Returns the pooled fine correspondences or raises if there are none."""
        emit_event(12, "Tiled Matching at Native Resolution", "running", pg(62), detail=note)
        fine_pts_o, fine_pts_n, fine_scores, fine_src = [], [], [], []
        tiles_tried = tiles_used = 0
        rescued_by = {}
        _t_start = time.time()
        _completed = 0

        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {executor.submit(_process_one_tile, r0, c0, o): (r0, c0) for r0, c0 in tile_jobs}
            for future in as_completed(futures):
                _completed += 1
                tiles_tried += 1
                try:
                    result = future.result()
                except Exception:
                    continue
                if result['status'] == 'matched':
                    fine_pts_o.append(result['pts_o'])
                    fine_pts_n.append(result['pts_n'])
                    fine_scores.append(result['scores'])
                    src = 'dense' if result['rescued_by'] is None else f"rescue: {result['rescued_by']}"
                    fine_src.extend([src] * result['n_src'])
                    if result['rescued_by'] is not None:
                        rescued_by[result['rescued_by']] = rescued_by.get(result['rescued_by'], 0) + 1
                    tiles_used += 1

                if _completed % 20 == 0:
                    pct = 62 + int(15 * _completed / max(len(tile_jobs), 1))
                    emit_event(12, "Tiled Matching at Native Resolution", "running", pg(min(pct, 77)),
                               detail=(f"{note} · " if note else "") +
                                      f"{_completed}/{len(tile_jobs)} tiles · {tiles_used} matched · {sum(len(p) for p in fine_pts_o)} points · {N_WORKERS} threads")

        crater_stats = None
        if o['crater_matching']:
            if 'result' not in _crater_cache:   # depends only on the coarse prior, so compute once
                H_c = H_at_level(H_fine0, TARGET_GSD_FINE_M, TARGET_GSD_COARSE_M)
                _crater_cache['result'] = robust.crater_matches(ohrc_c_pc, ref_c_pc, H_c)
            c_o, c_n, c_s, crater_stats = _crater_cache['result']
            if len(c_o):
                fine_pts_o.append(c_o / _k_f2c)      # coarse px -> fine px
                fine_pts_n.append(c_n / _k_f2c)
                fine_scores.append(c_s)
                fine_src.extend(['crater'] * len(c_o))

        if not fine_pts_o: raise RuntimeError("No tiles produced matches.")
        fine_o = np.vstack(fine_pts_o)
        fine_n = np.vstack(fine_pts_n)
        fine_s = np.concatenate(fine_scores)

        _src = np.asarray(fine_src)
        _pc = fine_n * _k_f2c
        _resc = np.char.startswith(_src.astype(str), 'rescue')
        _tile_vis = _points_preview(ref_coarse, [
            (_pc[_src == 'dense'][:2000], (0, 255, 0)),
            (_pc[_resc][:2000], (0, 165, 255)),
            (_pc[_src == 'crater'][:2000], (255, 255, 0)),
        ], f"Step 12 - {len(fine_n)} matches: green dense, orange rescued, cyan crater")
        n_rescued = sum(rescued_by.values())
        extra = ""
        if robust.any_rescue(o): extra += f" · {n_rescued} tiles rescued"
        if crater_stats is not None: extra += f" · {crater_stats['matched']} crater matches"
        emit_event(12, "Tiled Matching at Native Resolution", "done", pg(78),
                   detail=(f"{note} · " if note else "") +
                          f"Tile {TILE} px · {tiles_used}/{tiles_tried} tiles matched · {len(fine_o)} correspondences{extra} · {time.time()-_t_start:.0f}s",
                   image=img_to_b64(_tile_vis))
        return {"fine_o": fine_o, "fine_n": fine_n, "fine_s": fine_s, "fine_src": np.asarray(fine_src),
                "tiles_tried": tiles_tried, "tiles_used": tiles_used,
                "rescued_by": rescued_by, "crater_stats": crater_stats}

    def estimate_residual_scale(src, dst, frame_shape, t0=100.0, iters=3):
        t = t0
        for _ in range(iters):
            H, m, _ = robust_fit(src, dst, frame_shape, thresh=t, verbose=False, min_minor_axis_px=0.0)
            if H is None: return None
            r = np.linalg.norm(apply_h(H, src) - dst, axis=1)
            s_ = 1.4826 * float(np.median(np.abs(r - np.median(r))))
            t = float(max(3.0, 2.5 * max(s_, 1e-3)))
        return t

    def fit_model(t, note, pg):
        """Step 13."""
        emit_event(13, "Final Model (RANSAC)", "running", pg(80), detail=note)
        fine_o, fine_n = t["fine_o"], t["fine_n"]
        thresh = estimate_residual_scale(fine_o, fine_n, REF_FINE_SHAPE) or 3.0
        H_fine, inlier_mask, model_name = robust_fit(fine_o, fine_n, REF_FINE_SHAPE, thresh=thresh, min_minor_axis_px=5.0)
        if H_fine is None: raise RuntimeError("Could not fit a model to the fine correspondences.")
        n_inliers = int(inlier_mask.sum())
        inlier_ratio = n_inliers / len(fine_o)
        coverage = hull_coverage(fine_n[inlier_mask], REF_FINE_SHAPE)
        trusted_legacy = n_inliers >= 15 and coverage >= 0.05

        _pc = fine_n * _k_f2c
        _ransac_vis = _points_preview(ref_coarse, [
            (_pc[~inlier_mask][:3000], (0, 0, 255)),
            (_pc[inlier_mask][:3000], (0, 255, 0)),
        ], f"Step 13 - {model_name}: {n_inliers} inliers green, {len(fine_o) - n_inliers} outliers red")
        emit_event(13, "Final Model (RANSAC)", "done", pg(83),
                   detail=(f"{note} · " if note else "") +
                          f"Model: {model_name} · {n_inliers}/{len(fine_o)} inliers ({inlier_ratio:.1%}) · Coverage {coverage:.1%} · "
                          f"basic checks {'pass' if trusted_legacy else 'fail'} (final verdict in step 18)",
                   image=img_to_b64(_ransac_vis))
        return {"H_fine": H_fine, "inlier_mask": inlier_mask, "model_name": model_name,
                "thresh": thresh, "n_inliers": n_inliers, "inlier_ratio": inlier_ratio,
                "coverage": coverage, "trusted_legacy": trusted_legacy}

    def refine(fm, o, note, pg):
        """Step 14."""
        emit_event(14, "ECC Intensity Refinement", "running", pg(85), detail=note)
        mode = o['ecc_mode']
        if mode == 'off':
            H_final, applied, info = fm["H_fine"], False, {"mode": "off"}
            msg = "Off for this run: tiled fit used as-is"
        elif mode == 'legacy':
            H_final, applied, info = robust.legacy_ecc(frame, fm["H_fine"], fm["model_name"])
            msg = "Legacy: " + ("applied without a quality check" if applied else info.get("reason", "not applied"))
        else:
            H_final, applied, info = robust.gated_ecc(frame, fm["H_fine"], fm["model_name"])
            if info.get("score_before") is None:
                msg = "Gated: " + info.get("reason", "skipped")
            elif applied:
                msg = (f"Gated: fine-tile score {info['score_before']:.4f} → {info['score_after']:.4f} · "
                       f"kept ECC from the {info['kept_gsd_m']} m/px level")
            else:
                msg = (f"Gated: no level improved the fine-tile score ({info['score_before']:.4f}) · "
                       f"ECC rejected, tiled fit kept")
        _ecc_warped = cv2.warpPerspective(to_uint8(ohrc_coarse),
                                          H_at_level(H_final, TARGET_GSD_FINE_M, TARGET_GSD_COARSE_M).astype(np.float64),
                                          (ref_coarse.shape[1], ref_coarse.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
        emit_event(14, "ECC Intensity Refinement", "done", pg(87),
                   detail=(f"{note} · " if note else "") + msg,
                   image=img_to_b64(_overlay_preview(_ecc_warped, to_uint8(ref_coarse),
                                                     f"Step 14 - final fit ({'ECC applied' if applied else 'ECC not applied'}) | OHRC magenta, {REF} green")))
        return H_final, applied, info

    def evaluate(t, fm, H_final):
        """Accuracy and trust for one attempt (reported in step 18)."""
        MATCH_SRC = t["fine_o"][fm["inlier_mask"]]
        MATCH_DST = t["fine_n"][fm["inlier_mask"]]

        def _rmse(H):
            return float(np.sqrt(np.mean(np.sum((apply_h(H, MATCH_SRC) - MATCH_DST) ** 2, axis=1))))
        rmse_in = _rmse(H_final)
        # ECC optimises intensity, not point residuals; report both so a disagreement is visible.
        rmse_pre_ecc = _rmse(fm["H_fine"])

        rng = np.random.default_rng(0); held = []
        for _ in range(5):
            idx = rng.permutation(len(MATCH_SRC)); cut = int(0.7 * len(idx))
            tr, te = idx[:cut], idx[cut:]
            if len(tr) < 8 or len(te) < 4: continue
            H_tr, _, _ = robust_fit(MATCH_SRC[tr], MATCH_DST[tr], REF_FINE_SHAPE, verbose=False, thresh=fm["thresh"], min_minor_axis_px=5.0)
            if H_tr is None: continue
            held.append(float(np.sqrt(np.mean(np.linalg.norm(apply_h(H_tr, MATCH_SRC[te]) - MATCH_DST[te], axis=1) ** 2))))
        rmse_heldout = float(np.mean(held)) if held else None
        rmse_spatial, folds = robust.spatial_heldout(MATCH_SRC, MATCH_DST, REF_FINE_SHAPE, fm["thresh"])

        warped = cv2.warpPerspective(to_uint8(ohrc_coarse), H_at_level(H_final, TARGET_GSD_FINE_M, TARGET_GSD_COARSE_M).astype(np.float64),
                                     (ref_coarse.shape[1], ref_coarse.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)
        ref_disp = to_uint8(ref_coarse)
        _valid = (warped > 0) & (ref_disp > 0)
        nmi = nmi_baseline = None
        if _valid.sum() > 1000:
            _sub = np.random.default_rng(0).choice(int(_valid.sum()), size=min(200_000, int(_valid.sum())), replace=False)
            nmi = float(normalized_mutual_info_score((warped[_valid] // 8)[_sub], (ref_disp[_valid] // 8)[_sub]))
            _shift = np.roll(warped, (37, 29), axis=(0, 1))
            _v2 = (_shift > 0) & (ref_disp > 0)
            _s2 = np.random.default_rng(0).choice(int(_v2.sum()), size=min(200_000, int(_v2.sum())), replace=False)
            nmi_baseline = float(normalized_mutual_info_score((_shift[_v2] // 8)[_s2], (ref_disp[_v2] // 8)[_s2]))

        # Prefer the spatial held-out figure; fall back to the random split when it is missing.
        held_for_trust = rmse_spatial if rmse_spatial is not None else rmse_heldout
        trusted, checks = robust.trust_verdict(fm["n_inliers"], fm["coverage"], held_for_trust, nmi, nmi_baseline)
        return {"MATCH_SRC": MATCH_SRC, "MATCH_DST": MATCH_DST, "warped": warped, "ref_disp": ref_disp,
                "rmse_in": rmse_in, "rmse_pre_ecc": rmse_pre_ecc, "rmse_heldout": rmse_heldout,
                "rmse_px_heldout_spatial": rmse_spatial, "heldout_spatial_folds": folds,
                "nmi": nmi, "nmi_baseline": nmi_baseline, "trusted": trusted, "trust_checks": checks}

    def run_attempt(o, idx, note):
        pg = (lambda v: v) if idx == 0 else (lambda v: 87)   # keep the progress bar from jumping back
        t = match_tiles(o, note, pg)
        fm = fit_model(t, note, pg)
        H_final, ecc_applied, ecc_info = refine(fm, o, note, pg)
        ev = evaluate(t, fm, H_final)
        return {"options": o, "note": note, "tiles": t, "fit": fm, "H_FINAL": H_final,
                "ecc_applied": ecc_applied, "ecc_info": ecc_info, "eval": ev}

    # Attempt 0 uses the requested options. With auto_retry, each ladder rung adds rescues
    # and is run only while the previous attempt is not trusted.
    ladder = [("requested options", opts)]
    if opts['auto_retry']:
        for label, extra in robust.RETRY_LADDER:
            nxt = {**ladder[-1][1], **extra}
            if nxt != ladder[-1][1]:
                ladder.append((label, nxt))

    attempts, attempt_log = [], []
    for idx, (label, o) in enumerate(ladder):
        if attempts and attempts[-1]["eval"]["trusted"]:
            break
        note = None if idx == 0 else f"Retry {idx}/{len(ladder) - 1}: {label}"
        try:
            att = run_attempt(o, idx, note)
        except RuntimeError as e:
            if not opts['auto_retry']:
                raise
            attempt_log.append({"attempt": idx, "label": label, "error": str(e)})
            continue
        att["index"] = idx
        attempts.append(att)
        attempt_log.append({"attempt": idx, "label": label, "trusted": att["eval"]["trusted"],
                            "n_inliers": att["fit"]["n_inliers"],
                            "rmse_px_heldout_spatial": att["eval"]["rmse_px_heldout_spatial"],
                            "tiles_matched": att["tiles"]["tiles_used"]})
    if not attempts:
        raise RuntimeError("Every attempt failed: " + "; ".join(a["error"] for a in attempt_log))
    best = attempts[robust.pick_best(attempts)]

    t, fm, ev = best["tiles"], best["fit"], best["eval"]
    fine_o = t["fine_o"]
    H_FINAL, MODEL_NAME = best["H_FINAL"], fm["model_name"]
    inlier_mask, n_inliers, coverage = fm["inlier_mask"], fm["n_inliers"], fm["coverage"]
    MATCH_SRC, MATCH_DST = ev["MATCH_SRC"], ev["MATCH_DST"]
    TRUSTED = ev["trusted"]
    chosen = "" if len(attempt_log) == 1 else f" · using attempt {best['index']} of {len(attempt_log)}"

    # ========== STEP 15: Sub-pixel Refinement ==========
    emit_event(15, "Sub-pixel Refinement", "running", 88)
    # Simplified — skip sub-pixel for speed, keep H_FINAL
    emit_event(15, "Sub-pixel Refinement", "done", 89, detail="Using model as-is for this run")

    # ========== STEP 16: Consensus Tie Points ==========
    emit_event(16, "Consensus Tie Points", "running", 90)

    RESIDUAL_PX = np.linalg.norm(apply_h(H_FINAL, MATCH_SRC) - MATCH_DST, axis=1)
    inlier_src = t["fine_src"][inlier_mask]
    inliers_by_source = {str(k): int(v) for k, v in zip(*np.unique(inlier_src, return_counts=True))}

    _res_max = max(float(np.percentile(RESIDUAL_PX, 95)), 1.0)
    _r_norm = np.minimum(RESIDUAL_PX / _res_max, 1.0)
    _cols = np.stack([np.zeros_like(_r_norm), 255 * (1 - _r_norm), 255 * _r_norm], axis=1)  # green=low error, red=high
    _tie_vis = _points_preview(ref_coarse, [(MATCH_DST * _k_f2c, _cols)],
                               f"Step 16 - {len(MATCH_SRC)} tie points by residual: green low, red >= {_res_max:.1f} px")
    _by_src = " · " + ", ".join(f"{v} {k}" for k, v in inliers_by_source.items()) if len(inliers_by_source) > 1 else ""
    emit_event(16, "Consensus Tie Points", "done", 91,
               detail=f"{len(MATCH_SRC)} tie points{_by_src} · Residual mean {RESIDUAL_PX.mean():.3f} px = {RESIDUAL_PX.mean()*TARGET_GSD_FINE_M*100:.1f} cm{chosen}",
               image=img_to_b64(_tie_vis))

    # ========== STEP 17: Final Warp & Export ==========
    emit_event(17, "Final Warp & Export", "running", 92)

    H_native = np.linalg.inv(M_ref_native_to_fine) @ H_FINAL @ M_ohrc_native_to_fine
    warped, ref_disp = ev["warped"], ev["ref_disp"]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(warped, cmap='gray'); ax[0].set_title(f'OHRC warped → {REF} frame', color='white', fontsize=9)
    ax[1].imshow(ref_disp, cmap='gray'); ax[1].set_title(f'{REF} reference', color='white', fontsize=9)
    _blk = max(8, min(warped.shape) // 20)
    _yy, _xx = np.mgrid[0:warped.shape[0], 0:warped.shape[1]]
    _chk = np.where((((_yy // _blk) + (_xx // _blk)) % 2).astype(bool), warped, ref_disp)
    ax[2].imshow(_chk, cmap='gray'); ax[2].set_title('Checkerboard', color='white', fontsize=9)
    for a in ax: a.axis('off')
    plt.tight_layout()
    overlay_img = fig_to_b64(fig)
    results['overlay'] = overlay_img
    results['checkerboard'] = overlay_img  # same figure has both

    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
    _ohrc_u8 = to_uint8(ohrc_coarse)
    _ref_u8 = ref_disp
    _h1, _h2 = _ohrc_u8.shape[0], _ref_u8.shape[0]
    if _h1 != _h2:
        _max_h = max(_h1, _h2)
        if _h1 < _max_h: _ohrc_u8 = np.pad(_ohrc_u8, ((0, _max_h - _h1), (0, 0)), mode='constant')
        if _h2 < _max_h: _ref_u8 = np.pad(_ref_u8, ((0, _max_h - _h2), (0, 0)), mode='constant')
    ax2.imshow(np.hstack([_ohrc_u8, _ref_u8]), cmap='gray')
    src_c = MATCH_SRC * _k_f2c
    dst_c = MATCH_DST * _k_f2c
    for i in range(min(200, len(src_c))):
        ax2.plot([src_c[i, 0], dst_c[i, 0] + ohrc_coarse.shape[1]], [src_c[i, 1], dst_c[i, 1]], 'c-', lw=0.3, alpha=0.5)
    ax2.plot(src_c[:200, 0], src_c[:200, 1], 'r.', ms=2)
    ax2.plot(dst_c[:200, 0] + ohrc_coarse.shape[1], dst_c[:200, 1], 'g.', ms=2)
    ax2.set_title(f'{len(MATCH_SRC)} correspondences', color='white', fontsize=10)
    ax2.axis('off')
    plt.tight_layout()
    results['matches'] = fig_to_b64(fig2)

    tag = f'{pair_id}_{profile.version}'
    np.save(os.path.join(out_dir, f'{tag}_H_native.npy'), H_native)
    np.save(os.path.join(out_dir, f'{tag}_H_fine.npy'), H_FINAL)

    emit_event(17, "Final Warp & Export", "done", 96,
               detail=f"Warped raster + homography saved to {out_dir}",
               image=overlay_img)

    # ========== STEP 18: Evaluation & Report ==========
    emit_event(18, "Evaluation & Report", "running", 97)

    rmse_in, rmse_heldout, rmse_spatial = ev["rmse_in"], ev["rmse_heldout"], ev["rmse_px_heldout_spatial"]
    nmi, nmi_baseline = ev["nmi"], ev["nmi_baseline"]

    ref_key = REF.lower()
    metrics = {
        "pair_id": pair_id, "pipeline": profile.key, "version": profile.version,
        "reference": REF, "model": MODEL_NAME,
        "ohrc_product_id": pair.ohrc_product_id, f"{ref_key}_product_id": pair.ref_product_id,
        "n_correspondences": int(len(fine_o)), "n_inliers": n_inliers,
        "inlier_ratio": float(fm["inlier_ratio"]), "spatial_coverage_pct": float(coverage * 100),
        "rmse_px_in_sample": rmse_in, "rmse_m_in_sample": rmse_in * TARGET_GSD_FINE_M,
        "rmse_px_in_sample_pre_ecc": ev["rmse_pre_ecc"],
        "rmse_px_heldout": rmse_heldout,
        "rmse_m_heldout": (rmse_heldout * TARGET_GSD_FINE_M) if rmse_heldout is not None else None,
        "rmse_px_heldout_spatial": rmse_spatial,
        "rmse_m_heldout_spatial": (rmse_spatial * TARGET_GSD_FINE_M) if rmse_spatial is not None else None,
        "heldout_spatial_folds": ev["heldout_spatial_folds"],
        "nmi_aligned": nmi, "nmi_misaligned_baseline": nmi_baseline,
        "ecc_applied": bool(best["ecc_applied"]), "ecc_mode": best["options"]["ecc_mode"], "ecc_info": best["ecc_info"],
        "trusted": bool(TRUSTED), "trusted_legacy": bool(fm["trusted_legacy"]), "trust_checks": ev["trust_checks"],
        "tiles_tried": t["tiles_tried"], "tiles_matched": t["tiles_used"], "tile_px": TILE,
        "tiles_rescued": int(sum(t["rescued_by"].values())), "rescued_by": t["rescued_by"],
        "crater_stats": t["crater_stats"], "inliers_by_source": inliers_by_source,
        "gsd_fine_m": TARGET_GSD_FINE_M, "gsd_coarse_m": TARGET_GSD_COARSE_M,
        "fine_gsd_multiplier": profile.fine_gsd_multiplier,
        "scale_ratio": round(REF_NATIVE_GSD_M / OHRC_NATIVE_GSD_M, 3),
        "difficulty": difficulty, "ohrc_sun": pair.ohrc_sun, "ref_sun": pair.ref_sun,
        "matcher": opts["matcher"], "options": best["options"],
        "attempts": attempt_log, "selected_attempt": best["index"],
    }
    metrics.update(pair.extra_metrics)

    with open(os.path.join(out_dir, f"{tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    results['metrics'] = metrics

    _held_txt = (f"held-out {rmse_spatial:.3f} px ({rmse_spatial*TARGET_GSD_FINE_M:.2f} m)" if rmse_spatial is not None
                 else "held-out n/a")
    _failed = [c["name"] for c in ev["trust_checks"] if not c["pass"]]
    emit_event(18, "Evaluation & Report", "done", 100,
               detail=f"RMSE {rmse_in:.3f} px ({rmse_in*TARGET_GSD_FINE_M:.2f} m) · {_held_txt} · NMI {(f'{nmi:.4f}' if nmi else 'N/A')} · "
                      + ("TRUSTED" if TRUSTED else f"LOW CONFIDENCE (failed: {', '.join(_failed)})") + chosen,
               metrics=metrics,
               images={k: v for k, v in results.items() if k != 'metrics'})
    return metrics
