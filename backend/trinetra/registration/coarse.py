"""Steps 5-11: global alignment at the coarse level.

The model starts from the two corner footprints (step 9), is corrected for along-strip drift
(step 10) and refined with keypoint matches (step 11). Steps 6-8 are diagnostics shown on the
website; their results do not feed the model.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .difficulty import assess_difficulty
from .steps import Step
from ..geometry.geodesy import corner_affine
from ..geometry.transforms import H_at_level, apply_h, mat_translate, warp
from ..imaging.radiometry import grad_mag_u8, phase_randomize, prep_pair, texture_score, to_uint8, zscore, zscore_grad
from ..matching.fitting import robust_fit
from ..matching.keypoints import CHECKPOINTS, KeypointMatcher, default_device
from ..matching.orientation import ncc_orientation_search, rotation_sign
from ..reporting.encoding import img_to_b64
from ..reporting.figures import preprocessing_figure
from ..reporting.previews import match_lines, overlay_preview


@dataclass
class ShiftField:
    """Along-strip drift of the geometric prior (step 10): polynomials in coarse reference
    rows. The shift at mid-strip (base) is already folded into the coarse model."""
    coef: tuple = None
    y_range: tuple = None
    base: tuple = (0.0, 0.0)
    fine_to_coarse: float = 1.0

    def _at_coarse_row(self, yc):
        yc = float(np.clip(yc, self.y_range[0], self.y_range[1]))
        return float(np.polyval(self.coef[0], yc)), float(np.polyval(self.coef[1], yc))

    def at_fine_row(self, y_fine):
        """Residual drift (dx, dy) in fine pixels at a fine reference row."""
        if self.coef is None:
            return 0.0, 0.0
        k = self.fine_to_coarse
        dx, dy = self._at_coarse_row(float(y_fine) * k)
        return (dx - self.base[0]) / k, (dy - self.base[1]) / k


@dataclass
class CoarseResult:
    H_fine0: np.ndarray        # OHRC -> reference model at the fine level
    shift: ShiftField
    ohrc_pc: np.ndarray        # plain 8-bit coarse images (crater matching, difficulty)
    ref_pc: np.ndarray
    difficulty: dict
    preprocessed_image: str


def align_coarse(pair, levels, ref_name, opts, emit):
    ohrc_pc, ref_pc, ohrc_sg, ref_sg, difficulty, preprocessed = _preprocess(pair, levels, ref_name, emit)
    _orientation_check(ohrc_pc, ref_pc, emit)
    _rotation_check(ohrc_pc, emit)
    matcher = _keypoint_preview(ohrc_sg, ref_sg, opts["matcher"], emit)
    H_fine_geo, geo_warped = _geometric_prior(pair, levels, ohrc_sg, ref_sg, ref_name, emit)
    H_coarse_geo, H_fine_geo, shift, shifted = _dense_shift(levels, H_fine_geo, geo_warped, ohrc_sg, ref_sg, ref_name, emit)
    H_fine0 = _refine_coarse(levels, matcher, H_coarse_geo, H_fine_geo, shift, shifted, ohrc_sg, ref_sg, ref_name, emit)
    return CoarseResult(H_fine0=H_fine0, shift=shift, ohrc_pc=ohrc_pc, ref_pc=ref_pc,
                        difficulty=difficulty, preprocessed_image=preprocessed)


def _preprocess(pair, levels, ref_name, emit):
    """Step 5: plain stretches and matcher inputs, plus the pair difficulty (report only)."""
    step = Step(emit, 5, "Appearance Preprocessing")
    step.running(22)
    ohrc_pc, ref_pc, ohrc_sg, ref_sg = prep_pair(levels.ohrc_coarse, levels.ref_coarse)
    image = preprocessing_figure(ohrc_pc, ref_pc, ohrc_sg, ref_sg, ref_name)
    difficulty = assess_difficulty(pair.ohrc_sun, pair.ref_sun, levels.scale_ratio,
                                   texture_score(ohrc_pc), texture_score(ref_pc), ref_name)
    step.done(26, detail=f"CLAHE + histogram matching applied · Pair difficulty: {difficulty['level'].upper()} "
                         f"(score {difficulty['score']}) · " + "; ".join(difficulty["factors"]),
              image=image)
    return ohrc_pc, ref_pc, ohrc_sg, ref_sg, difficulty, image


def _orientation_check(ohrc_pc, ref_pc, emit):
    """Step 6 (diagnostic): does a flip/scale of the reference match the OHRC above the noise floor?"""
    step = Step(emit, 6, "Orientation Search (ZNCC)")
    step.running(28)
    rows, floor = ncc_orientation_search(ohrc_pc, ref_pc)
    top = rows[0] if rows else None
    margin = top[0] / max(floor, 1e-6) if top else 0
    if margin >= 2.5 and top and top[0] >= 0.12:
        verdict = "SAME GROUND"
    elif margin >= 1.5:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "NO DETECTABLE OVERLAP"
    clear_winner = verdict == "SAME GROUND" and len(rows) > 1 and top[0] / max(rows[1][0], 1e-6) >= 1.15
    candidates = 1 if clear_winner else len(rows)
    step.done(32, detail=f"Verdict: {verdict} · Margin ×{margin:.1f} · Best flip: {top[1] if top else '?'} · "
                         f"{candidates} candidate(s)",
              image=img_to_b64(grad_mag_u8(ohrc_pc)))


def _rotation_check(ohrc_pc, emit):
    """Step 7 (diagnostic): calibrate the rotation estimator's sign on a known 7 degree turn."""
    step = Step(emit, 7, "Residual Rotation Estimate")
    step.running(34)
    sign, _recovered, rotated = rotation_sign(ohrc_pc)
    step.done(36, detail=f"Self-test passed · Rotation sign: {sign:+.0f}", image=img_to_b64(to_uint8(rotated)))


def _keypoint_preview(ohrc_sg, ref_sg, matcher_name, emit):
    """Step 8 (diagnostic): keypoint matches on the raw coarse pair. Returns the matcher for step 11."""
    step = Step(emit, 8, "SuperPoint + SuperGlue")
    step.running(38)
    device = default_device()
    label = CHECKPOINTS[matcher_name][0]
    matcher = KeypointMatcher(matcher_name, device)
    k0, k1, _ = matcher.match(ohrc_sg, ref_sg)
    step.done(42, detail=f"Matcher: SuperPoint + {label} · Device: {device} · {len(k0)} coarse matches at threshold 0.15",
              image=img_to_b64(match_lines(ohrc_sg, ref_sg, k0, k1)))
    return matcher


def _geometric_prior(pair, levels, ohrc_sg, ref_sg, ref_name, emit):
    """Step 9: OHRC pixel -> lat/lon -> reference pixel through the two corner models.
    Returns (H_fine_geo, the coarse OHRC warped by it)."""
    step = Step(emit, 9, "Geometric Prior")
    step.running(44)
    lon_ref = pair.ohrc_corners["UL"]["lon"]
    M_o, t_o = corner_affine(pair.ohrc_corners, pair.ohrc_shape, lon_ref)
    M_r, t_r = corner_affine(pair.ref_corners, pair.ref_shape, lon_ref)
    R = np.linalg.inv(M_r) @ M_o
    T = np.linalg.inv(M_r) @ (t_o - t_r)
    # R and T act on (row, col); homographies act on (x, y) = (col, row), hence the swap.
    H_native_geo = np.array([[R[1, 1], R[1, 0], T[1]], [R[0, 1], R[0, 0], T[0]], [0.0, 0.0, 1.0]])
    H_fine_geo = levels.ref_native_to_fine @ H_native_geo @ np.linalg.inv(levels.ohrc_native_to_fine)

    A = H_native_geo[:2, :2]
    det = float(np.linalg.det(A))
    geo_warped = warp(ohrc_sg, levels.to_coarse(H_fine_geo), ref_sg.shape)
    step.done(48, detail=f"Scale ×{np.hypot(*A[:, 0]):.4f}, ×{np.hypot(*A[:, 1]):.4f} · "
                         f"{'MIRRORED' if det < 0 else 'Not mirrored'} · det={det:+.4f}",
              image=img_to_b64(overlay_preview(geo_warped, ref_sg,
                                               f"Step 9 - geometric prior | OHRC magenta, {ref_name} green, grey = aligned")))
    return H_fine_geo, geo_warped


def _band_shifts(warped_u8, ref_u8, valid_u8, n_bands=8, slack=160):
    """Shift of the prior-warped OHRC against the reference in horizontal bands along the strip."""
    h, w = ref_u8.shape
    a_all, b_all = zscore_grad(warped_u8), zscore_grad(ref_u8)
    edges = np.linspace(0, h, n_bands + 1).astype(int)
    rows = []
    for k in range(n_bands):
        y0, y1 = edges[k], edges[k + 1]
        if y1 - y0 < 64:
            continue
        cov = (valid_u8[y0:y1] > 0).mean()
        if cov < 0.35:
            continue
        tw = max(32, int(w * 0.6))
        x0 = (w - tw) // 2
        th = max(32, int((y1 - y0) * 0.8))
        ty = y0 + ((y1 - y0) - th) // 2
        tpl = a_all[ty:ty + th, x0:x0 + tw]
        sy0, sy1 = max(0, ty - slack), min(h, ty + th + slack)
        seg = b_all[sy0:sy1]
        if tpl.shape[0] > seg.shape[0] or tpl.shape[1] > seg.shape[1]:
            continue
        _, mx, _, loc = cv2.minMaxLoc(cv2.matchTemplate(seg, tpl, cv2.TM_CCOEFF_NORMED))
        null = max(cv2.minMaxLoc(cv2.matchTemplate(zscore(phase_randomize(ref_u8[sy0:sy1], seed=q)), tpl,
                                                   cv2.TM_CCOEFF_NORMED))[1] for q in range(2))
        rows.append(dict(yc=ty + th / 2.0, dx=loc[0] - x0, dy=(sy0 + loc[1]) - ty,
                         ncc=float(mx), margin=float(mx / max(null, 1e-6)), cov=float(cov)))
    return rows


def _dense_shift(levels, H_fine_geo, geo_warped, ohrc_sg, ref_sg, ref_name, emit):
    """Step 10: fit a 1st/2nd-order polynomial to the band shifts and fold its mid-strip value
    into the prior. Returns (H_coarse_geo, H_fine_geo, shift field, OHRC warped by the result)."""
    step = Step(emit, 10, "Dense Shift Field")
    step.running(50)
    H_coarse_geo = levels.to_coarse(H_fine_geo)
    valid = warp(np.full(ohrc_sg.shape, 255, np.uint8), H_coarse_geo, ref_sg.shape, cv2.INTER_NEAREST)
    bands = _band_shifts(geo_warped, ref_sg, valid)
    good = [r for r in bands if r["margin"] >= 2.0]

    shift = ShiftField(fine_to_coarse=levels.fine_to_coarse)
    if len(good) >= 3:
        yc = np.array([r["yc"] for r in good], float)
        dx = np.array([r["dx"] for r in good], float)
        dy = np.array([r["dy"] for r in good], float)
        deg = 2 if len(good) >= 5 else 1
        shift.coef = (np.polyfit(yc, dx, deg), np.polyfit(yc, dy, deg))
        shift.y_range = (float(yc.min()), float(yc.max()))
        mid_row = levels.ref_fine_shape[0] * levels.fine_to_coarse / 2.0
        shift.base = (float(np.polyval(shift.coef[0], mid_row)), float(np.polyval(shift.coef[1], mid_row)))
        H_coarse_geo = mat_translate(*shift.base) @ H_coarse_geo
        H_fine_geo = H_at_level(H_coarse_geo, levels.coarse_gsd, levels.fine_gsd)

    shifted = warp(ohrc_sg, H_coarse_geo, ref_sg.shape)
    step.done(54, detail=f"{len(good)}/{len(bands)} bands cleared null · "
                         f"{'Polynomial shift applied' if shift.coef is not None else 'No shift needed'}",
              image=img_to_b64(overlay_preview(shifted, ref_sg,
                                               f"Step 10 - after dense shift correction | OHRC magenta, {ref_name} green")))
    return H_coarse_geo, H_fine_geo, shift, shifted


def _refine_coarse(levels, matcher, H_coarse_geo, H_fine_geo, shift, shifted, ohrc_sg, ref_sg, ref_name, emit):
    """Step 11: keypoint matches on the prior-warped pair; the correction is kept only with 25+
    inliers and a centre move of at most 250 px. Returns the fine-level model H_fine0."""
    step = Step(emit, 11, "Coarse Refinement")
    step.running(56)
    pool0, pool1, _ = matcher.match(shifted, ref_sg)

    h, w = ref_sg.shape
    H_fine0 = H_fine_geo
    model = "geometric prior + dense shift" if shift.coef is not None else "geometric prior"
    refined = False
    if len(pool0) >= 4:
        C, mask, name = robust_fit(pool0, pool1, ref_sg.shape)
        if C is not None:
            centre_move = float(np.linalg.norm(apply_h(C, [[w / 2, h / 2]])[0] - [w / 2, h / 2]))
            if int(mask.sum()) >= 25 and centre_move <= 250:
                H_c = C @ H_coarse_geo
                if abs(np.linalg.det(H_c[:2, :2])) > 1e-9:
                    H_fine0 = H_at_level(H_c, levels.coarse_gsd, levels.fine_gsd)
                    model = f"geometric prior + {name} correction"
                    refined = True

    refined_warped = warp(ohrc_sg, levels.to_coarse(H_fine0), ref_sg.shape)
    step.done(60, detail=f"Model: {model} · {len(pool0)} matches · {'Correction applied' if refined else 'Using prior as-is'}",
              image=img_to_b64(overlay_preview(refined_warped, ref_sg,
                                               f"Step 11 - after coarse refinement | OHRC magenta, {ref_name} green")))
    return H_fine0
