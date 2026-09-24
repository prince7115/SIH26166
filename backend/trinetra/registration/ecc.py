"""Step 14: ECC intensity refinement of the fitted model (gated, legacy or off)."""
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from ..geometry.transforms import H_at_level, apply_h, mat_translate, to_h
from ..imaging.radiometry import grad_mag_u8, to_uint8
from ..imaging.resample import level_tile, pixels_at


@dataclass
class FineFrame:
    """What ECC and its scoring need to render either product at any level."""
    ohrc_crop: object
    ref_crop: object
    ohrc_gsd: float
    ref_gsd: float
    fine_gsd: float
    coarse_gsd: float
    ohrc_fine_shape: tuple
    level_view: Callable

    @classmethod
    def from_levels(cls, levels):
        return cls(ohrc_crop=levels.ohrc_crop, ref_crop=levels.ref_crop,
                   ohrc_gsd=levels.ohrc_gsd, ref_gsd=levels.ref_gsd,
                   fine_gsd=levels.fine_gsd, coarse_gsd=levels.coarse_gsd,
                   ohrc_fine_shape=levels.ohrc_fine_shape, level_view=levels.level_view)


def _ecc_refine(fr, H_in, gsd, model_name, iters=200, eps=1e-6):
    """One findTransformECC pass on gradient magnitude at gsd. Returns (fine-level H, correlation)."""
    o_g = grad_mag_u8(to_uint8(fr.level_view(fr.ohrc_crop, fr.ohrc_gsd, gsd)))
    n_g = grad_mag_u8(to_uint8(fr.level_view(fr.ref_crop, fr.ref_gsd, gsd)))
    W = H_at_level(H_in, fr.fine_gsd, gsd).astype(np.float32)
    motion = cv2.MOTION_HOMOGRAPHY if model_name == "homography" else cv2.MOTION_AFFINE
    warp = W.astype(np.float32) if motion == cv2.MOTION_HOMOGRAPHY else W[:2].astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    # ECC's warp maps template (OHRC) -> input (reference), the same direction as ours.
    cc, warp = cv2.findTransformECC(o_g, n_g, warp, motion, crit, None, 5)
    W_ref = warp if motion == cv2.MOTION_HOMOGRAPHY else to_h(warp)
    return H_at_level(W_ref, gsd, fr.fine_gsd), float(cc)


def fine_alignment_score(fr, H, n_tiles=6, size=640):
    """Mean ECC correlation over fine-level tiles, with the reference rendered into each OHRC tile."""
    rows = np.linspace(0, max(1, fr.ohrc_fine_shape[0] - size), max(2, n_tiles)).astype(int)
    cols = np.linspace(0, max(1, fr.ohrc_fine_shape[1] - size), 2).astype(int)
    vals = []
    for r0 in rows:
        for c0 in cols:
            o_tile, o_org = level_tile(fr.ohrc_crop, fr.ohrc_gsd, fr.fine_gsd, int(r0), int(c0), size, size)
            if o_tile is None:
                continue
            ox, oy = o_org
            oh, ow = o_tile.shape[:2]
            corners = np.array([[ox, oy], [ox + ow, oy], [ox, oy + oh], [ox + ow, oy + oh]])
            proj = apply_h(H, corners)
            n_r0 = int(np.floor(proj[:, 1].min())) - 8
            n_c0 = int(np.floor(proj[:, 0].min())) - 8
            n_h = int(np.ceil(proj[:, 1].max() - proj[:, 1].min())) + 16
            n_w = int(np.ceil(proj[:, 0].max() - proj[:, 0].min())) + 16
            if n_h < 32 or n_w < 32:
                continue
            n_tile, n_org = level_tile(fr.ref_crop, fr.ref_gsd, fr.fine_gsd, max(0, n_r0), max(0, n_c0), n_h, n_w)
            if n_tile is None:
                continue
            nx, ny = n_org
            H_tile = mat_translate(-nx, -ny) @ H @ mat_translate(ox, oy)
            n_in_o = cv2.warpPerspective(to_uint8(n_tile), H_tile.astype(np.float32), (ow, oh),
                                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
            m = (n_in_o > 0).astype(np.uint8) * 255
            if m.mean() / 255.0 < 0.5:
                continue
            try:
                vals.append(float(cv2.computeECC(grad_mag_u8(to_uint8(o_tile)), grad_mag_u8(n_in_o), m)))
            except cv2.error:
                continue
            if len(vals) >= n_tiles:
                break
        if len(vals) >= n_tiles:
            break
    return float(np.mean(vals)) if vals else None


def legacy_ecc(fr, H_fine, model_name):
    """One ECC pass at the coarse level, kept whenever it is finite."""
    try:
        H_try, cc = _ecc_refine(fr, H_fine, fr.coarse_gsd, model_name)
    except cv2.error:
        return H_fine, False, {"mode": "legacy", "reason": "ECC did not converge"}
    if np.all(np.isfinite(H_try)):
        return H_try, True, {"mode": "legacy", "cc": cc}
    return H_fine, False, {"mode": "legacy", "reason": "ECC returned a non-finite warp"}


def gated_ecc(fr, H_fine, model_name, px_budget=6_000_000):
    """ECC at levels from coarse to fine (halving the GSD within a pixel budget), each result
    kept only if it raises the fine-level alignment score. Worst case: the input unchanged."""
    info = {"mode": "gated", "tried": []}
    score_before = fine_alignment_score(fr, H_fine)
    info["score_before"] = score_before
    if score_before is None:
        info["reason"] = "no fine tiles to score on; ECC skipped"
        return H_fine, False, info

    ladder, g = [fr.coarse_gsd], fr.coarse_gsd / 2.0
    while g > fr.fine_gsd * 0.99 and max(pixels_at(fr.ohrc_crop.shape, fr.ohrc_gsd, g),
                                         pixels_at(fr.ref_crop.shape, fr.ref_gsd, g)) <= px_budget:
        ladder.append(g)
        g /= 2.0

    best_H, best_score, best_g = H_fine, score_before, None
    for g in ladder:
        try:
            H_try, _ = _ecc_refine(fr, H_fine, g, model_name)
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
