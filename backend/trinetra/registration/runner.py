"""Steps 3-18 for a loaded pair."""
import json
import os

import numpy as np

from .attempts import AttemptRunner
from .coarse import align_coarse
from .fine import TileMatcher, tile_size
from .levels import choose_levels, crop_overlap
from .options import resolve_options
from .steps import Step
from ..geometry.transforms import apply_h
from ..reporting.encoding import img_to_b64
from ..reporting.figures import correspondence_figure, overlay_figure
from ..reporting.previews import points_preview


def run_registration(pair_id, pair, profile, options, emit, out_dir):
    """Register a loaded pair, write its outputs to out_dir and return the metrics."""
    opts = resolve_options(profile.options, options)
    ref_name = profile.ref_name

    levels = choose_levels(pair, profile, crop_overlap(pair, ref_name, emit), emit)
    coarse = align_coarse(pair, levels, ref_name, opts, emit)
    tiles = TileMatcher(levels, coarse, tile_size(profile, levels.ohrc_fine_shape), emit)
    best, attempt_log = AttemptRunner(levels, tiles, ref_name, emit).run(opts)
    chosen = "" if len(attempt_log) == 1 else f" · using attempt {best.index} of {len(attempt_log)}"

    step = Step(emit, 15, "Sub-pixel Refinement")
    step.running(88)
    step.done(89, detail="Using model as-is for this run")

    inliers_by_source = _report_tie_points(best, levels, chosen, emit)
    images = {"preprocessed": coarse.preprocessed_image}
    images.update(_export(pair_id, best, levels, profile, out_dir, emit))

    step = Step(emit, 18, "Evaluation & Report")
    step.running(97)
    metrics = _metrics(pair_id, pair, profile, opts, levels, coarse, tiles, best, attempt_log, inliers_by_source)
    with open(os.path.join(out_dir, f"{profile.output_tag(pair_id)}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    step.done(100, detail=_summary(best, levels.fine_gsd) + chosen, metrics=metrics, images=images)
    return metrics


def _report_tie_points(best, levels, chosen, emit):
    """Step 16: residual of every inlier under the final model. Returns inlier counts by source."""
    step = Step(emit, 16, "Consensus Tie Points")
    step.running(90)
    ev = best.evaluation
    residual = np.linalg.norm(apply_h(best.H_final, ev.src) - ev.dst, axis=1)
    sources = best.tiles.sources[best.fit.inlier_mask]
    by_source = {str(k): int(v) for k, v in zip(*np.unique(sources, return_counts=True))}

    res_max = max(float(np.percentile(residual, 95)), 1.0)
    r = np.minimum(residual / res_max, 1.0)
    colours = np.stack([np.zeros_like(r), 255 * (1 - r), 255 * r], axis=1)    # BGR: green low, red high
    preview = points_preview(levels.ref_coarse, [(ev.dst * levels.fine_to_coarse, colours)],
                             f"Step 16 - {len(ev.src)} tie points by residual: green low, red >= {res_max:.1f} px")
    split = " · " + ", ".join(f"{v} {k}" for k, v in by_source.items()) if len(by_source) > 1 else ""
    step.done(91, detail=f"{len(ev.src)} tie points{split} · Residual mean {residual.mean():.3f} px = "
                         f"{residual.mean() * levels.fine_gsd * 100:.1f} cm{chosen}",
              image=img_to_b64(preview))
    return by_source


def _export(pair_id, best, levels, profile, out_dir, emit):
    """Step 17: result figures, and the homography at native and fine resolution (.npy)."""
    step = Step(emit, 17, "Final Warp & Export")
    step.running(92)
    ev = best.evaluation
    H_native = np.linalg.inv(levels.ref_native_to_fine) @ best.H_final @ levels.ohrc_native_to_fine
    overlay = overlay_figure(ev.warped, levels.ref_coarse_u8, profile.ref_name)
    k = levels.fine_to_coarse
    matches = correspondence_figure(levels.ohrc_coarse_u8, levels.ref_coarse_u8, ev.src * k, ev.dst * k)

    os.makedirs(out_dir, exist_ok=True)
    tag = profile.output_tag(pair_id)
    np.save(os.path.join(out_dir, f"{tag}_H_native.npy"), H_native)
    np.save(os.path.join(out_dir, f"{tag}_H_fine.npy"), best.H_final)
    step.done(96, detail=f"Warped raster + homography saved to {out_dir}", image=overlay)
    return {"overlay": overlay, "matches": matches}


def _metrics(pair_id, pair, profile, opts, levels, coarse, tiles, best, attempt_log, inliers_by_source):
    fit, ev, t = best.fit, best.evaluation, best.tiles
    fine = levels.fine_gsd
    return {
        "pair_id": pair_id, "pipeline": profile.key, "version": profile.version,
        "reference": profile.ref_name, "model": fit.model_name,
        "ohrc_product_id": pair.ohrc_product_id, f"{profile.ref_name.lower()}_product_id": pair.ref_product_id,
        "n_correspondences": int(len(t.fine_o)), "n_inliers": fit.n_inliers,
        "inlier_ratio": float(fit.inlier_ratio), "spatial_coverage_pct": float(fit.coverage * 100),
        "rmse_px_in_sample": ev.rmse_in, "rmse_m_in_sample": ev.rmse_in * fine,
        "rmse_px_in_sample_pre_ecc": ev.rmse_pre_ecc,
        "rmse_px_heldout": ev.rmse_heldout,
        "rmse_m_heldout": (ev.rmse_heldout * fine) if ev.rmse_heldout is not None else None,
        "rmse_px_heldout_spatial": ev.rmse_heldout_spatial,
        "rmse_m_heldout_spatial": (ev.rmse_heldout_spatial * fine) if ev.rmse_heldout_spatial is not None else None,
        "heldout_spatial_folds": ev.heldout_spatial_folds,
        "nmi_aligned": ev.nmi, "nmi_misaligned_baseline": ev.nmi_baseline,
        "ecc_applied": bool(best.ecc_applied), "ecc_mode": best.options["ecc_mode"], "ecc_info": best.ecc_info,
        "trusted": bool(ev.trusted), "trusted_legacy": bool(fit.trusted_legacy), "trust_checks": ev.trust_checks,
        "tiles_tried": t.tiles_tried, "tiles_matched": t.tiles_used, "tile_px": tiles.tile,
        "tiles_rescued": int(sum(t.rescued_by.values())), "rescued_by": t.rescued_by,
        "crater_stats": t.crater_stats, "inliers_by_source": inliers_by_source,
        "gsd_fine_m": fine, "gsd_coarse_m": levels.coarse_gsd,
        "fine_gsd_multiplier": profile.fine_gsd_multiplier,
        "scale_ratio": round(pair.ref_gsd / pair.ohrc_gsd, 3),
        "difficulty": coarse.difficulty, "ohrc_sun": pair.ohrc_sun, "ref_sun": pair.ref_sun,
        "matcher": opts["matcher"], "options": best.options,
        "attempts": attempt_log, "selected_attempt": best.index,
        **pair.extra_metrics,
    }


def _summary(best, fine_gsd):
    ev = best.evaluation
    held = (f"held-out {ev.rmse_heldout_spatial:.3f} px ({ev.rmse_heldout_spatial * fine_gsd:.2f} m)"
            if ev.rmse_heldout_spatial is not None else "held-out n/a")
    failed = [c["name"] for c in ev.trust_checks if not c["pass"]]
    verdict = "TRUSTED" if ev.trusted else f"LOW CONFIDENCE (failed: {', '.join(failed)})"
    return (f"RMSE {ev.rmse_in:.3f} px ({ev.rmse_in * fine_gsd:.2f} m) · {held} · "
            f"NMI {(f'{ev.nmi:.4f}' if ev.nmi else 'N/A')} · {verdict}")
