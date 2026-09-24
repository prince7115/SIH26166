"""Steps 12-14 and their evaluation, run once per attempt.

With auto_retry, further attempts switch on more tile rescues while the result is not trusted;
the best attempt is kept.
"""
from dataclasses import dataclass

import numpy as np

from .ecc import FineFrame, gated_ecc, legacy_ecc
from .evaluation import nmi_with_baseline, random_heldout_rmse, rmse, spatial_heldout, trust_verdict
from .fine import TileMatches
from .options import retry_ladder
from .steps import Step, with_note
from ..matching.fitting import hull_coverage, residual_scale, robust_fit
from ..reporting.encoding import img_to_b64
from ..reporting.previews import overlay_preview, points_preview


@dataclass
class ModelFit:
    H_fine: np.ndarray
    inlier_mask: np.ndarray
    model_name: str
    thresh: float
    n_inliers: int
    inlier_ratio: float
    coverage: float
    trusted_legacy: bool      # the old count-only check, kept in metrics.json


@dataclass
class Evaluation:
    src: np.ndarray           # inlier tie points, OHRC fine pixels
    dst: np.ndarray           # and reference fine pixels
    warped: np.ndarray        # coarse OHRC under the final model
    rmse_in: float
    rmse_pre_ecc: float
    rmse_heldout: float
    rmse_heldout_spatial: float
    heldout_spatial_folds: int
    nmi: float
    nmi_baseline: float
    trusted: bool
    trust_checks: list


@dataclass
class Attempt:
    index: int
    options: dict
    tiles: TileMatches
    fit: ModelFit
    H_final: np.ndarray
    ecc_applied: bool
    ecc_info: dict
    evaluation: Evaluation


class AttemptRunner:
    def __init__(self, levels, tile_matcher, ref_name, emit):
        self.levels, self.tiles, self.ref_name, self.emit = levels, tile_matcher, ref_name, emit
        self.frame = FineFrame.from_levels(levels)

    def fit_model(self, tiles, note, progress):
        """Step 13: RANSAC threshold from the residual scale, then the guarded fit."""
        lv = self.levels
        step = Step(self.emit, 13, "Final Model (RANSAC)")
        step.running(progress(80), detail=note)
        thresh = residual_scale(tiles.fine_o, tiles.fine_n, lv.ref_fine_shape) or 3.0
        H_fine, inliers, model_name = robust_fit(tiles.fine_o, tiles.fine_n, lv.ref_fine_shape,
                                                 thresh=thresh, min_minor_axis_px=5.0)
        if H_fine is None:
            raise RuntimeError("Could not fit a model to the fine correspondences.")
        n_inliers = int(inliers.sum())
        coverage = hull_coverage(tiles.fine_n[inliers], lv.ref_fine_shape)
        fit = ModelFit(H_fine=H_fine, inlier_mask=inliers, model_name=model_name, thresh=thresh,
                       n_inliers=n_inliers, inlier_ratio=n_inliers / len(tiles.fine_o), coverage=coverage,
                       trusted_legacy=n_inliers >= 15 and coverage >= 0.05)

        pc = tiles.fine_n * lv.fine_to_coarse
        preview = points_preview(lv.ref_coarse, [
            (pc[~inliers][:3000], (0, 0, 255)),
            (pc[inliers][:3000], (0, 255, 0)),
        ], f"Step 13 - {model_name}: {n_inliers} inliers green, {len(tiles.fine_o) - n_inliers} outliers red")
        step.done(progress(83), detail=with_note(
            note, f"Model: {model_name} · {n_inliers}/{len(tiles.fine_o)} inliers ({fit.inlier_ratio:.1%}) · "
                  f"Coverage {fit.coverage:.1%} · basic checks {'pass' if fit.trusted_legacy else 'fail'} "
                  f"(final verdict in step 18)"),
                  image=img_to_b64(preview))
        return fit

    def refine(self, fit, opts, note, progress):
        """Step 14. Returns (H_final, applied, info, coarse OHRC warped by H_final)."""
        step = Step(self.emit, 14, "ECC Intensity Refinement")
        step.running(progress(85), detail=note)
        mode = opts["ecc_mode"]
        if mode == "off":
            H_final, applied, info = fit.H_fine, False, {"mode": "off"}
            msg = "Off for this run: tiled fit used as-is"
        elif mode == "legacy":
            H_final, applied, info = legacy_ecc(self.frame, fit.H_fine, fit.model_name)
            msg = "Legacy: " + ("applied without a quality check" if applied else info.get("reason", "not applied"))
        else:
            H_final, applied, info = gated_ecc(self.frame, fit.H_fine, fit.model_name)
            if info.get("score_before") is None:
                msg = "Gated: " + info.get("reason", "skipped")
            elif applied:
                msg = (f"Gated: fine-tile score {info['score_before']:.4f} → {info['score_after']:.4f} · "
                       f"kept ECC from the {info['kept_gsd_m']} m/px level")
            else:
                msg = (f"Gated: no level improved the fine-tile score ({info['score_before']:.4f}) · "
                       f"ECC rejected, tiled fit kept")
        warped = self.levels.warp_ohrc_coarse(H_final)
        caption = (f"Step 14 - final fit ({'ECC applied' if applied else 'ECC not applied'}) | "
                   f"OHRC magenta, {self.ref_name} green")
        step.done(progress(87), detail=with_note(note, msg),
                  image=img_to_b64(overlay_preview(warped, self.levels.ref_coarse_u8, caption)))
        return H_final, applied, info, warped

    def evaluate(self, tiles, fit, H_final, warped):
        """Accuracy and trust of one attempt (reported in step 18)."""
        shape = self.levels.ref_fine_shape
        src, dst = tiles.fine_o[fit.inlier_mask], tiles.fine_n[fit.inlier_mask]
        rmse_heldout = random_heldout_rmse(src, dst, shape, fit.thresh)
        rmse_spatial, folds = spatial_heldout(src, dst, shape, fit.thresh)
        nmi, nmi_base = nmi_with_baseline(warped, self.levels.ref_coarse_u8)
        # The spatial held-out figure is preferred; the random split is the fallback.
        held = rmse_spatial if rmse_spatial is not None else rmse_heldout
        trusted, checks = trust_verdict(fit.n_inliers, fit.coverage, held, nmi, nmi_base)
        return Evaluation(src=src, dst=dst, warped=warped,
                          # ECC optimises intensity, not point residuals; both are reported.
                          rmse_in=rmse(H_final, src, dst), rmse_pre_ecc=rmse(fit.H_fine, src, dst),
                          rmse_heldout=rmse_heldout, rmse_heldout_spatial=rmse_spatial,
                          heldout_spatial_folds=folds, nmi=nmi, nmi_baseline=nmi_base,
                          trusted=trusted, trust_checks=checks)

    def run_attempt(self, index, opts, note):
        # Retries keep the progress bar where the first attempt left it.
        progress = (lambda v: v) if index == 0 else (lambda v: 87)
        tiles = self.tiles.match(opts, note, progress)
        fit = self.fit_model(tiles, note, progress)
        H_final, applied, info, warped = self.refine(fit, opts, note, progress)
        return Attempt(index=index, options=opts, tiles=tiles, fit=fit, H_final=H_final,
                       ecc_applied=applied, ecc_info=info,
                       evaluation=self.evaluate(tiles, fit, H_final, warped))

    def run(self, opts):
        """All attempts of the retry ladder. Returns (best attempt, attempt log)."""
        ladder = retry_ladder(opts)
        attempts, log = [], []
        for index, (label, attempt_opts) in enumerate(ladder):
            if attempts and attempts[-1].evaluation.trusted:
                break
            note = None if index == 0 else f"Retry {index}/{len(ladder) - 1}: {label}"
            try:
                attempt = self.run_attempt(index, attempt_opts, note)
            except RuntimeError as e:
                if not opts["auto_retry"]:
                    raise
                log.append({"attempt": index, "label": label, "error": str(e)})
                continue
            attempts.append(attempt)
            log.append({"attempt": index, "label": label, "trusted": attempt.evaluation.trusted,
                        "n_inliers": attempt.fit.n_inliers,
                        "rmse_px_heldout_spatial": attempt.evaluation.rmse_heldout_spatial,
                        "tiles_matched": attempt.tiles.tiles_used})
        if not attempts:
            raise RuntimeError("Every attempt failed: " + "; ".join(a["error"] for a in log))
        return min(attempts, key=_rank), log


def _rank(attempt):
    """Trusted first, then lowest spatial held-out RMSE, then most inliers."""
    ev = attempt.evaluation
    h = ev.rmse_heldout_spatial
    return (not ev.trusted, h if h is not None else float("inf"), -attempt.fit.n_inliers)
