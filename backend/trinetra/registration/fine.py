"""Step 12: dense matches in overlapping fine-level tiles, each pre-warped by the coarse model."""
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import cv2
import numpy as np

from .options import any_rescue
from .steps import Step, with_note
from ..geometry.transforms import H_at_level, apply_h, mat_translate, warp
from ..imaging.radiometry import texture_score, to_uint8
from ..imaging.resample import level_tile
from ..matching.craters import crater_matches
from ..matching.dense import match_dense
from ..matching.rescue import rescue_tile, tile_consensus
from ..reporting.encoding import img_to_b64
from ..reporting.previews import points_preview

log = logging.getLogger(__name__)

TILE_PX = 640
MAX_TILES = 400
TILE_OVERLAP = 0.25
SEARCH_MARGIN = 128       # reference window margin around the projected tile, fine px
MIN_VALID_FRAC = 0.25     # of the reference window covered by the warped tile
MIN_TEXTURE = 12.0
TILE_CONSENSUS_PX = 6.0
MAX_WORKERS = 14


def tile_size(profile, ohrc_fine_shape):
    if profile.adaptive_tile:
        # About four tile columns across the OHRC overlap, clamped to [256, 640].
        return int(np.clip(2 ** int(np.floor(np.log2(max(256.0, ohrc_fine_shape[1] / 4.0)))), 256, TILE_PX))
    return TILE_PX


@dataclass
class TileMatches:
    fine_o: np.ndarray       # OHRC fine pixels (x, y)
    fine_n: np.ndarray       # reference fine pixels (x, y)
    sources: np.ndarray      # "dense", "rescue: <method>" or "crater" per point
    tiles_tried: int
    tiles_used: int
    rescued_by: dict
    crater_stats: dict


class TileMatcher:
    def __init__(self, levels, coarse, tile, emit):
        self.levels, self.coarse, self.tile, self.emit = levels, coarse, tile, emit
        stride = max(1, int(tile * (1.0 - TILE_OVERLAP)))
        rows = range(0, max(1, levels.ohrc_fine_shape[0] - tile // 2), stride)
        cols = range(0, max(1, levels.ohrc_fine_shape[1] - tile // 2), stride)
        self.jobs = [(r0, c0) for r0 in rows for c0 in cols][:MAX_TILES]
        self.workers = min(os.cpu_count() or 4, MAX_WORKERS)
        self._craters = None

    def _match_tile(self, r0, c0, opts):
        """Correspondences of one tile as (pts_ohrc_fine, pts_ref_fine, rescue method), or None."""
        lv, H0 = self.levels, self.coarse.H_fine0
        o_tile, o_origin = level_tile(lv.ohrc_crop, lv.ohrc_gsd, lv.fine_gsd, r0, c0, self.tile, self.tile)
        if o_tile is None:
            return None
        ox, oy = o_origin
        oh, ow = o_tile.shape[:2]
        if texture_score(to_uint8(o_tile)) < MIN_TEXTURE:
            return None

        # Local model: the coarse model plus the along-strip drift at this tile's row.
        corners = np.array([[ox, oy], [ox + ow, oy], [ox, oy + oh], [ox + ow, oy + oh]])
        row = apply_h(H0, [[ox + ow / 2.0, oy + oh / 2.0]])[0][1]
        H_local = mat_translate(*self.coarse.shift.at_fine_row(row)) @ H0
        proj = apply_h(H_local, corners)
        n_c0 = int(np.floor(proj[:, 0].min())) - SEARCH_MARGIN
        n_r0 = int(np.floor(proj[:, 1].min())) - SEARCH_MARGIN
        n_w = int(np.ceil(proj[:, 0].max() - proj[:, 0].min())) + 2 * SEARCH_MARGIN
        n_h = int(np.ceil(proj[:, 1].max() - proj[:, 1].min())) + 2 * SEARCH_MARGIN
        if n_w < 32 or n_h < 32 or n_r0 > lv.ref_fine_shape[0] or n_c0 > lv.ref_fine_shape[1]:
            return None
        n_tile, n_origin = level_tile(lv.ref_crop, lv.ref_gsd, lv.fine_gsd, max(0, n_r0), max(0, n_c0), n_h, n_w)
        if n_tile is None:
            return None
        nx, ny = n_origin
        nh, nw = n_tile.shape[:2]

        H_tile = mat_translate(-nx, -ny) @ H_local @ mat_translate(ox, oy)
        valid = warp(np.full((oh, ow), 255, np.uint8), H_tile, (nh, nw), cv2.INTER_NEAREST)
        if valid.mean() / 255.0 < MIN_VALID_FRAC:
            return None
        vmask = valid > 0
        o_pc = to_uint8(warp(np.asarray(o_tile), H_tile, (nh, nw)), mask=vmask)
        n_pc = to_uint8(n_tile)
        if texture_score(n_pc) < MIN_TEXTURE:
            return None

        agreed = tile_consensus(*match_dense(o_pc, n_pc, vmask), TILE_CONSENSUS_PX)
        rescued_by = None
        if agreed is None:
            if not any_rescue(opts):
                return None
            agreed, rescued_by = rescue_tile(o_pc, n_pc, vmask, opts, TILE_CONSENSUS_PX)
            if agreed is None:
                return None
        ka, kb, _ = agreed
        return (apply_h(np.linalg.inv(H_local) @ mat_translate(nx, ny), ka),
                apply_h(mat_translate(nx, ny), kb), rescued_by)

    def _crater_matches(self):
        """Crater correspondences in fine pixels; they depend only on the coarse model, so they
        are computed once and reused by every attempt."""
        if self._craters is None:
            lv = self.levels
            c_o, c_n, _, stats = crater_matches(self.coarse.ohrc_pc, self.coarse.ref_pc,
                                                H_at_level(self.coarse.H_fine0, lv.fine_gsd, lv.coarse_gsd))
            self._craters = (c_o / lv.fine_to_coarse, c_n / lv.fine_to_coarse, stats)
        return self._craters

    def match(self, opts, note, progress):
        step = Step(self.emit, 12, "Tiled Matching at Native Resolution")
        step.running(progress(62), detail=note)
        pts_o, pts_n, sources = [], [], []
        tried = used = 0
        rescued_by = {}
        started = time.time()

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._match_tile, r0, c0, opts) for r0, c0 in self.jobs]
            for future in as_completed(futures):
                tried += 1
                try:
                    result = future.result()
                except Exception:
                    log.exception("Tile matching failed")
                    continue
                if result is not None:
                    o, n, method = result
                    pts_o.append(o)
                    pts_n.append(n)
                    sources.extend(["dense" if method is None else f"rescue: {method}"] * len(o))
                    if method is not None:
                        rescued_by[method] = rescued_by.get(method, 0) + 1
                    used += 1
                if tried % 20 == 0:
                    pct = 62 + int(15 * tried / max(len(self.jobs), 1))
                    step.running(progress(min(pct, 77)), detail=with_note(
                        note, f"{tried}/{len(self.jobs)} tiles · {used} matched · "
                              f"{sum(len(p) for p in pts_o)} points · {self.workers} threads"))

        crater_stats = None
        if opts["crater_matching"]:
            c_o, c_n, crater_stats = self._crater_matches()
            if len(c_o):
                pts_o.append(c_o)
                pts_n.append(c_n)
                sources.extend(["crater"] * len(c_o))

        if not pts_o:
            raise RuntimeError("No tiles produced matches.")
        result = TileMatches(fine_o=np.vstack(pts_o), fine_n=np.vstack(pts_n), sources=np.asarray(sources),
                             tiles_tried=tried, tiles_used=used, rescued_by=rescued_by, crater_stats=crater_stats)

        pc = result.fine_n * self.levels.fine_to_coarse
        src = result.sources
        rescued = np.char.startswith(src.astype(str), "rescue")
        preview = points_preview(self.levels.ref_coarse, [
            (pc[src == "dense"][:2000], (0, 255, 0)),
            (pc[rescued][:2000], (0, 165, 255)),
            (pc[src == "crater"][:2000], (255, 255, 0)),
        ], f"Step 12 - {len(result.fine_n)} matches: green dense, orange rescued, cyan crater")
        extra = ""
        if any_rescue(opts):
            extra += f" · {sum(rescued_by.values())} tiles rescued"
        if crater_stats is not None:
            extra += f" · {crater_stats['matched']} crater matches"
        step.done(progress(78), detail=with_note(
            note, f"Tile {self.tile} px · {used}/{tried} tiles matched · {len(result.fine_o)} correspondences"
                  f"{extra} · {time.time() - started:.0f}s"),
                  image=img_to_b64(preview))
        return result
