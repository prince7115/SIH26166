"""Steps 3-4: crop both products to their shared ground and choose the working resolutions.

Two levels are used: a coarse one (whole overlap within a pixel budget) for global alignment,
and a fine one (the coarser product's native GSD, times the profile multiplier) for tie points.
"""
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .steps import Step
from ..geometry.geodesy import KM_PER_DEG_LAT, km_per_deg_lon, latlon_to_pixel, unwrap_lon
from ..geometry.transforms import H_at_level, mat_scale, mat_translate, warp
from ..imaging.radiometry import stretch_to_uint8, to_uint8
from ..imaging.resample import level_shape, level_view, level_view_antialiased, pixels_at
from ..reporting.encoding import img_to_b64
from ..reporting.previews import side_by_side

PAD_M = 150.0                    # ground margin kept around the overlap polygon
COARSE_PIXEL_BUDGET = 2_000_000  # the coarse GSD grows by 1.25x until both crops fit this


@dataclass
class Levels:
    ohrc_crop: np.ndarray
    ref_crop: np.ndarray
    ohrc_gsd: float                     # native GSDs
    ref_gsd: float
    fine_gsd: float
    coarse_gsd: float
    ohrc_fine_shape: tuple
    ref_fine_shape: tuple
    ohrc_native_to_fine: np.ndarray     # full-product pixels -> fine crop pixels
    ref_native_to_fine: np.ndarray
    ohrc_coarse: np.ndarray
    ref_coarse: np.ndarray
    level_view: Callable
    scale_ratio: float                  # coarser / finer native GSD

    def __post_init__(self):
        self.ohrc_coarse_u8 = to_uint8(self.ohrc_coarse)
        self.ref_coarse_u8 = to_uint8(self.ref_coarse)

    @property
    def fine_to_coarse(self):
        """Scale from fine-level to coarse-level pixels."""
        return self.fine_gsd / self.coarse_gsd

    def to_coarse(self, H_fine):
        return H_at_level(H_fine, self.fine_gsd, self.coarse_gsd)

    def warp_ohrc_coarse(self, H_fine):
        """Coarse OHRC (8-bit) in the coarse reference frame under a fine-level model."""
        return warp(self.ohrc_coarse_u8, self.to_coarse(H_fine), self.ref_coarse.shape)


def _corner_bounds(corners, lon_ref):
    lats = [c["lat"] for c in corners.values()]
    lons = [float(unwrap_lon(c["lon"], lon_ref)) for c in corners.values()]
    return min(lats), max(lats), min(lons), max(lons)


def _footprint_polygon(corners, lon_ref):
    """(lon, lat) vertices in ring order."""
    return np.array([[float(unwrap_lon(corners[k]["lon"], lon_ref)), corners[k]["lat"]]
                     for k in ("UL", "UR", "LR", "LL")], np.float32)


def _crop_to_polygon(raster, latlon_to_px, polygon, gsd, name):
    """Window of raster around a (lon, lat) polygon, padded by PAD_M (at most 15% of its size)."""
    n_lines, n_samples = raster.shape[:2]
    pix = [latlon_to_px(float(v[1]), float(v[0])) for v in polygon]
    rows = [p[0] for p in pix]
    cols = [p[1] for p in pix]
    ext_r, ext_c = max(rows) - min(rows), max(cols) - min(cols)
    pad_px = min(PAD_M / max(gsd, 1e-6), 0.15 * max(min(ext_r, ext_c), 1.0))
    r0 = int(max(0, np.floor(min(rows) - pad_px)))
    r1 = int(min(n_lines, np.ceil(max(rows) + pad_px)))
    c0 = int(max(0, np.floor(min(cols) - pad_px)))
    c1 = int(min(n_samples, np.ceil(max(cols) + pad_px)))
    if r0 >= r1 or c0 >= c1:
        raise ValueError(f"{name}: empty crop window")
    return raster[r0:r1, c0:c1], (r0, r1, c0, c1)


def crop_overlap(pair, ref_name, emit):
    """Step 3. Returns (ohrc_crop, ref_crop, ohrc_window, ref_window); windows are (r0, r1, c0, c1)."""
    ohrc_to_px = latlon_to_pixel(pair.ohrc_corners, pair.ohrc_shape)
    ref_to_px = latlon_to_pixel(pair.ref_corners, pair.ref_shape)
    step = Step(emit, 3, "Overlap Detection & Crop")
    step.running(12)

    lon_ref = pair.ohrc_corners["UL"]["lon"]
    o_lat0, o_lat1, o_lon0, o_lon1 = _corner_bounds(pair.ohrc_corners, lon_ref)
    n_lat0, n_lat1, n_lon0, n_lon1 = _corner_bounds(pair.ref_corners, lon_ref)
    ov_lat = min(o_lat1, n_lat1) - max(o_lat0, n_lat0)
    ov_lon = min(o_lon1, n_lon1) - max(o_lon0, n_lon0)
    if ov_lat <= 0 or ov_lon <= 0:
        raise RuntimeError(f"OHRC and {ref_name} footprints do not overlap.")

    area, polygon = cv2.intersectConvexConvex(_footprint_polygon(pair.ohrc_corners, lon_ref),
                                              _footprint_polygon(pair.ref_corners, lon_ref))
    if area > 0 and polygon is not None and len(polygon) >= 3:
        polygon = np.asarray(polygon, np.float64).reshape(-1, 2)
    else:
        lat0, lat1 = max(o_lat0, n_lat0), min(o_lat1, n_lat1)
        lon0, lon1 = max(o_lon0, n_lon0), min(o_lon1, n_lon1)
        polygon = np.array([[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1]])

    ohrc_crop, ohrc_win = _crop_to_polygon(pair.ohrc_raster, ohrc_to_px, polygon, pair.ohrc_gsd, "OHRC")
    ref_crop, ref_win = _crop_to_polygon(pair.ref_raster, ref_to_px, polygon, pair.ref_gsd, ref_name)
    # 16-bit products (TMC-2) are stretched to 8 bit here, on the crop, never on the full strip.
    if ohrc_crop.dtype != np.uint8:
        ohrc_crop = stretch_to_uint8(ohrc_crop)
    if ref_crop.dtype != np.uint8:
        ref_crop = stretch_to_uint8(ref_crop)

    mid_lat = 0.5 * (max(o_lat0, n_lat0) + min(o_lat1, n_lat1))
    step.done(16, detail=f"Overlap: {ov_lat * KM_PER_DEG_LAT:.1f} × {ov_lon * km_per_deg_lon(mid_lat):.1f} km · "
                         f"OHRC crop {ohrc_crop.shape} · {ref_name} crop {ref_crop.shape}",
              image=img_to_b64(side_by_side(ohrc_crop, ref_crop)))
    return ohrc_crop, ref_crop, ohrc_win, ref_win


def choose_levels(pair, profile, crops, emit):
    """Step 4."""
    ohrc_crop, ref_crop, ohrc_win, ref_win = crops
    step = Step(emit, 4, "Resolution Scheme")
    step.running(18)

    o_gsd, r_gsd = pair.ohrc_gsd, pair.ref_gsd
    fine = max(o_gsd, r_gsd) * profile.fine_gsd_multiplier
    coarse = fine
    while max(pixels_at(ohrc_crop.shape, o_gsd, coarse), pixels_at(ref_crop.shape, r_gsd, coarse)) > COARSE_PIXEL_BUDGET:
        coarse *= 1.25

    view = level_view_antialiased if profile.antialiased_resampling else level_view
    levels = Levels(
        ohrc_crop=ohrc_crop, ref_crop=ref_crop, ohrc_gsd=o_gsd, ref_gsd=r_gsd,
        fine_gsd=fine, coarse_gsd=coarse,
        ohrc_fine_shape=level_shape(ohrc_crop.shape, o_gsd, fine),
        ref_fine_shape=level_shape(ref_crop.shape, r_gsd, fine),
        ohrc_native_to_fine=mat_scale(o_gsd / fine) @ mat_translate(-ohrc_win[2], -ohrc_win[0]),
        ref_native_to_fine=mat_scale(r_gsd / fine) @ mat_translate(-ref_win[2], -ref_win[0]),
        ohrc_coarse=view(ohrc_crop, o_gsd, coarse),
        ref_coarse=view(ref_crop, r_gsd, coarse),
        level_view=view,
        scale_ratio=max(o_gsd, r_gsd) / min(o_gsd, r_gsd),
    )
    step.done(20, detail=f"Scale ratio {levels.scale_ratio:.1f}× · Fine {fine:.3f} m/px · Coarse {coarse:.3f} m/px · "
                         f"OHRC coarse {levels.ohrc_coarse.shape} · {profile.ref_name} coarse {levels.ref_coarse.shape}",
              image=img_to_b64(side_by_side(levels.ohrc_coarse, levels.ref_coarse)))
    return levels
