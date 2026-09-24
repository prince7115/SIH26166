"""OHRC <-> Chandrayaan-2 TMC-2 (v5): both products PDS4, corners and GSD from their labels.

TMC-2 differs from NAC in three ways handled here and in PROFILE:
  * 16-bit samples, stretched to 8 bit after cropping;
  * a strip tens of degrees long, whose footprint no single affine fits, so the corner model is
    refit on the OHRC's latitude band (_substrip);
  * a ~19x scale gap: anti-aliased resampling, a finer working GSD and adaptive tiles.
"""
import os

import numpy as np

from .ohrc import preview, read_ohrc
from ..geometry.geodesy import unwrap_lon
from ..imaging.radiometry import stretch_to_uint8
from ..products import metadata
from ..products.discovery import has_product, pair_dirs, product_id, resolve_pair_dir
from ..products.pds4 import load_pds4
from ..products.sun import apply_sun_override, azimuth_gap, label_sun, read_label_sun, read_sun_json
from ..registration.models import LoadedPair, Profile
from ..registration.steps import Step
from ..reporting.encoding import img_to_b64

PROFILE = Profile(key="ohrc_tmc", ref_name="TMC", version="v5",
                  fine_gsd_multiplier=0.5,       # doubles the tile grid across a narrow overlap
                  antialiased_resampling=True,   # decimation destroys structure at ~19x
                  adaptive_tile=True)            # ~4 tile columns across the OHRC overlap
LABEL = "OHRC ↔ TMC-2"
REF_KIND = "tmc"

SUBSTRIP_PAD_DEG = 0.05      # ~1.5 km of slack above and below the OHRC band

# Sensor from the product id: ncf / ncn / nca = fore / nadir / aft, looking +25 / 0 / -25 degrees.
_SENSORS = {"ncf": ("fore", +25.0), "ncn": ("nadir", 0.0), "nca": ("aft", -25.0)}


def _sensor(tmc_id):
    parts = tmc_id.split("_")
    return _SENSORS.get(parts[2][:3].lower() if len(parts) > 2 else "", ("unknown", 0.0))


def _tmc_dir(data_raw, pair_id):
    return os.path.join(data_raw, REF_KIND, pair_id)


def available_pairs(data_raw):
    return [p for p in pair_dirs(data_raw, REF_KIND)
            if has_product(data_raw, REF_KIND, p) and has_product(data_raw, "ohrc", p)]


def sun_info(pair_id, data_raw):
    """Both from the PDS4 labels; a sun.json next to the TMC product replaces its label values."""
    _, ohrc_xml = resolve_pair_dir(data_raw, "ohrc", pair_id)
    _, tmc_xml = resolve_pair_dir(data_raw, REF_KIND, pair_id)
    return {"ohrc": read_label_sun(ohrc_xml),
            "ref": read_sun_json(_tmc_dir(data_raw, pair_id)) or read_label_sun(tmc_xml)}


def dataset_info(pair_id, data_raw):
    ohrc_img, ohrc_xml = resolve_pair_dir(data_raw, "ohrc", pair_id)
    tmc_img, tmc_xml = resolve_pair_dir(data_raw, REF_KIND, pair_id)
    ref = metadata.pds4_record(tmc_img, tmc_xml)
    sensor, deg = _sensor(ref["product_id"])
    ref["notes"] = {"TMC-2 sensor": f"{sensor} ({deg:+.0f}°)"}
    return {"ohrc": metadata.pds4_record(ohrc_img, ohrc_xml), "ref": ref}


def _edge_at(lat, top, bot):
    """Longitude and row fraction where one edge of the strip crosses lat."""
    f = (top["lat"] - lat) / (top["lat"] - bot["lat"])
    lon_t = float(unwrap_lon(top["lon"], top["lon"]))
    lon_b = float(unwrap_lon(bot["lon"], top["lon"]))
    return lon_t + f * (lon_b - lon_t), f


def _substrip(tmc_corners, tmc_shape, ohrc_corners):
    """Corner model refit on the OHRC's latitude band. Returns (corners, row0, row1).

    Over a whole strip a 6-parameter affine is off by hundreds of samples (the footprint tapers
    as 1/cos(lat)); over the OHRC's ~1 degree band it is not.
    """
    c = tmc_corners
    t_lat_hi = max(c["UL"]["lat"], c["LL"]["lat"])
    t_lat_lo = min(c["UL"]["lat"], c["LL"]["lat"])
    o_hi = max(v["lat"] for v in ohrc_corners.values()) + SUBSTRIP_PAD_DEG
    o_lo = min(v["lat"] for v in ohrc_corners.values()) - SUBSTRIP_PAD_DEG
    band_hi, band_lo = min(o_hi, t_lat_hi), max(o_lo, t_lat_lo)
    if band_hi <= band_lo:
        raise RuntimeError(
            f"The OHRC latitude band ({o_lo + SUBSTRIP_PAD_DEG:.4f}..{o_hi - SUBSTRIP_PAD_DEG:.4f} N) "
            f"does not intersect the TMC strip ({t_lat_lo:.4f}..{t_lat_hi:.4f} N). "
            f"These two products image different ground.")

    lon_l_hi, f_l_hi = _edge_at(band_hi, c["UL"], c["LL"])
    lon_r_hi, f_r_hi = _edge_at(band_hi, c["UR"], c["LR"])
    lon_l_lo, f_l_lo = _edge_at(band_lo, c["UL"], c["LL"])
    lon_r_lo, f_r_lo = _edge_at(band_lo, c["UR"], c["LR"])
    r_hi = int(np.floor(0.5 * (f_l_hi + f_r_hi) * tmc_shape[0]))
    r_lo = int(np.ceil(0.5 * (f_l_lo + f_r_lo) * tmc_shape[0]))
    row0 = max(0, min(r_hi, r_lo))
    row1 = min(tmc_shape[0], max(r_hi, r_lo))
    if row1 - row0 < 64:
        raise RuntimeError(f"TMC sub-strip is only {row1 - row0} lines -- the OHRC band "
                           f"barely touches this strip's end.")
    if (row1 - row0) / tmc_shape[0] > 0.60:
        # The band covers most of the strip already: the local fit is the global fit.
        return tmc_corners, 0, tmc_shape[0]

    corners = {"UL": {"lat": band_hi, "lon": lon_l_hi % 360.0},
               "UR": {"lat": band_hi, "lon": lon_r_hi % 360.0},
               "LL": {"lat": band_lo, "lon": lon_l_lo % 360.0},
               "LR": {"lat": band_lo, "lon": lon_r_lo % 360.0}}
    return corners, row0, row1


def load(pair_id, data_raw, emit, ref_sun_override=None):
    step = Step(emit, 1, "Loading OHRC Image")
    step.running(2)
    ohrc = read_ohrc(pair_id, data_raw)
    step.done(5, detail=f"{ohrc.summary()} · sun elev {ohrc.meta['sun_elev']}° · {ohrc.product_id}",
              image=preview(ohrc.raster))

    step = Step(emit, 2, "Loading TMC Reference")
    step.running(7)
    tmc_img, tmc_xml = resolve_pair_dir(data_raw, REF_KIND, pair_id)
    tmc_id = product_id(tmc_img)
    sensor, sensor_deg = _sensor(tmc_id)
    full, full_corners, full_shape, gsd, meta = load_pds4(tmc_img, tmc_xml, "TMC")
    if gsd is None:
        raise RuntimeError("No GSD in the TMC label (isda:pixel_resolution missing).")
    corners, row0, row1 = _substrip(full_corners, full_shape, ohrc.corners)
    raster = full[row0:row1]
    step.done(10, detail=f"TMC {full_shape[0]:,} × {full_shape[1]:,} px ({meta['dtype_pds4']}) @ "
                         f"{gsd:.2f} m/px · {sensor} sensor · sub-strip rows {row0:,}–{row1:,} · "
                         f"scale ratio {gsd / ohrc.gsd:.1f}× · {tmc_id}",
              image=img_to_b64(stretch_to_uint8(raster[::16, ::16])))

    tmc_sun = read_sun_json(_tmc_dir(data_raw, pair_id)) or label_sun(meta["sun_elev"], meta["sun_azim"])
    azim_gap = (None if ohrc.meta["sun_azim"] is None or meta["sun_azim"] is None
                else azimuth_gap(ohrc.meta["sun_azim"], meta["sun_azim"]))
    return LoadedPair(
        ohrc_raster=ohrc.raster, ohrc_corners=ohrc.corners, ohrc_shape=ohrc.shape,
        ohrc_gsd=ohrc.gsd, ohrc_product_id=ohrc.product_id,
        ref_raster=raster, ref_corners=corners, ref_shape=(row1 - row0, full_shape[1]),
        ref_gsd=gsd, ref_product_id=tmc_id,
        ohrc_sun=ohrc.sun,
        ref_sun=apply_sun_override(tmc_sun, ref_sun_override),
        extra_metrics={
            "corners_source": "PDS4 labels (both products)",
            "tmc_sensor": sensor, "tmc_sensor_deg": sensor_deg,
            "tmc_substrip_rows": [int(row0), int(row1)],
            "sun_elev_ohrc": ohrc.meta["sun_elev"], "sun_elev_tmc": meta["sun_elev"],
            "sun_azim_ohrc": ohrc.meta["sun_azim"], "sun_azim_tmc": meta["sun_azim"],
            "sun_azim_gap_deg": azim_gap,
        },
    )
