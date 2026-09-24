"""OHRC <-> Chandrayaan-2 TMC-2 (v5-tmc). Steps 1-2, ported from sih_final_v5_ohrc_tmc_1.ipynb.

Both products are PDS4, so corners and GSD come from their own labels — nothing is
hardcoded. What TMC needs that NAC does not:
  * dtype from the label (TMC-2 is 16-bit UnsignedLSB2), stretched to 8-bit after cropping;
  * a local sub-strip refit of the corner model, because a TMC strip spans tens of degrees
    of latitude and its trapezoidal footprint cannot be represented by one affine;
  * anti-aliased resampling, a finer target GSD and adaptive tiles (see PROFILE) to cope
    with the ~19x scale ratio.
"""
import os
import numpy as np

from ..common import (unwrap_lon, has_product, resolve_pair_dir, load_pds4, stretch_to_uint8,
                      to_uint8, img_to_b64)
from ..engine import Profile, LoadedPair
from .. import robust, metadata

PROFILE = Profile(key="ohrc_tmc", ref_name="TMC", version="v5",
                  fine_gsd_multiplier=0.5,       # doubles the tile grid across a narrow overlap
                  antialiased_resampling=True,   # decimation destroys structure at ~19x
                  adaptive_tile=True)            # ~4 tile columns across the OHRC overlap
LABEL = "OHRC ↔ TMC-2"
REF_KIND = "tmc"

REFUSE_POLAR = True
TMC_SUBSTRIP_PAD_DEG = 0.05      # ~1.5 km of slack above and below the OHRC band

# TMC sensor from the product id: ncf/ncn/nca = Fore/Nadir/Aft at +25/0/-25 deg.
_SENSOR = {"ncf": ("fore", +25.0), "ncn": ("nadir", 0.0), "nca": ("aft", -25.0)}


def available_pairs(data_raw):
    """Pairs with both an OHRC product (or LINK.json) and a TMC product on disk."""
    d = os.path.join(data_raw, REF_KIND)
    if not os.path.isdir(d): return []
    return [p for p in sorted(os.listdir(d))
            if p.startswith("pair") and has_product(data_raw, REF_KIND, p)
            and has_product(data_raw, "ohrc", p)]


def _tmc_sun(tmc_xml, data_raw, pair_id):
    """TMC sun from its label; a sun.json next to the TMC product replaces it if present."""
    return (robust.read_sun_json(os.path.join(data_raw, REF_KIND, pair_id))
            or robust.read_label_sun(tmc_xml))


def sun_info(pair_id, data_raw):
    """Sun records for the website: both from the PDS4 labels (or sun.json overrides)."""
    _, ohrc_xml = resolve_pair_dir(data_raw, "ohrc", pair_id)
    _, tmc_xml = resolve_pair_dir(data_raw, REF_KIND, pair_id)
    return {"ohrc": robust.read_label_sun(ohrc_xml), "ref": _tmc_sun(tmc_xml, data_raw, pair_id)}


def dataset_info(pair_id, data_raw):
    """Label metadata for the website's Dataset block (no raster is read)."""
    ohrc_img, ohrc_xml = resolve_pair_dir(data_raw, "ohrc", pair_id)
    tmc_img, tmc_xml = resolve_pair_dir(data_raw, REF_KIND, pair_id)
    ref = metadata.pds4_record(tmc_img, tmc_xml)
    _parts = ref["product_id"].split("_")
    sensor, deg = _SENSOR.get(_parts[2][:3].lower() if len(_parts) > 2 else "", ("unknown", 0.0))
    ref["notes"] = {"TMC-2 sensor": f"{sensor} ({deg:+.0f}°)"}
    return {"ohrc": metadata.pds4_record(ohrc_img, ohrc_xml), "ref": ref}


def _edge_at(lat, top, bot):
    """Longitude and row-fraction where an edge of the strip crosses `lat`."""
    f = (top["lat"] - lat) / (top["lat"] - bot["lat"])
    lon_t = float(unwrap_lon(top["lon"], top["lon"]))
    lon_b = float(unwrap_lon(bot["lon"], top["lon"]))
    return lon_t + f * (lon_b - lon_t), f


def _substrip(tmc_corners_full, tmc_shape_full, ohrc_corners):
    """Refit the TMC corner model on just the OHRC's latitude band.

    Returns (corners, row0, row1). Over the whole strip a 6-DOF affine is off by hundreds of
    samples (the footprint tapers as 1/cos(lat)); over the OHRC's ~1 deg band it is not."""
    _c = tmc_corners_full
    _t_lat_hi = max(_c["UL"]["lat"], _c["LL"]["lat"])
    _t_lat_lo = min(_c["UL"]["lat"], _c["LL"]["lat"])
    _o_hi = max(v["lat"] for v in ohrc_corners.values()) + TMC_SUBSTRIP_PAD_DEG
    _o_lo = min(v["lat"] for v in ohrc_corners.values()) - TMC_SUBSTRIP_PAD_DEG
    _band_hi, _band_lo = min(_o_hi, _t_lat_hi), max(_o_lo, _t_lat_lo)

    if _band_hi <= _band_lo:
        raise RuntimeError(
            f"The OHRC latitude band ({_o_lo + TMC_SUBSTRIP_PAD_DEG:.4f}..{_o_hi - TMC_SUBSTRIP_PAD_DEG:.4f} N) "
            f"does not intersect the TMC strip ({_t_lat_lo:.4f}..{_t_lat_hi:.4f} N). "
            f"These two products image different ground.")

    _lonL_hi, _fL_hi = _edge_at(_band_hi, _c["UL"], _c["LL"])
    _lonR_hi, _fR_hi = _edge_at(_band_hi, _c["UR"], _c["LR"])
    _lonL_lo, _fL_lo = _edge_at(_band_lo, _c["UL"], _c["LL"])
    _lonR_lo, _fR_lo = _edge_at(_band_lo, _c["UR"], _c["LR"])
    _r_hi = int(np.floor(0.5 * (_fL_hi + _fR_hi) * tmc_shape_full[0]))
    _r_lo = int(np.ceil(0.5 * (_fL_lo + _fR_lo) * tmc_shape_full[0]))
    row0 = max(0, min(_r_hi, _r_lo))
    row1 = min(tmc_shape_full[0], max(_r_hi, _r_lo))
    if row1 - row0 < 64:
        raise RuntimeError(f"TMC sub-strip is only {row1 - row0} lines -- the OHRC band "
                           f"barely touches this strip's end.")

    if (row1 - row0) / tmc_shape_full[0] > 0.60:
        # The OHRC band already covers most of the strip, so the local fit is the global fit.
        return tmc_corners_full, 0, tmc_shape_full[0]

    corners = {"UL": {"lat": _band_hi, "lon": _lonL_hi % 360.0},
               "UR": {"lat": _band_hi, "lon": _lonR_hi % 360.0},
               "LL": {"lat": _band_lo, "lon": _lonL_lo % 360.0},
               "LR": {"lat": _band_lo, "lon": _lonR_lo % 360.0}}
    return corners, row0, row1


def load(pair_id, config, data_raw, emit_event):
    # ========== STEP 1: Load OHRC ==========
    emit_event(1, "Loading OHRC Image", "running", 2)

    ohrc_img, ohrc_xml = resolve_pair_dir(data_raw, "ohrc", pair_id)
    ohrc_product_id = os.path.splitext(os.path.basename(ohrc_img))[0]
    ohrc_raster, ohrc_corners, ohrc_shape, OHRC_GSD_FROM_LABEL, ohrc_meta = load_pds4(
        ohrc_img, ohrc_xml, "OHRC", allow_truncated=False, refuse_polar=REFUSE_POLAR)
    OHRC_NATIVE_GSD_M = OHRC_GSD_FROM_LABEL or 0.26

    emit_event(1, "Loading OHRC Image", "done", 5,
               detail=f"OHRC {ohrc_shape[0]:,} × {ohrc_shape[1]:,} px @ {OHRC_NATIVE_GSD_M:.2f} m/px · "
                      f"sun elev {ohrc_meta['sun_elev']}° · {ohrc_product_id}",
               image=img_to_b64(to_uint8(stretch_to_uint8(ohrc_raster[::16, ::16]))))

    # ========== STEP 2: Load TMC ==========
    emit_event(2, "Loading TMC Reference", "running", 7)

    tmc_img, tmc_xml = resolve_pair_dir(data_raw, REF_KIND, pair_id)
    tmc_product_id = os.path.splitext(os.path.basename(tmc_img))[0]
    _parts = tmc_product_id.split("_")
    TMC_SENSOR, TMC_SENSOR_DEG = _SENSOR.get(_parts[2][:3].lower() if len(_parts) > 2 else "",
                                             ("unknown", 0.0))

    tmc_raster_full, tmc_corners_full, tmc_shape_full, TMC_NATIVE_GSD_M, tmc_meta = load_pds4(
        tmc_img, tmc_xml, "TMC", allow_truncated=False, refuse_polar=REFUSE_POLAR)
    if TMC_NATIVE_GSD_M is None:
        raise RuntimeError("No GSD in the TMC label (isda:pixel_resolution missing).")

    tmc_corners, row0, row1 = _substrip(tmc_corners_full, tmc_shape_full, ohrc_corners)
    tmc_raster = tmc_raster_full[row0:row1]
    tmc_shape = (row1 - row0, tmc_shape_full[1])

    _preview = stretch_to_uint8(tmc_raster[::16, ::16])
    emit_event(2, "Loading TMC Reference", "done", 10,
               detail=f"TMC {tmc_shape_full[0]:,} × {tmc_shape_full[1]:,} px ({tmc_meta['dtype_pds4']}) @ "
                      f"{TMC_NATIVE_GSD_M:.2f} m/px · {TMC_SENSOR} sensor · sub-strip rows "
                      f"{row0:,}–{row1:,} · scale ratio {TMC_NATIVE_GSD_M / OHRC_NATIVE_GSD_M:.1f}× · {tmc_product_id}",
               image=img_to_b64(_preview))

    def _gap(a, b, wrap=False):
        if a is None or b is None: return None
        return abs((a - b + 180.0) % 360.0 - 180.0) if wrap else abs(a - b)

    return LoadedPair(
        ohrc_raster=ohrc_raster, ohrc_corners=ohrc_corners, ohrc_shape=ohrc_shape,
        ohrc_gsd=OHRC_NATIVE_GSD_M, ohrc_product_id=ohrc_product_id,
        ref_raster=tmc_raster, ref_corners=tmc_corners, ref_shape=tmc_shape,
        ref_gsd=TMC_NATIVE_GSD_M, ref_product_id=tmc_product_id,
        ohrc_sun=robust.read_label_sun(ohrc_xml),
        ref_sun=robust.apply_sun_override(_tmc_sun(tmc_xml, data_raw, pair_id), config.get('ref_sun')),
        extra_metrics={
            "corners_source": "PDS4 labels (both products)",
            "tmc_sensor": TMC_SENSOR, "tmc_sensor_deg": TMC_SENSOR_DEG,
            "tmc_substrip_rows": [int(row0), int(row1)],
            "sun_elev_ohrc": ohrc_meta["sun_elev"], "sun_elev_tmc": tmc_meta["sun_elev"],
            "sun_azim_ohrc": ohrc_meta["sun_azim"], "sun_azim_tmc": tmc_meta["sun_azim"],
            "sun_azim_gap_deg": _gap(ohrc_meta["sun_azim"], tmc_meta["sun_azim"], wrap=True),
        },
    )
