"""OHRC <-> LRO NAC (v4): PDS4 OHRC and PDS3 NAC EDR.

NAC EDR labels carry no footprint, so corners and GSD come from NAC_FOOTPRINTS; add a row
there to register a new NAC product.
"""
import os

from .ohrc import preview, read_ohrc
from ..products import metadata
from ..products.discovery import find_first, has_product, pair_dirs, product_id, resolve_pair_dir
from ..products.pds3 import read_image
from ..products.sun import apply_sun_override, read_label_sun, read_sun_json
from ..registration.models import LoadedPair, Profile
from ..registration.steps import Step

PROFILE = Profile(key="ohrc_nac", ref_name="NAC", version="v4")
LABEL = "OHRC ↔ LRO NAC"
REF_KIND = "nac"

# Corner (lat, lon) in degrees and native GSD in metres, per NAC product id.
NAC_FOOTPRINTS = {
    "M177656091LE": {"corners": {"UL": (61.61, 355.45), "UR": (61.60, 355.29),
                                 "LL": (59.92, 355.53), "LR": (59.92, 355.38)},
                     "gsd_m": 0.933909314468014},
    "M175124932RE": {"corners": {"UL": (1.11, 23.49), "UR": (1.11, 23.45),
                                 "LL": (0.15, 23.50), "LR": (0.15, 23.46)},
                     "gsd_m": 0.39935843279144334},
}


def _corners(nac_id):
    entry = NAC_FOOTPRINTS.get(nac_id)
    if entry is None:
        return None
    return {k: {"lat": float(lat), "lon": float(lon)} for k, (lat, lon) in entry["corners"].items()}


def _nac_dir(data_raw, pair_id):
    return os.path.join(data_raw, REF_KIND, pair_id)


def _find_nac_image(data_raw, pair_id):
    return find_first(_nac_dir(data_raw, pair_id), [".img"], recursive=False)


def _nac_image(data_raw, pair_id):
    img = _find_nac_image(data_raw, pair_id)
    if img is None:
        raise FileNotFoundError(f"No NAC .IMG under data/raw/{REF_KIND}/{pair_id}")
    return img


def available_pairs(data_raw):
    return [p for p in pair_dirs(data_raw, REF_KIND)
            if _find_nac_image(data_raw, p) and has_product(data_raw, "ohrc", p)]


def sun_info(pair_id, data_raw):
    """OHRC sun from its label; NAC sun from sun.json (copied from the LROC product page)."""
    _, ohrc_xml = resolve_pair_dir(data_raw, "ohrc", pair_id)
    return {"ohrc": read_label_sun(ohrc_xml), "ref": read_sun_json(_nac_dir(data_raw, pair_id))}


def dataset_info(pair_id, data_raw):
    ohrc_img, ohrc_xml = resolve_pair_dir(data_raw, "ohrc", pair_id)
    nac_img = _nac_image(data_raw, pair_id)
    nac_id = product_id(nac_img)
    gsd = NAC_FOOTPRINTS.get(nac_id, {}).get("gsd_m")
    return {"ohrc": metadata.pds4_record(ohrc_img, ohrc_xml),
            "ref": metadata.lroc_nac_record(nac_img, _corners(nac_id), gsd,
                                            read_sun_json(_nac_dir(data_raw, pair_id)))}


def load(pair_id, data_raw, emit, ref_sun_override=None):
    step = Step(emit, 1, "Loading OHRC Image")
    step.running(2)
    ohrc = read_ohrc(pair_id, data_raw)
    step.done(5, detail=f"{ohrc.summary()} · {ohrc.product_id}", image=preview(ohrc.raster))

    step = Step(emit, 2, "Loading NAC Reference")
    step.running(7)
    nac_img = _nac_image(data_raw, pair_id)
    nac_id = product_id(nac_img)
    nac = read_image(nac_img)
    corners = _corners(nac_id)
    if corners is None:
        raise RuntimeError(f"No corners for NAC '{nac_id}'. Add it to NAC_FOOTPRINTS.")
    gsd = NAC_FOOTPRINTS[nac_id]["gsd_m"]
    step.done(10, detail=f"NAC {nac.shape[0]:,} × {nac.shape[1]:,} px @ {gsd:.4f} m/px · {nac_id}",
              image=preview(nac))

    return LoadedPair(
        ohrc_raster=ohrc.raster, ohrc_corners=ohrc.corners, ohrc_shape=ohrc.shape,
        ohrc_gsd=ohrc.gsd, ohrc_product_id=ohrc.product_id,
        ref_raster=nac, ref_corners=corners, ref_shape=nac.shape,
        ref_gsd=gsd, ref_product_id=nac_id,
        ohrc_sun=ohrc.sun,
        ref_sun=apply_sun_override(read_sun_json(_nac_dir(data_raw, pair_id)), ref_sun_override),
    )
