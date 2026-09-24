"""OHRC <-> LRO NAC (v4). Steps 1-2: PDS4 OHRC + PDS3 NAC with hand-maintained footprints.

LROC NAC EDR labels carry no footprint, so corners and GSD come from KNOWN_NAC_CORNERS /
KNOWN_NAC_GSD. Add a row there to register a new NAC product.
"""
import os
import numpy as np
import xml.etree.ElementTree as ET

# ---- Numpy compatibility shims (for planetaryimage) ----
for _alias, _real in (("product", np.prod), ("cumproduct", np.cumprod),
                      ("alltrue", np.all), ("sometrue", np.any)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _real)

# Patch numpy.fromstring → frombuffer (removed in NumPy 2.x, needed by planetaryimage)
_orig_fromstring = np.fromstring
def _patched_fromstring(string, dtype=float, count=-1, *, sep='', like=None):
    if isinstance(string, bytes) and sep == '':
        return np.frombuffer(string, dtype=dtype, count=count)
    return _orig_fromstring(string, dtype=dtype, count=count, sep=sep)
np.fromstring = _patched_fromstring

from planetaryimage import PDS3Image

from ..common import NS, find_first, has_product, resolve_pair_dir, to_uint8, img_to_b64
from ..engine import Profile, LoadedPair
from .. import robust, metadata

PROFILE = Profile(key="ohrc_nac", ref_name="NAC", version="v4")
LABEL = "OHRC ↔ LRO NAC"
REF_KIND = "nac"

KNOWN_NAC_CORNERS = {
    "M177656091LE": {"UL": (61.61, 355.45), "UR": (61.60, 355.29),
                     "LL": (59.92, 355.53), "LR": (59.92, 355.38)},
    "M175124932RE": {"UL": ( 1.11,  23.49), "UR": ( 1.11,  23.45),
                     "LL": ( 0.15,  23.50), "LR": ( 0.15,  23.46)},
}
KNOWN_NAC_GSD = {
    "M177656091LE": 0.933909314468014,
    "M175124932RE": 0.39935843279144334,
}


def _corners_from_table(entry):
    if isinstance(entry, dict) and set(entry) == {"UL", "UR", "LL", "LR"}:
        return {k: {"lat": float(v[0]), "lon": float(v[1])} for k, v in entry.items()}
    raise ValueError("KNOWN_NAC_CORNERS entries must be a dict of four corners.")


def load_ohrc(img_path, xml_path, allow_truncated=False):
    root = ET.parse(xml_path).getroot()
    axes = root.findall(".//pds:Array_2D_Image/pds:Axis_Array", NS)
    dims = {a.find("pds:axis_name", NS).text: int(a.find("pds:elements", NS).text) for a in axes}
    n_lines, n_samples = dims["Line"], dims["Sample"]
    geom = root.find(".//isda:Refined_Corner_Coordinates", NS)
    if geom is None: raise ValueError("No isda:Refined_Corner_Coordinates in OHRC label.")
    corners_deg = {k: {"lat": float(geom.find(f"isda:{t}_latitude", NS).text),
                       "lon": float(geom.find(f"isda:{t}_longitude", NS).text)}
                   for k, t in (("UL", "upper_left"), ("UR", "upper_right"),
                                ("LL", "lower_left"), ("LR", "lower_right"))}
    gsd = None
    for el in root.iter():
        tag = el.tag.split('}')[-1].lower()
        if ('resolution' in tag or 'pixel_size' in tag or 'sampling' in tag) and el.text:
            try:
                v = float(el.text.strip().split()[0])
                if gsd is None and 0.05 < v < 5.0: gsd = v
            except ValueError: continue
    expected, actual = n_lines * n_samples, os.path.getsize(img_path)
    if actual < expected:
        avail = actual // n_samples
        if not allow_truncated:
            raise ValueError(f"OHRC file truncated: {avail}/{n_lines} lines")
        frac = avail / float(n_lines)
        for top, bot in (("UL", "LL"), ("UR", "LR")):
            for k in ("lat", "lon"):
                corners_deg[bot][k] = corners_deg[top][k] + frac * (corners_deg[bot][k] - corners_deg[top][k])
        n_lines = avail
    raster = np.memmap(img_path, dtype=np.uint8, mode="r", shape=(n_lines, n_samples))
    return raster, corners_deg, (n_lines, n_samples), gsd


def available_pairs(data_raw):
    """Pairs with both an OHRC product and a NAC .IMG on disk."""
    d = os.path.join(data_raw, REF_KIND)
    if not os.path.isdir(d): return []
    return [p for p in sorted(os.listdir(d))
            if p.startswith("pair") and find_first(os.path.join(d, p), [".img"], recursive=False)
            and has_product(data_raw, "ohrc", p)]


def sun_info(pair_id, data_raw):
    """Sun records for the website: OHRC from its label, NAC from sun.json (the NAC EDR
    label carries no sun geometry; copy it from the LROC product page)."""
    _, ohrc_xml = resolve_pair_dir(data_raw, "ohrc", pair_id)
    return {"ohrc": robust.read_label_sun(ohrc_xml),
            "ref": robust.read_sun_json(os.path.join(data_raw, REF_KIND, pair_id))}


def dataset_info(pair_id, data_raw):
    """Label metadata for the website's Dataset block (no raster is read)."""
    ohrc_img, ohrc_xml = resolve_pair_dir(data_raw, "ohrc", pair_id)
    nac_img = find_first(os.path.join(data_raw, REF_KIND, pair_id), [".img"], recursive=False)
    if nac_img is None: raise FileNotFoundError(f"No NAC .IMG under data/raw/{REF_KIND}/{pair_id}")
    nac_id = os.path.splitext(os.path.basename(nac_img))[0]
    corners = _corners_from_table(KNOWN_NAC_CORNERS[nac_id]) if nac_id in KNOWN_NAC_CORNERS else None
    sun = robust.read_sun_json(os.path.join(data_raw, REF_KIND, pair_id))
    return {"ohrc": metadata.pds4_record(ohrc_img, ohrc_xml),
            "ref": metadata.lroc_nac_record(nac_img, corners, KNOWN_NAC_GSD.get(nac_id), sun)}


def load(pair_id, config, data_raw, emit_event):
    # ========== STEP 1: Load OHRC ==========
    emit_event(1, "Loading OHRC Image", "running", 2)

    ohrc_img, ohrc_xml = resolve_pair_dir(data_raw, "ohrc", pair_id)
    ohrc_product_id = os.path.splitext(os.path.basename(ohrc_img))[0]

    ohrc_raster, ohrc_corners, ohrc_shape, OHRC_GSD_FROM_LABEL = load_ohrc(ohrc_img, ohrc_xml, False)
    OHRC_NATIVE_GSD_M = OHRC_GSD_FROM_LABEL or 0.26

    emit_event(1, "Loading OHRC Image", "done", 5,
               detail=f"OHRC {ohrc_shape[0]:,} × {ohrc_shape[1]:,} px @ {OHRC_NATIVE_GSD_M:.2f} m/px · {ohrc_product_id}",
               image=img_to_b64(to_uint8(np.asarray(ohrc_raster[::16, ::16]))))

    # ========== STEP 2: Load NAC ==========
    emit_event(2, "Loading NAC Reference", "running", 7)

    nac_dir = os.path.join(data_raw, REF_KIND, pair_id)
    nac_img = find_first(nac_dir, [".img"], recursive=False)
    if nac_img is None: raise FileNotFoundError(f"No NAC .IMG under {nac_dir}")
    nac_product_id = os.path.splitext(os.path.basename(nac_img))[0]

    nac_pds = PDS3Image.open(nac_img)
    nac_full = np.asarray(nac_pds.image)
    if nac_full.ndim == 3: nac_full = nac_full[0]
    if nac_full.dtype == np.int8: nac_full = nac_full.view(np.uint8)
    nac_shape = nac_full.shape

    _entry = KNOWN_NAC_CORNERS.get(nac_product_id)
    if _entry is None: raise RuntimeError(f"No corners for NAC '{nac_product_id}'. Add to KNOWN_NAC_CORNERS.")
    nac_corners = _corners_from_table(_entry)

    NAC_NATIVE_GSD_M = KNOWN_NAC_GSD.get(nac_product_id)
    if NAC_NATIVE_GSD_M is None: raise RuntimeError(f"No GSD for NAC '{nac_product_id}'.")

    emit_event(2, "Loading NAC Reference", "done", 10,
               detail=f"NAC {nac_shape[0]:,} × {nac_shape[1]:,} px @ {NAC_NATIVE_GSD_M:.4f} m/px · {nac_product_id}",
               image=img_to_b64(to_uint8(np.asarray(nac_full[::16, ::16]))))

    return LoadedPair(
        ohrc_raster=ohrc_raster, ohrc_corners=ohrc_corners, ohrc_shape=ohrc_shape,
        ohrc_gsd=OHRC_NATIVE_GSD_M, ohrc_product_id=ohrc_product_id,
        ref_raster=nac_full, ref_corners=nac_corners, ref_shape=nac_shape,
        ref_gsd=NAC_NATIVE_GSD_M, ref_product_id=nac_product_id,
        ohrc_sun=robust.read_label_sun(ohrc_xml),
        ref_sun=robust.apply_sun_override(robust.read_sun_json(nac_dir), config.get('ref_sun')),
    )
