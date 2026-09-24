"""Sun geometry records: from PDS4 labels, a sun.json next to the product, or the website."""
import json
import os

from .pds4 import Pds4Label

AZIM_CONVENTIONS = ("north_cw", "image_cw", "unknown")


def make_sun(elev=None, incidence=None, azim=None, conv=None, source=None):
    """Normalised sun record, or None if nothing is known.

    Elevation wins over incidence (elevation = 90 - incidence); the azimuth convention is
    recorded only when an azimuth is given.
    """
    def _num(v):
        try:
            return None if v is None or v == "" else float(v)
        except (TypeError, ValueError):
            return None
    elev, incidence, azim = _num(elev), _num(incidence), _num(azim)
    if elev is None and incidence is not None:
        elev = 90.0 - incidence
    if azim is not None:
        azim = azim % 360.0
        conv = conv if conv in AZIM_CONVENTIONS else "unknown"
    else:
        conv = None
    if elev is None and azim is None:
        return None
    return {"elev_deg": elev, "azim_deg": azim, "azim_convention": conv, "source": source}


def label_sun(elev, azim):
    """Sun record from ISRO PDS4 label values; their azimuth is clockwise from north."""
    return make_sun(elev=elev, azim=azim, conv="north_cw", source="label")


def read_label_sun(xml_path):
    label = Pds4Label(xml_path)
    return label_sun(label.number(".//isda:sun_elevation"), label.number(".//isda:sun_azimuth"))


def read_sun_json(folder):
    """sun.json in a product folder, with values copied from the product's web page:
    {"incidence_deg" or "elevation_deg", "azimuth_deg", "azimuth_convention"}."""
    path = os.path.join(folder, "sun.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return make_sun(elev=d.get("elevation_deg"), incidence=d.get("incidence_deg"),
                    azim=d.get("azimuth_deg"), conv=d.get("azimuth_convention"), source="sun.json")


def apply_sun_override(base, override):
    """Values typed on the website replace the file/label values field by field."""
    if not override:
        return base
    merged = dict(base or {"elev_deg": None, "azim_deg": None, "azim_convention": None, "source": None})
    changed = False
    for k in ("elev_deg", "azim_deg", "azim_convention"):
        if override.get(k) not in (None, ""):
            merged[k] = override[k]
            changed = True
    if not changed:
        return base
    return make_sun(elev=merged["elev_deg"], azim=merged["azim_deg"],
                    conv=merged["azim_convention"], source="website")


def azimuth_gap(a, b):
    """Smallest angle between two azimuths, in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)
