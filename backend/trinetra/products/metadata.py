"""Label metadata for the website's Dataset panel.

Only labels are read (PDS4 XML, or the PDS3 header of a NAC .IMG), never rasters, so this
answers in milliseconds even for multi-GB strips. Every product becomes the same record shape.
"""
import os
from datetime import datetime

from .discovery import product_id
from .pds3 import read_header
from .pds4 import Pds4Label, parse_number
from .sun import azimuth_gap
from ..geometry.geodesy import KM_PER_DEG_LAT, km_per_deg_lon, unwrap_lon


def _int(v):
    f = parse_number(v)
    return None if f is None else int(f)


def _file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def footprint(corners):
    """Centre, along/across-track extent (km) and lat/lon bounds of a four-corner footprint."""
    if not corners:
        return None
    ref = corners["UL"]["lon"]
    lats = [c["lat"] for c in corners.values()]
    lons = [float(unwrap_lon(c["lon"], ref)) for c in corners.values()]
    mid_lat = sum(lats) / 4.0

    def _dist(a, b):
        dlat = (corners[a]["lat"] - corners[b]["lat"]) * KM_PER_DEG_LAT
        dlon = float(unwrap_lon(corners[a]["lon"], ref) - unwrap_lon(corners[b]["lon"], ref)) * km_per_deg_lon(mid_lat)
        return (dlat ** 2 + dlon ** 2) ** 0.5

    return {
        "center_lat": mid_lat, "center_lon": (sum(lons) / 4.0) % 360.0,
        "length_km": 0.5 * (_dist("UL", "LL") + _dist("UR", "LR")),
        "width_km": 0.5 * (_dist("UL", "UR") + _dist("LL", "LR")),
        "lat_min": min(lats), "lat_max": max(lats), "lon_min": min(lons), "lon_max": max(lons),
    }


def pds4_record(img_path, xml_path):
    """An ISRO PDS4 product (OHRC, TMC-2)."""
    label = Pds4Label(xml_path)
    dims = label.dims()
    corners, source = label.corners()
    instrument = label.instrument()
    return {
        "product_id": product_id(img_path),
        "format": "PDS4",
        "mission": "Chandrayaan-2",
        "instrument": instrument.title() if instrument else label.text(".//pds:title"),
        "start_time": label.text(".//pds:start_date_time"),
        "stop_time": label.text(".//pds:stop_date_time"),
        "orbit": _int(label.text(".//isda:imaging_orbit_number")),
        "processing_level": label.text(".//pds:processing_level"),
        "lines": dims.get("Line"), "samples": dims.get("Sample"),
        "data_type": label.text(".//pds:Element_Array/pds:data_type"),
        "gsd_m": label.number(".//isda:pixel_resolution"),
        "altitude_km": label.number(".//isda:spacecraft_altitude"),
        "sun_elev_deg": label.number(".//isda:sun_elevation"),
        "sun_azim_deg": label.number(".//isda:sun_azimuth"),
        "sun_azim_convention": "north_cw",
        "incidence_deg": label.number(".//isda:solar_incidence"),
        "roll_deg": label.number(".//isda:roll"),
        "pitch_deg": label.number(".//isda:pitch"),
        "orbit_direction": label.text(".//isda:orbit_limb_direction"),
        "projection": label.text(".//isda:projection"),
        "area": label.text(".//isda:area"),
        "corners": corners, "corners_source": f"label ({source})" if source else None,
        "footprint": footprint(corners),
        "file_size_bytes": _int(label.text(".//pds:File/pds:file_size")) or _file_size(img_path),
    }


def lroc_nac_record(img_path, corners, gsd_m, sun=None):
    """An LROC NAC EDR. Its label has no footprint or sun geometry, so corners and GSD come from
    the pipeline's lookup table and the sun from sun.json when present."""
    h = read_header(img_path)
    frame = h.get("FRAME_ID")
    return {
        "product_id": h.get("PRODUCT_ID") or product_id(img_path),
        "format": "PDS3",
        "mission": "Lunar Reconnaissance Orbiter",
        "instrument": f"LROC Narrow Angle Camera ({frame.title()})" if frame else "LROC Narrow Angle Camera",
        "start_time": h.get("START_TIME"),
        "stop_time": h.get("STOP_TIME"),
        "orbit": _int(h.get("ORBIT_NUMBER")),
        "processing_level": h.get("PRODUCT_TYPE"),
        "lines": _int(h.get("LINES")), "samples": _int(h.get("LINE_SAMPLES")),
        "data_type": f"{h.get('SAMPLE_BITS', '?')}-bit {h.get('SAMPLE_TYPE', '')}".strip(),
        "gsd_m": gsd_m,
        "altitude_km": None,
        "sun_elev_deg": (sun or {}).get("elev_deg"),
        "sun_azim_deg": (sun or {}).get("azim_deg"),
        "sun_azim_convention": (sun or {}).get("azim_convention"),
        "incidence_deg": 90.0 - sun["elev_deg"] if sun and sun.get("elev_deg") is not None else None,
        "roll_deg": None, "pitch_deg": None,
        "orbit_direction": None,
        "projection": None, "area": None,
        "corners": corners, "corners_source": "pipeline lookup table (NAC EDR labels carry none)",
        "footprint": footprint(corners),
        "file_size_bytes": _file_size(img_path),
        "notes": {k: v for k, v in (("Data set", h.get("DATA_SET_ID")),
                                    ("Rationale", h.get("RATIONALE_DESC")),
                                    ("Line exposure", h.get("LINE_EXPOSURE_DURATION")),
                                    ("Cross-track summing", h.get("CROSSTRACK_SUMMING")),
                                    ("Sun geometry", "sun.json" if sun else "not available (add sun.json)"))
                  if v},
    }


def _parse_time(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.rstrip("Z")[:26])
    except ValueError:
        return None


def compare(ohrc, ref):
    """Pair-level numbers: overlap, time between acquisitions, scale and illumination gaps."""
    out = {}
    fo, fr = ohrc.get("footprint"), ref.get("footprint")
    if fo and fr:
        lon0_ref = ohrc["corners"]["UL"]["lon"]
        o_lon = [float(unwrap_lon(c["lon"], lon0_ref)) for c in ohrc["corners"].values()]
        r_lon = [float(unwrap_lon(c["lon"], lon0_ref)) for c in ref["corners"].values()]
        lat0, lat1 = max(fo["lat_min"], fr["lat_min"]), min(fo["lat_max"], fr["lat_max"])
        lon0, lon1 = max(min(o_lon), min(r_lon)), min(max(o_lon), max(r_lon))
        if lat1 > lat0 and lon1 > lon0:
            mid = 0.5 * (lat0 + lat1)
            h_km, w_km = (lat1 - lat0) * KM_PER_DEG_LAT, (lon1 - lon0) * km_per_deg_lon(mid)
            o_area = ((fo["lat_max"] - fo["lat_min"]) * KM_PER_DEG_LAT *
                      (max(o_lon) - min(o_lon)) * km_per_deg_lon(fo["center_lat"]))
            out["overlap_km"] = [h_km, w_km]
            out["overlap_pct_of_ohrc"] = 100.0 * h_km * w_km / o_area if o_area > 0 else None
        else:
            out["overlap_km"] = None
    t_o, t_r = _parse_time(ohrc.get("start_time")), _parse_time(ref.get("start_time"))
    if t_o and t_r:
        out["days_apart"] = abs((t_o - t_r).total_seconds()) / 86400.0
    if ohrc.get("gsd_m") and ref.get("gsd_m"):
        out["scale_ratio"] = ref["gsd_m"] / ohrc["gsd_m"]
    # Azimuths are only comparable when both are measured clockwise from north.
    if (ohrc.get("sun_azim_deg") is not None and ref.get("sun_azim_deg") is not None
            and ohrc.get("sun_azim_convention") == ref.get("sun_azim_convention") == "north_cw"):
        out["sun_azim_gap_deg"] = azimuth_gap(ohrc["sun_azim_deg"], ref["sun_azim_deg"])
    if ohrc.get("sun_elev_deg") is not None and ref.get("sun_elev_deg") is not None:
        out["sun_elev_gap_deg"] = abs(ohrc["sun_elev_deg"] - ref["sun_elev_deg"])
    return out
