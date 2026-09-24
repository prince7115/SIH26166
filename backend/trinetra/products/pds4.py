"""ISRO PDS4 products (OHRC, TMC-2): XML label access and a memory-mapped raster loader."""
import os
import xml.etree.ElementTree as ET

import numpy as np

NS = {"pds": "http://pds.nasa.gov/pds4/pds/v1",
      "isda": "https://isda.issdc.gov.in/pds4/isda/v1"}

DTYPES = {
    "UnsignedByte": "u1", "SignedByte": "i1",
    "UnsignedLSB2": "<u2", "SignedLSB2": "<i2",
    "UnsignedMSB2": ">u2", "SignedMSB2": ">i2",
    "UnsignedLSB4": "<u4", "SignedLSB4": "<i4",
    "IEEE754LSBSingle": "<f4", "IEEE754LSBDouble": "<f8",
}

_CORNER_TAGS = (("UL", "upper_left"), ("UR", "upper_right"), ("LL", "lower_left"), ("LR", "lower_right"))


def parse_number(value):
    """Leading number of a label value ("0.28 m" -> 0.28), or None."""
    try:
        return None if value is None else float(str(value).strip().split()[0])
    except (TypeError, ValueError, IndexError):
        return None


class Pds4Label:
    def __init__(self, xml_path):
        self.root = ET.parse(xml_path).getroot()

    def text(self, path):
        el = self.root.find(path, NS)
        return el.text.strip() if el is not None and el.text else None

    def number(self, path):
        return parse_number(self.text(path))

    def dims(self):
        """{"Line": n, "Sample": m} of the 2-D image array."""
        return {a.find("pds:axis_name", NS).text: int(a.find("pds:elements", NS).text)
                for a in self.root.findall(".//pds:Array_2D_Image/pds:Axis_Array", NS)}

    def corners(self):
        """Footprint corners with longitudes in [0, 360), and "refined" or "system level"
        for the label block they came from; (None, None) if the label has none."""
        for tag, source in (("Refined_Corner_Coordinates", "refined"),
                            ("System_Level_Coordinates", "system level")):
            geom = self.root.find(f".//isda:{tag}", NS)
            if geom is not None:
                return {k: {"lat": float(geom.find(f"isda:{t}_latitude", NS).text),
                            "lon": float(geom.find(f"isda:{t}_longitude", NS).text) % 360.0}
                        for k, t in _CORNER_TAGS}, source
        return None, None

    def instrument(self):
        name = None
        for obs in self.root.findall(".//pds:Observing_System_Component", NS):
            if (obs.findtext("pds:type", default="", namespaces=NS) or "").strip() == "Instrument":
                name = (obs.findtext("pds:name", default="", namespaces=NS) or "").strip() or None
        return name

    def gsd(self):
        """isda:pixel_resolution, else the first plausible resolution-like value in the label."""
        gsd = self.number(".//isda:pixel_resolution")
        if gsd is not None:
            return gsd
        for el in self.root.iter():
            tag = el.tag.split("}")[-1].lower()
            if ("resolution" in tag or "pixel_size" in tag) and el.text:
                try:
                    v = float(el.text.strip().split()[0])
                except ValueError:
                    continue
                if 0.05 < v < 500.0:
                    return v
        return None


def load_pds4(img_path, xml_path, name, allow_truncated=False, refuse_polar=True):
    """Memory-mapped raster in the label's own dtype, and its geometry.

    Returns (raster, corners_deg, (lines, samples), gsd_m, meta). A truncated file is refused
    unless allow_truncated, which keeps the complete lines and moves the lower corners up to match.
    """
    label = Pds4Label(xml_path)
    dims = label.dims()
    n_lines, n_samples = dims["Line"], dims["Sample"]

    dt_raw = label.text(".//pds:Element_Array/pds:data_type") or "UnsignedByte"
    if dt_raw not in DTYPES:
        raise ValueError(f"{name}: unmapped PDS4 data_type {dt_raw!r}.")
    dtype = np.dtype(DTYPES[dt_raw])

    corners, _ = label.corners()
    if corners is None:
        raise ValueError(f"{name}: the label carries no footprint corners. This is probably a "
                         f"BROWSE product (*_b_brw_*); use the data product (*_d_img_*).")

    meta = {
        "area": label.text(".//isda:area") or "?",
        "projection": label.text(".//isda:projection") or "?",
        "sun_elev": label.number(".//isda:sun_elevation"),
        "sun_azim": label.number(".//isda:sun_azimuth"),
        "dtype_pds4": dt_raw,
    }
    if refuse_polar and ("pole" in meta["area"].lower() or meta["projection"].lower().startswith("polar")):
        raise RuntimeError(f"{name} is a POLAR product (area={meta['area']}, projection="
                           f"{meta['projection']}); the plane-affine corner model does not "
                           f"hold there. Pick a Selenographic/Equatorial product.")

    expected = n_lines * n_samples * dtype.itemsize
    actual = os.path.getsize(img_path)
    if actual < expected:
        avail = actual // (n_samples * dtype.itemsize)
        if not allow_truncated:
            raise ValueError(f"{name} file truncated: {avail:,}/{n_lines:,} lines present.")
        frac = avail / float(n_lines)
        for top, bot in (("UL", "LL"), ("UR", "LR")):
            for k in ("lat", "lon"):
                corners[bot][k] = corners[top][k] + frac * (corners[bot][k] - corners[top][k])
        n_lines = avail

    raster = np.memmap(img_path, dtype=dtype, mode="r", shape=(n_lines, n_samples))
    return raster, corners, (n_lines, n_samples), label.gsd(), meta
