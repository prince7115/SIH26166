"""The OHRC side of every pair."""
from dataclasses import dataclass

import numpy as np

from ..products.discovery import product_id, resolve_pair_dir
from ..products.pds4 import load_pds4
from ..products.sun import label_sun
from ..imaging.radiometry import stretch_to_uint8, to_uint8
from ..reporting.encoding import img_to_b64

FALLBACK_GSD_M = 0.26    # used only if the label has no resolution field


@dataclass
class OhrcProduct:
    raster: np.ndarray
    corners: dict
    shape: tuple
    gsd: float
    product_id: str
    xml_path: str
    meta: dict

    @property
    def sun(self):
        return label_sun(self.meta["sun_elev"], self.meta["sun_azim"])

    def summary(self):
        return f"OHRC {self.shape[0]:,} × {self.shape[1]:,} px @ {self.gsd:.2f} m/px"


def read_ohrc(pair_id, data_raw):
    img, xml = resolve_pair_dir(data_raw, "ohrc", pair_id)
    raster, corners, shape, gsd, meta = load_pds4(img, xml, "OHRC")
    return OhrcProduct(raster=raster, corners=corners, shape=shape, gsd=gsd or FALLBACK_GSD_M,
                       product_id=product_id(img), xml_path=xml, meta=meta)


def preview(raster):
    """Every 16th pixel, stretched to 8 bit, as a PNG data URL."""
    return img_to_b64(to_uint8(stretch_to_uint8(raster[::16, ::16])))
