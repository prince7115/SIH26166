import json
import os
import tempfile
import unittest

from trinetra import config
from trinetra.products.discovery import resolve_pair_dir
from trinetra.products.pds4 import load_pds4, parse_number
from trinetra.products.sun import apply_sun_override, azimuth_gap, make_sun, read_sun_json

OHRC_PAIR = os.path.join(config.DATA_RAW, "ohrc", "pair01_equatorial")


class SunTest(unittest.TestCase):
    def test_incidence_becomes_elevation(self):
        sun = make_sun(incidence=81.6, azim=365.0, conv="north_cw")
        self.assertAlmostEqual(sun["elev_deg"], 8.4)
        self.assertAlmostEqual(sun["azim_deg"], 5.0)

    def test_unknown_convention_is_marked(self):
        self.assertEqual(make_sun(azim=10, conv="sideways")["azim_convention"], "unknown")
        self.assertIsNone(make_sun())

    def test_website_override_replaces_single_fields(self):
        base = make_sun(elev=30, azim=100, conv="north_cw", source="label")
        merged = apply_sun_override(base, {"elev_deg": 12.5})
        self.assertEqual((merged["elev_deg"], merged["azim_deg"], merged["source"]), (12.5, 100.0, "website"))
        self.assertIs(apply_sun_override(base, {}), base)

    def test_sun_json(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "sun.json"), "w") as f:
                json.dump({"elevation_deg": 20, "azimuth_deg": 90, "azimuth_convention": "image_cw"}, f)
            self.assertEqual(read_sun_json(d)["azim_convention"], "image_cw")
            self.assertIsNone(read_sun_json(os.path.join(d, "missing")))

    def test_azimuth_gap_wraps(self):
        self.assertAlmostEqual(azimuth_gap(350, 10), 20)


class LabelTest(unittest.TestCase):
    def test_parse_number(self):
        self.assertEqual(parse_number("0.28 m"), 0.28)
        self.assertIsNone(parse_number(""))
        self.assertIsNone(parse_number("n/a"))

    @unittest.skipUnless(os.path.isdir(OHRC_PAIR), "sample data not present")
    def test_ohrc_label_and_raster(self):
        img, xml = resolve_pair_dir(config.DATA_RAW, "ohrc", "pair01_equatorial")
        raster, corners, shape, gsd, meta = load_pds4(img, xml, "OHRC")
        self.assertEqual(raster.shape, shape)
        self.assertEqual(set(corners), {"UL", "UR", "LL", "LR"})
        self.assertAlmostEqual(gsd, 0.28)
        self.assertEqual(meta["dtype_pds4"], "UnsignedByte")


if __name__ == "__main__":
    unittest.main()
