import unittest

import numpy as np

from trinetra.geometry.geodesy import corner_affine, latlon_to_pixel, unwrap_lon
from trinetra.geometry.transforms import H_at_level, apply_h, mat_scale, mat_translate

CORNERS = {"UL": {"lat": 1.11, "lon": 359.9}, "UR": {"lat": 1.11, "lon": 0.1},
           "LL": {"lat": 0.15, "lon": 359.92}, "LR": {"lat": 0.15, "lon": 0.12}}
SHAPE = (5000, 800)


class GeodesyTest(unittest.TestCase):
    def test_unwrap_lon_stays_near_reference(self):
        self.assertAlmostEqual(float(unwrap_lon(0.1, 359.9)), 360.1)
        self.assertAlmostEqual(float(unwrap_lon(359.9, 0.1)), -0.1)

    def test_corner_affine_hits_the_corners_across_the_meridian(self):
        M, t = corner_affine(CORNERS, SHAPE, lon_ref=359.9)
        lat, lon = M @ np.array([0.0, SHAPE[1] - 1]) + t
        self.assertAlmostEqual(lat, 1.11, places=6)
        self.assertAlmostEqual(lon, 360.1, places=6)

    def test_latlon_to_pixel_inverts_the_corner_model(self):
        to_px = latlon_to_pixel(CORNERS, SHAPE)
        row, col = to_px(0.15, 359.92)
        self.assertAlmostEqual(row, SHAPE[0] - 1, places=3)
        self.assertAlmostEqual(col, 0.0, places=3)


class TransformTest(unittest.TestCase):
    def test_apply_h_translates_points(self):
        pts = apply_h(mat_translate(3, -2), [[1, 1], [0, 0]])
        np.testing.assert_allclose(pts, [[4, -1], [3, -2]])

    def test_H_at_level_round_trip(self):
        H = mat_translate(5, 7) @ mat_scale(1.1)
        back = H_at_level(H_at_level(H, 0.5, 2.0), 2.0, 0.5)
        np.testing.assert_allclose(back, H)

    def test_H_at_level_keeps_ground_meaning(self):
        H = mat_translate(10, 0)          # 10 px at 1 m/px = 10 m
        np.testing.assert_allclose(apply_h(H_at_level(H, 1.0, 2.0), [[0, 0]]), [[5, 0]])


if __name__ == "__main__":
    unittest.main()
