import unittest

import numpy as np

from trinetra.geometry.transforms import apply_h, mat_scale, mat_translate
from trinetra.matching.dense import match_dense
from trinetra.matching.fitting import robust_fit
from trinetra.matching.rescue import tile_consensus
from trinetra.registration.evaluation import nmi_with_baseline, spatial_heldout, trust_verdict
from trinetra.registration.options import resolve_options, retry_ladder


class OptionsTest(unittest.TestCase):
    def test_defaults_then_profile_then_run(self):
        opts = resolve_options({"ecc_mode": "off"}, {"auto_retry": 1})
        self.assertEqual(opts["ecc_mode"], "off")
        self.assertIs(opts["auto_retry"], True)
        self.assertEqual(opts["matcher"], "superglue")

    def test_rejects_unknown_option_and_value(self):
        with self.assertRaises(ValueError):
            resolve_options(None, {"nope": True})
        with self.assertRaises(ValueError):
            resolve_options(None, {"ecc_mode": "always"})

    def test_retry_ladder_adds_rescues(self):
        self.assertEqual(len(retry_ladder(resolve_options())), 1)
        ladder = retry_ladder(resolve_options(None, {"auto_retry": True}))
        self.assertEqual(len(ladder), 3)
        self.assertTrue(ladder[-1][1]["crater_matching"])


class FittingTest(unittest.TestCase):
    def test_robust_fit_recovers_a_similarity_despite_outliers(self):
        rng = np.random.default_rng(1)
        src = rng.uniform(0, 1000, (200, 2))
        H_true = mat_translate(12, -7) @ mat_scale(1.02)
        dst = apply_h(H_true, src)
        dst[:20] += rng.uniform(40, 80, (20, 2))
        H, mask, _ = robust_fit(src, dst, (1000, 1000))
        self.assertGreaterEqual(mask.sum(), 180)
        np.testing.assert_allclose(apply_h(H, src[20:]), dst[20:], atol=0.05)

    def test_tile_consensus_rejects_disagreeing_points(self):
        ka = np.zeros((4, 2), np.float32)
        kb = np.array([[5, 0], [5, 0], [40, 0], [-40, 0]], np.float32)
        self.assertIsNone(tile_consensus(ka, kb, np.ones(4), 6.0))
        agreed = tile_consensus(ka[:3], kb[[0, 1, 1]], np.ones(3), 6.0)
        self.assertEqual(len(agreed[0]), 3)

    def test_match_dense_finds_a_known_shift(self):
        rng = np.random.default_rng(2)
        ref = (rng.uniform(0, 255, (600, 600))).astype(np.uint8)
        ref = np.clip(np.cumsum(np.cumsum(ref.astype(float) - 127, 0), 1) / 400 + 128, 0, 255).astype(np.uint8)
        warped = np.roll(ref, (-4, -6), axis=(0, 1))       # ref content sits 6 px right, 4 px down
        valid = np.zeros_like(ref, bool)
        valid[40:560, 40:560] = True
        for subpixel in ("phase", "parabola"):
            ka, kb, _ = match_dense(warped, ref, valid, subpixel=subpixel)
            self.assertGreater(len(ka), 0)
            np.testing.assert_allclose(np.median(kb - ka, axis=0), [6, 4], atol=0.6)


class EvaluationTest(unittest.TestCase):
    def test_trust_needs_every_check(self):
        trusted, checks = trust_verdict(100, 0.4, 1.2, 0.3, 0.1)
        self.assertTrue(trusted)
        trusted, checks = trust_verdict(100, 0.4, 7.0, 0.3, 0.1)
        self.assertFalse(trusted)
        self.assertEqual([c["name"] for c in checks if not c["pass"]], ["Held-out RMSE"])

    def test_spatial_heldout_is_small_for_an_exact_model(self):
        rng = np.random.default_rng(3)
        src = np.column_stack([rng.uniform(0, 300, 300), rng.uniform(0, 5000, 300)])
        dst = apply_h(mat_translate(3, 4), src)
        rmse, folds = spatial_heldout(src, dst, (5000, 300), thresh=3.0)
        self.assertEqual(folds, 5)
        self.assertLess(rmse, 1e-3)

    def test_nmi_skips_tiny_overlaps(self):
        self.assertEqual(nmi_with_baseline(np.zeros((10, 10), np.uint8), np.ones((10, 10), np.uint8)), (None, None))


if __name__ == "__main__":
    unittest.main()
