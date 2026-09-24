import unittest

from trinetra import config
from trinetra.api import create_app
from trinetra.sensors import SENSORS


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = create_app().test_client()

    def test_pipelines_lists_every_sensor(self):
        data = self.client.get("/api/pipelines").get_json()
        self.assertEqual({p["id"] for p in data["pipelines"]}, set(SENSORS))

    def test_unknown_pipeline(self):
        resp = self.client.get("/api/dataset?pipeline=nope&pair_id=pair01_equatorial")
        self.assertEqual(resp.status_code, 404)

    def test_pair_id_must_be_an_available_pair(self):
        for pair_id in ("../../..", "..\\..", "pair99_missing", ""):
            with self.subTest(pair_id=pair_id):
                resp = self.client.get("/api/dataset", query_string={"pipeline": "ohrc_nac", "pair_id": pair_id})
                self.assertEqual(resp.status_code, 404)
                resp = self.client.get("/api/run", query_string={"pipeline": "ohrc_nac", "pair_id": pair_id})
                self.assertEqual(resp.status_code, 404)

    def test_only_metrics_can_be_downloaded(self):
        pairs = SENSORS["ohrc_nac"].available_pairs(config.DATA_RAW)
        if not pairs:
            self.skipTest("sample data not present")
        resp = self.client.get("/api/results/download/report",
                               query_string={"pipeline": "ohrc_nac", "pair_id": pairs[0]})
        self.assertEqual(resp.status_code, 404)


if __name__ == "__main__":
    unittest.main()
