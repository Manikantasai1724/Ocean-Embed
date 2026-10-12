"""Unit tests for OceanPredictor and FastAPI endpoints."""

import unittest
from fastapi.testclient import TestClient
import numpy as np

from oceanembed.infer.predict import OceanPredictor
from api.main import app, startup_event


class TestApiAndInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Run startup to initialize predictor
        startup_event()
        cls.client = TestClient(app)
        cls.predictor = OceanPredictor()

    def test_predictor_metadata(self):
        meta = self.predictor.get_metadata()
        self.assertEqual(meta["domain_name"], "North Indian Ocean")
        self.assertEqual(len(meta["latitudes"]), 101)
        self.assertEqual(len(meta["longitudes"]), 241)
        self.assertEqual(len(meta["target_depths"]), 15)
        self.assertIn("attention_hybrid", meta["available_models"])

    def test_predictor_3d_shape(self):
        date = self.predictor.times[-1]
        field_3d = self.predictor.predict_3d(date, "attention_hybrid")
        self.assertEqual(field_3d.shape, (15, 101, 241))
        # Top-left corner (Persian gulf land) must be NaN masked
        self.assertTrue(np.isnan(field_3d[0, -1, 0]))

    def test_predictor_slice_and_profile(self):
        date = self.predictor.times[-1]
        field_slice = self.predictor.get_field_slice(date, depth_m=50.0)
        self.assertEqual(field_slice["depth_m"], 50.0)
        self.assertEqual(len(field_slice["grid"]), 101)
        self.assertEqual(len(field_slice["grid"][0]), 241)

        profile = self.predictor.get_profile(date, lat=15.0, lon=65.0)
        self.assertEqual(len(profile["profile"]), 15)
        self.assertTrue(profile["is_ocean"])

    def test_api_health_and_meta(self):
        res_h = self.client.get("/health")
        self.assertEqual(res_h.status_code, 200)
        data_h = res_h.json()
        self.assertEqual(data_h["status"], "healthy")

        res_m = self.client.get("/meta")
        self.assertEqual(res_m.status_code, 200)
        data_m = res_m.json()
        self.assertEqual(data_m["grid_shape"], [101, 241])

    def test_api_field_and_profile(self):
        date = self.predictor.times[-1]
        res_f = self.client.get(f"/field?date={date}&depth=75&model=attention_hybrid")
        self.assertEqual(res_f.status_code, 200)
        data_f = res_f.json()
        self.assertEqual(data_f["depth_m"], 75.0)

        res_p = self.client.get(f"/profile?date={date}&lat=12.0&lon=65.0&model=attention_hybrid")
        self.assertEqual(res_p.status_code, 200)
        data_p = res_p.json()
        self.assertEqual(len(data_p["profile"]), 15)

    def test_api_compare_and_metrics(self):
        date = self.predictor.times[-1]
        res_c = self.client.get(f"/compare?date={date}&depth=50&model=attention_hybrid")
        self.assertEqual(res_c.status_code, 200)
        data_c = res_c.json()
        self.assertIn("rmse_c", data_c)
        self.assertIn("prediction_grid", data_c)
        self.assertIn("error_grid", data_c)

        res_met = self.client.get("/metrics?model=attention_hybrid")
        self.assertEqual(res_met.status_code, 200)


if __name__ == "__main__":
    unittest.main()
