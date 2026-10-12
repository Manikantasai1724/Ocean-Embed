"""Unit tests for Phase 2 baseline models and evaluation metrics."""

import unittest
import numpy as np
from oceanembed.models.baselines import ClimatologyBaseline, RidgeSurfaceBaseline
from oceanembed.eval.metrics import compute_level_metrics, evaluate_model_per_depth_and_region


class TestBaselinesAndMetrics(unittest.TestCase):
    def setUp(self):
        self.n_depths = 15
        self.n_lat = 10
        self.n_lon = 10
        self.dates = [f"2018-01-{d:02d}" for d in range(1, 11)]

    def test_climatology_baseline_fit_and_predict(self):
        # Synthetic target: (10, 15, 10, 10)
        target = 20.0 + np.random.randn(10, self.n_depths, self.n_lat, self.n_lon).astype(np.float32)
        mask_depth = np.ones((self.n_depths, self.n_lat, self.n_lon), dtype=bool)

        clim = ClimatologyBaseline(self.n_depths, self.n_lat, self.n_lon)
        clim.fit(self.dates, target, mask_depth)

        pred = clim.predict_for_date("2018-01-05")
        self.assertEqual(pred.shape, (self.n_depths, self.n_lat, self.n_lon))
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_ridge_baseline_fit_and_predict(self):
        n_times = 10
        surface_dict = {
            "sst": 28.0 + np.random.randn(n_times, self.n_lat, self.n_lon).astype(np.float32),
            "sss": 34.0 + np.random.randn(n_times, self.n_lat, self.n_lon).astype(np.float32),
            "sla": 0.05 * np.random.randn(n_times, self.n_lat, self.n_lon).astype(np.float32),
            "u_cur": 0.1 * np.random.randn(n_times, self.n_lat, self.n_lon).astype(np.float32),
            "v_cur": 0.1 * np.random.randn(n_times, self.n_lat, self.n_lon).astype(np.float32),
            "u_wind": -4.0 + np.random.randn(n_times, self.n_lat, self.n_lon).astype(np.float32),
            "v_wind": -5.0 + np.random.randn(n_times, self.n_lat, self.n_lon).astype(np.float32),
        }
        target = 22.0 + np.random.randn(n_times, self.n_depths, self.n_lat, self.n_lon).astype(np.float32)
        mask_ocean = np.ones((self.n_lat, self.n_lon), dtype=bool)
        mask_depth = np.ones((self.n_depths, self.n_lat, self.n_lon), dtype=bool)

        ridge = RidgeSurfaceBaseline(alpha=1.0, n_depths=self.n_depths)
        ridge.fit(surface_dict, target, mask_ocean, mask_depth)

        pred_3d = ridge.predict(surface_dict, mask_ocean, mask_depth, time_idx=0)
        self.assertEqual(pred_3d.shape, (self.n_depths, self.n_lat, self.n_lon))
        self.assertTrue(np.all(np.isfinite(pred_3d)))

    def test_compute_level_metrics(self):
        y_true = np.array([20.0, 21.0, 22.0, 23.0, 24.0])
        y_pred = np.array([20.1, 20.9, 22.2, 22.8, 24.0])
        mask = np.ones(5, dtype=bool)

        m = compute_level_metrics(y_pred, y_true, mask)
        self.assertIn("rmse", m)
        self.assertIn("mae", m)
        self.assertIn("bias", m)
        self.assertIn("corr", m)
        self.assertLess(m["rmse"], 0.3)
        self.assertGreater(m["corr"], 0.95)


if __name__ == "__main__":
    unittest.main()
