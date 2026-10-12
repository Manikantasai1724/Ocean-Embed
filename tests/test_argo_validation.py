"""Unit tests for in-situ Argo profiling float extraction, QC filtering, and collocation."""

import unittest
import numpy as np
from datetime import datetime

from oceanembed.eval.argo_collocate import (
    ArgoProfile,
    haversine_distance_km,
    filter_qc,
    interpolate_argo_to_target_depths,
    collocate_argo_with_model,
    compute_collocated_metrics,
)


class TestArgoValidation(unittest.TestCase):
    def test_haversine_distance(self):
        # Mumbai (18.9220 N, 72.8347 E) to Goa (15.4909 N, 73.8278 E) ~400 km
        dist = haversine_distance_km(18.9220, 72.8347, 15.4909, 73.8278)
        self.assertGreater(dist, 380.0)
        self.assertLess(dist, 430.0)

        # Same point distance is 0
        self.assertAlmostEqual(haversine_distance_km(10.0, 60.0, 10.0, 60.0), 0.0, places=4)

    def test_filter_qc_flags(self):
        profile = ArgoProfile(
            float_id="2901234",
            cycle_number=1,
            time="2018-01-15T10:00:00",
            latitude=12.0,
            longitude=65.0,
            depths=[5.0, 10.0, 20.0, 50.0, 100.0],
            temperatures=[28.0, 27.8, 99.9, 25.0, 21.0],
            qc_flags=[1, 2, 4, 1, 3],  # flags 4 and 3 are bad
        )

        clean = filter_qc(profile, allowed_flags=(1, 2))
        self.assertIsNotNone(clean)
        self.assertEqual(len(clean.depths), 3)
        self.assertEqual(clean.depths, [5.0, 10.0, 50.0])
        self.assertEqual(clean.temperatures, [28.0, 27.8, 25.0])
        self.assertEqual(clean.qc_flags, [1, 2, 1])

    def test_pchip_vertical_interpolation(self):
        target_depths = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 75.0, 100.0]
        raw_depths = np.array([4.5, 9.8, 19.5, 31.0, 52.0, 78.0, 105.0], dtype=np.float32)
        raw_temps = np.array([28.5, 28.2, 27.9, 27.1, 24.5, 21.8, 19.2], dtype=np.float32)

        interp_temps = interpolate_argo_to_target_depths(raw_depths, raw_temps, target_depths)
        self.assertEqual(len(interp_temps), len(target_depths))

        # Check surface extrapolation from 4.5m to 0m
        self.assertAlmostEqual(interp_temps[0], 28.5, places=1)
        # Check monotonic decrease in thermocline
        self.assertGreater(interp_temps[1], interp_temps[5])
        self.assertGreater(interp_temps[5], interp_temps[7])

    def test_collocation_tolerances(self):
        target_depths = [0.0, 50.0, 100.0]
        times = ["2018-01-15"]
        lats = np.array([12.0, 12.25, 12.5], dtype=np.float32)
        lons = np.array([65.0, 65.25, 65.5], dtype=np.float32)

        mod_preds = np.ones((1, 3, 3, 3), dtype=np.float32) * 25.0
        clim_preds = np.ones((1, 3, 3, 3), dtype=np.float32) * 24.5

        # 1. Matching profile: within 2 hours and 5 km
        p_match = ArgoProfile(
            float_id="MATCH1",
            cycle_number=1,
            time="2018-01-15T02:00:00",
            latitude=12.01,
            longitude=65.01,
            depths=[0.0, 50.0, 100.0],
            temperatures=[25.1, 24.9, 24.8],
            qc_flags=[1, 1, 1],
        )

        # 2. Out of time tolerance: 20 hours difference (> 12h)
        p_far_time = ArgoProfile(
            float_id="FARTIME",
            cycle_number=1,
            time="2018-01-15T22:00:00",  # diff from 2018-01-15T00:00:00 is 22h
            latitude=12.01,
            longitude=65.01,
            depths=[0.0, 50.0, 100.0],
            temperatures=[25.1, 24.9, 24.8],
            qc_flags=[1, 1, 1],
        )

        # 3. Out of spatial tolerance: > 50 km (> 25 km)
        p_far_space = ArgoProfile(
            float_id="FARSPACE",
            cycle_number=1,
            time="2018-01-15T02:00:00",
            latitude=14.0,  # > 200 km away
            longitude=65.0,
            depths=[0.0, 50.0, 100.0],
            temperatures=[25.1, 24.9, 24.8],
            qc_flags=[1, 1, 1],
        )

        matched = collocate_argo_with_model(
            profiles=[p_match, p_far_time, p_far_space],
            model_predictions=mod_preds,
            climatology_predictions=clim_preds,
            times=times,
            lats=lats,
            lons=lons,
            depths=target_depths,
            temporal_tolerance_hours=12.0,
            spatial_tolerance_km=25.0,
        )

        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0].float_id, "MATCH1")
        self.assertLessEqual(matched[0].distance_km, 25.0)
        self.assertLessEqual(matched[0].time_diff_hours, 12.0)


if __name__ == "__main__":
    unittest.main()
