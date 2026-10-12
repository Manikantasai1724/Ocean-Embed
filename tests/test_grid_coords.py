"""Unit test verifying grid coordinates, dimension counts, and depth targets."""

import unittest
import numpy as np


class TestGridCoordinates(unittest.TestCase):
    def test_domain_grid_dimensions(self):
        """Verify the 0.25 degree North Indian Ocean grid definition."""
        lat_min, lat_max, d_lat = 5.0, 30.0, 0.25
        lon_min, lon_max, d_lon = 45.0, 105.0, 0.25

        lats = np.arange(lat_min, lat_max + d_lat / 2.0, d_lat)
        lons = np.arange(lon_min, lon_max + d_lon / 2.0, d_lon)

        self.assertEqual(len(lats), 101, f"Expected 101 latitude points, got {len(lats)}")
        self.assertEqual(len(lons), 241, f"Expected 241 longitude points, got {len(lons)}")
        self.assertTrue(np.isclose(lats[0], 5.0))
        self.assertTrue(np.isclose(lats[-1], 30.0))
        self.assertTrue(np.isclose(lons[0], 45.0))
        self.assertTrue(np.isclose(lons[-1], 105.0))

    def test_target_depths(self):
        """Verify target depth levels count and values."""
        target_depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
        self.assertEqual(len(target_depths), 15)
        self.assertEqual(target_depths[0], 0)
        self.assertEqual(target_depths[-1], 1000)


if __name__ == "__main__":
    unittest.main()
