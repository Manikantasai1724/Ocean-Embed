"""Unit tests for OceanEmbed masks, regridding, and vertical interpolation."""

import unittest
import numpy as np
from oceanembed.data.masks import (
    get_domain_coords,
    create_realistic_nio_bathymetry,
    compute_per_depth_bathymetry_mask,
    compute_coastal_buffer_mask,
)
from oceanembed.data.regrid import (
    regrid_bilinear_2d,
    regrid_area_average_2d,
    compare_regrid_methods,
)
from oceanembed.data.harmonize import (
    interpolate_vertical_profiles_pchip,
    GLORYS_NATIVE_DEPTHS,
)


class TestMasksAndBathymetry(unittest.TestCase):
    def setUp(self):
        self.lats, self.lons = get_domain_coords()
        self.target_depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

    def test_bathymetry_dimensions(self):
        depth, is_ocean = create_realistic_nio_bathymetry(self.lats, self.lons)
        self.assertEqual(depth.shape, (101, 241))
        self.assertEqual(is_ocean.shape, (101, 241))
        self.assertTrue(np.any(is_ocean))
        self.assertTrue(np.any(~is_ocean))

    def test_per_depth_bathymetry_mask(self):
        depth, is_ocean = create_realistic_nio_bathymetry(self.lats, self.lons)
        mask = compute_per_depth_bathymetry_mask(depth, self.target_depths)
        self.assertEqual(mask.shape, (15, 101, 241))

        # Shallow water parcel (e.g. 50m shelf) must be False at 500m
        shallow_idx = np.where((depth > 0) & (depth < 100.0))
        if len(shallow_idx[0]) > 0:
            i, j = shallow_idx[0][0], shallow_idx[1][0]
            # Valid at surface
            self.assertTrue(mask[0, i, j])
            # Must be MASKED at 500m (index 12)
            self.assertFalse(mask[12, i, j])

    def test_coastal_buffer_mask(self):
        depth, is_ocean = create_realistic_nio_bathymetry(self.lats, self.lons)
        is_coastal = compute_coastal_buffer_mask(is_ocean, buffer_cells=2)
        self.assertEqual(is_coastal.shape, (101, 241))
        # Land points must not be flagged as coastal ocean
        self.assertTrue(np.all(~is_coastal[~is_ocean]))


class TestRegriddingRoutines(unittest.TestCase):
    def test_regrid_comparison(self):
        # Create synthetic high-res source grid (0.05 degree, 11x11)
        src_lats = np.linspace(10.0, 10.5, 11, dtype=np.float32)
        src_lons = np.linspace(70.0, 70.5, 11, dtype=np.float32)
        lon_g, lat_g = np.meshgrid(src_lons, src_lats)
        src_data = 28.0 + 0.5 * np.sin(lat_g) + 0.5 * np.cos(lon_g)

        # Destination grid (0.25 degree, 3x3)
        dst_lats = np.array([10.0, 10.25, 10.5], dtype=np.float32)
        dst_lons = np.array([70.0, 70.25, 70.5], dtype=np.float32)

        res_bilinear = regrid_bilinear_2d(src_data, src_lats, src_lons, dst_lats, dst_lons)
        res_area = regrid_area_average_2d(src_data, src_lats, src_lons, dst_lats, dst_lons, dst_resolution=0.25)

        self.assertEqual(res_bilinear.shape, (3, 3))
        self.assertEqual(res_area.shape, (3, 3))

        metrics = compare_regrid_methods(src_data, src_lats, src_lons, dst_lats, dst_lons, var_name="SST_test")
        self.assertIn("mean_abs_diff", metrics)
        self.assertIn("correlation", metrics)
        self.assertGreater(metrics["correlation"], 0.9)


class TestVerticalInterpolation(unittest.TestCase):
    def test_pchip_vertical_profile(self):
        # Native profile with thermocline
        n_native = len(GLORYS_NATIVE_DEPTHS)
        z = GLORYS_NATIVE_DEPTHS
        profile = 5.0 + 23.0 / (1.0 + np.exp((z - 80.0) / 40.0))  # Surface ~27C, deep ~5C

        # 3D tensor: shape (n_native, 2, 2)
        native_tensor = np.zeros((n_native, 2, 2), dtype=np.float32)
        for i in range(2):
            for j in range(2):
                native_tensor[:, i, j] = profile

        target_depths = np.array([0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000], dtype=np.float32)
        interp = interpolate_vertical_profiles_pchip(
            native_tensor,
            native_depths=z,
            target_depths=target_depths,
            map_surface_level=True,
        )

        self.assertEqual(interp.shape, (15, 2, 2))
        # Surface must be close to ~27C
        self.assertAlmostEqual(interp[0, 0, 0], profile[0], delta=0.5)
        # Deepest level must be close to ~5C
        self.assertAlmostEqual(interp[-1, 0, 0], 5.0, delta=1.0)
        # Monotonically decreasing with depth
        self.assertTrue(np.all(np.diff(interp[:, 0, 0]) <= 0.0))


if __name__ == "__main__":
    unittest.main()
