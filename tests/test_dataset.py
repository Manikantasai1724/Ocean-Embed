"""Unit test verifying OceanPatchDataset loading, patch extraction, and shape compatibility."""

import unittest
import torch
from oceanembed.data.dataset import OceanPatchDataset


class TestOceanPatchDataset(unittest.TestCase):
    def setUp(self):
        self.zarr_path = "data/processed/oceanembed_slice_201801.zarr"

    def test_patch_extraction_and_shapes(self):
        dataset = OceanPatchDataset(
            zarr_path=self.zarr_path,
            patch_size=64,
            is_train=True,
            include_static_channels=True,
        )

        self.assertEqual(len(dataset), 31)

        x, y, mask = dataset[0]

        # x channels: 7 inputs + 7 masks + 4 static (lat, lon, sin_doy, cos_doy) = 18 channels
        self.assertEqual(x.shape, (18, 64, 64))
        # y: 15 target depths
        self.assertEqual(y.shape, (15, 64, 64))
        # mask: 15 target depths
        self.assertEqual(mask.shape, (15, 64, 64))

        # Check tensor dtypes
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.float32)

        # Check masked MSE computation
        masked_diff = (x[0:1] - y[0:1]) * mask[0:1]
        loss = torch.sum(masked_diff ** 2) / (torch.sum(mask[0:1]) + 1e-6)
        self.assertTrue(bool(torch.isfinite(loss).item()))


if __name__ == "__main__":
    unittest.main()
