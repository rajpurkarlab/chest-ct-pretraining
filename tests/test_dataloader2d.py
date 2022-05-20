import pandas as pd
import unittest
import torch
import os
import sys

sys.path.append(os.getcwd())

from pe_models.constants import *
from pe_models import utils
from pe_models.datasets import dataset_2d, dataset_3d
from torch.utils.data import DataLoader
from omegaconf import OmegaConf


class RSNA2DDataLoaderTestCase(unittest.TestCase):
    def test_repeat_channel(self):
        config = {
            "data": {
                "use_hdf5": False,
                "dataset": "rsna",
                "type": "2d",
                "targets": "rsna_pe_slice_target",
                "channels": "repeat",
                "weighted_sample": True,
                "positive_only": True,
                "imsize": 512,
            }
        }
        config = OmegaConf.create(config)
        dataset = dataset_2d.PEDataset2D(config, split="train")
        dataloader = DataLoader(dataset, batch_size=10, num_workers=8, shuffle=True)

        x, y, ids = next(iter(dataloader))

        self.assertEqual(x.shape[0], 10, "batch size inccorect")
        self.assertEqual(x.shape[1], 3, "incorrect num channels")
        self.assertEqual(x.shape[2], 512, "incorrect height")
        self.assertEqual(x.shape[3], 512, "incorrect width")
        self.assertTrue(
            torch.all(x[0, 0, :, :].eq(x[0, 1, :, :])), "channels not repeating"
        )
        self.assertTrue(
            torch.all(x[0, 1, :, :].eq(x[0, 2, :, :])), "channels not repeating"
        )
        self.assertTrue((x.max() <= 1.0 and x.min() >= -1.0), "input normalized")
        utils.visualize_examples(x.numpy(), 10, "./data/test/rsna_2d_repeat.png")

    def test_window_channel(self):
        config = {
            "data": {
                "use_hdf5": False,
                "dataset": "rsna",
                "type": "2d",
                "targets": "rsna_pe_slice_target",
                "channels": "window",
                "weighted_sample": True,
                "positive_only": True,
                "imsize": 512,
            }
        }
        config = OmegaConf.create(config)
        dataset = dataset_2d.PEDataset2D(config, split="train")
        dataloader = DataLoader(dataset, batch_size=10, num_workers=8, shuffle=True)

        x, y, ids = next(iter(dataloader))

        self.assertEqual(x.shape[0], 10, "batch size inccorect")
        self.assertEqual(x.shape[1], 3, "incorrect num channels")
        self.assertEqual(x.shape[2], 512, "incorrect height")
        self.assertEqual(x.shape[3], 512, "incorrect width")
        self.assertFalse(
            torch.all(x[0, 0, :, :].eq(x[0, 1, :, :])), "channels not repeating"
        )
        self.assertFalse(
            torch.all(x[0, 1, :, :].eq(x[0, 2, :, :])), "channels not repeating"
        )
        self.assertTrue((x.max() <= 1.0 and x.min() >= -1.0), "input normalized")
        utils.visualize_examples(x.numpy(), 10, "./data/test/rsna_2d_window.png")

    def test_neighbor_channel(self):
        config = {
            "data": {
                "use_hdf5": False,
                "dataset": "rsna",
                "type": "2d",
                "targets": "rsna_pe_slice_target",
                "channels": "neighbor",
                "weighted_sample": True,
                "positive_only": True,
                "imsize": 512,
            }
        }
        config = OmegaConf.create(config)
        dataset = dataset_2d.PEDataset2D(config, split="train")
        dataloader = DataLoader(dataset, batch_size=10, num_workers=8, shuffle=True)

        x, y, ids = next(iter(dataloader))

        self.assertEqual(x.shape[0], 10, "batch size inccorect")
        self.assertEqual(x.shape[1], 3, "incorrect num channels")
        self.assertEqual(x.shape[2], 512, "incorrect height")
        self.assertEqual(x.shape[3], 512, "incorrect width")
        self.assertFalse(
            torch.all(x[0, 0, :, :].eq(x[0, 1, :, :])), "channels not repeating"
        )
        self.assertFalse(
            torch.all(x[0, 1, :, :].eq(x[0, 2, :, :])), "channels not repeating"
        )
        self.assertTrue((x.max() <= 1.0 and x.min() >= -1.0), "input normalized")
        utils.visualize_examples(x.numpy(), 10, "./data/test/rsna_2d_neighbor.png")


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
