import pandas as pd
import unittest
import torch
import os
import sys

sys.path.append(os.getcwd())

from pe_models import utils
from pe_models.constants import *
from pe_models.datasets import dataset_3d
from torch.utils.data import DataLoader
from omegaconf import OmegaConf


class RSNA3DDataLoaderTestCase(unittest.TestCase):
    def test_repeat_channel(self):
        config = {
            "data": {
                "use_hdf5": False,
                "dataset": "rsna",
                "type": "3d",
                "targets": "rsna_pe_target",
                "channels": "repeat",
                "weighted_sample": True,
                "positive_only": True,
                "num_slices": 150,
                "sample_strategy": "random",
                "imsize": 256,
            },
            "transforms": {"RandomCrop": {"height": 224, "width": 224}},
        }
        config = OmegaConf.create(config)
        dataset = dataset_3d.PEDataset3D(config, split="train")
        dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

        x, y, ids = next(iter(dataloader))

        self.assertEqual(x.shape[0], 2, "batch size inccorect")
        self.assertEqual(x.shape[1], 3, "number of channels incorrect")
        self.assertEqual(x.shape[2], 150, "slice number incorrect")
        self.assertEqual(x.shape[3], 224, "width incorrect")
        self.assertEqual(x.shape[4], 224, "height incorrect")
        self.assertTrue(
            torch.all(x[0, 0, :, :, :].eq(x[0, 1, :, :, :])), "channels not repeating"
        )
        self.assertTrue(
            torch.all(x[0, 1, :, :, :].eq(x[0, 2, :, :, :])), "channels not repeating"
        )
        self.assertTrue((x.max() <= 1.0 and x.min() >= -1.0), "input normalized")

        utils.visualize_examples(x[0].numpy(), 10, "./data/test/rsna_3d_repeat.png")

    def test_window_channel(self):
        config = {
            "data": {
                "use_hdf5": False,
                "dataset": "rsna",
                "type": "3d",
                "targets": "rsna_pe_target",
                "channels": "window",
                "weighted_sample": True,
                "positive_only": True,
                "num_slices": 150,
                "sample_strategy": "random",
                "imsize": 256,
            },
            "transforms": {"RandomCrop": {"height": 224, "width": 224}},
        }
        config = OmegaConf.create(config)
        dataset = dataset_3d.PEDataset3D(config, split="train")
        dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

        x, y, ids = next(iter(dataloader))

        self.assertEqual(x.shape[0], 2, "batch size inccorect")
        self.assertEqual(x.shape[1], 3, "number of channels incorrect")
        self.assertEqual(x.shape[2], 150, "slice number incorrect")
        self.assertEqual(x.shape[3], 224, "width incorrect")
        self.assertEqual(x.shape[4], 224, "height incorrect")
        self.assertFalse(
            torch.all(x[0, 0, :, :].eq(x[0, 1, :, :])), "channels repeating"
        )
        self.assertFalse(
            torch.all(x[0, 1, :, :].eq(x[0, 2, :, :])), "channels repeating"
        )
        self.assertTrue((x.max() <= 1.0 and x.min() >= -1.0), "input normalized")

        utils.visualize_examples(x[0].numpy(), 10, "./data/test/rsna_3d_window.png")

    def test_neighbor_channel(self):
        config = {
            "data": {
                "use_hdf5": False,
                "dataset": "rsna",
                "type": "3d",
                "targets": "rsna_pe_target",
                "channels": "neighbor",
                "weighted_sample": True,
                "positive_only": True,
                "num_slices": 150,
                "sample_strategy": "random",
                "imsize": 256,
            },
            "transforms": {"RandomCrop": {"height": 224, "width": 224}},
        }
        config = OmegaConf.create(config)
        dataset = dataset_3d.PEDataset3D(config, split="train")
        dataloader = DataLoader(dataset, batch_size=2, num_workers=0, shuffle=False)

        x, y, ids = next(iter(dataloader))

        self.assertEqual(x.shape[0], 2, "batch size inccorect")
        self.assertEqual(x.shape[1], 3, "number of channels incorrect")
        self.assertEqual(x.shape[2], 150, "slice number incorrect")
        self.assertEqual(x.shape[3], 224, "width incorrect")
        self.assertEqual(x.shape[4], 224, "height incorrect")
        self.assertFalse(
            torch.all(x[1, 0, :, :].eq(x[1, 1, :, :])), "channels repeating"
        )
        self.assertFalse(
            torch.all(x[1, 1, :, :].eq(x[1, 2, :, :])), "channels repeating"
        )
        self.assertTrue((x.max() <= 1.0 and x.min() >= -1.0), "input normalized")
        utils.visualize_examples(x[0].numpy(), 10, "./data/test/rsna_3d_neighbor.png")

    def test_repeat_channel_from_hdf5(self):
        config = {
            "data": {
                "use_hdf5": True,
                "dataset": "rsna",
                "type": "3d",
                "targets": "rsna_pe_target",
                "channels": "repeat",
                "weighted_sample": True,
                "positive_only": True,
                "num_slices": 150,
                "sample_strategy": "random",
                "imsize": 256,
            },
            "transforms": {"RandomCrop": {"height": 224, "width": 224}},
        }
        config = OmegaConf.create(config)
        dataset = dataset_3d.PEDataset3D(config, split="train")
        dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

        x, y, ids = next(iter(dataloader))

        self.assertEqual(x.shape[0], 2, "batch size inccorect")
        self.assertEqual(x.shape[1], 3, "number of channels incorrect")
        self.assertEqual(x.shape[2], 150, "slice number incorrect")
        self.assertEqual(x.shape[3], 224, "width incorrect")
        self.assertEqual(x.shape[4], 224, "height incorrect")
        self.assertTrue(
            torch.all(x[0, 0, :, :, :].eq(x[0, 1, :, :, :])), "channels not repeating"
        )
        self.assertTrue(
            torch.all(x[0, 1, :, :, :].eq(x[0, 2, :, :, :])), "channels not repeating"
        )
        self.assertTrue((x.max() <= 1.0 and x.min() >= -1.0), "input normalized")

        utils.visualize_examples(x[0].numpy(), 10, "./data/test/rsna_3d_repeat_hdf5.png")


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
