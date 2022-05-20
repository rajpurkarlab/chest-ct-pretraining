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


class RSNAWindowDataLoaderTestCase(unittest.TestCase):
    def test_repeat_channel(self):
        config = {
            "data": {
                "use_hdf5": True,
                "dataset": "rsna",
                "type": "window",
                "targets": "rsna_pe_target",
                "channels": "repeat",
                "weighted_sample": True,
                "positive_only": True,
                "num_slices": 24,
                "min_abnormal_slice": 4,
                "sample_strategy": "random",
                "imsize": 256,
            },
            "transforms": {"RandomCrop": {"height": 224, "width": 224}},
        }
        config = OmegaConf.create(config)
        dataset = dataset_3d.PEDatasetWindow(config, split="train")
        dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

        x, y, ids = next(iter(dataloader))

        self.assertEqual(x.shape[0], 2, "batch size inccorect")
        self.assertEqual(x.shape[1], 3, "number of channels incorrect")
        self.assertEqual(x.shape[2], 24, "slice number incorrect")
        self.assertEqual(x.shape[3], 224, "width incorrect")
        self.assertEqual(x.shape[4], 224, "height incorrect")
        self.assertTrue(
            torch.all(x[0, 0, :, :, :].eq(x[0, 1, :, :, :])), "channels not repeating"
        )
        self.assertTrue(
            torch.all(x[0, 1, :, :, :].eq(x[0, 2, :, :, :])), "channels not repeating"
        )
        self.assertTrue((x.max() <= 1.0 and x.min() >= -1.0), "input normalized")

        utils.visualize_examples(x[0].numpy(), 10, "./data/test/rsna_window_repeat.png")

    def test_eq_read_from_dicom(self):
        hdf5_config = {
            "data": {
                "use_hdf5": True,
                "dataset": "rsna",
                "type": "window",
                "targets": "rsna_pe_target",
                "channels": "repeat",
                "weighted_sample": True,
                "positive_only": True,
                "num_slices": 24,
                "min_abnormal_slice": 4,
                "sample_strategy": "random",
                "imsize": 256,
            },
            "transforms": {"RandomCrop": {"height": 224, "width": 224}},
        }
        hdf5_config = OmegaConf.create(hdf5_config)
        raw_config = {
            "data": {
                "use_hdf5": False,
                "dataset": "rsna",
                "type": "window",
                "targets": "rsna_pe_target",
                "channels": "repeat",
                "weighted_sample": True,
                "positive_only": True,
                "num_slices": 24,
                "min_abnormal_slice": 4,
                "sample_strategy": "random",
                "imsize": 256,
            },
            "transforms": {"RandomCrop": {"height": 224, "width": 224}},
        }
        raw_config = OmegaConf.create(raw_config)
        hdf5_dataset = dataset_3d.PEDatasetWindow(hdf5_config, split="test")
        raw_dataset = dataset_3d.PEDatasetWindow(raw_config, split="test")
        hdf5_dataloader = DataLoader(hdf5_dataset, batch_size=4, num_workers=0, shuffle=False)
        raw_dataloader = DataLoader(raw_dataset, batch_size=4, num_workers=0, shuffle=False)

        hdf5_x, hdf5_y, hdf5_ids = next(iter(hdf5_dataloader))
        raw_x, raw_y, raw_ids = next(iter(raw_dataloader))

        self.assertTrue(
            torch.all(hdf5_x.eq(raw_x)), "different inputs values when reading from raw dicom vs hdf5"
        )
        self.assertTrue(
            torch.all(hdf5_y.eq(raw_y)), "different target labels when reading from raw dicom vs hdf5"
        )


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
