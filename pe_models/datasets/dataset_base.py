import torch
import h5py
import pydicom
import numpy as np
import pandas as pd
import cv2
import numpy.random as random

from ..constants import *
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import rotate
from omegaconf import OmegaConf


class PEDatasetBase(Dataset):
    def __init__(self, cfg, split="train", transform=None):

        self.cfg = cfg
        self.transform = transform
        self.split = split
        self.hdf5_dataset = None

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def read_dicom(self, file_path: str):

        # read dicom
        dcm = pydicom.dcmread(file_path)
        pixel_array = dcm.pixel_array

        # rescale
        intercept = dcm.RescaleIntercept
        slope = dcm.RescaleSlope
        pixel_array = pixel_array * slope + intercept

        # resize
        # TODO: maybe use augmentation resize instead
        resize_shape = self.cfg.data.imsize
        pixel_array = cv2.resize(
            pixel_array, (resize_shape, resize_shape), interpolation=cv2.INTER_AREA
        )

        return pixel_array

    def windowing(self, pixel_array: np.array, window_center: int, window_width: int):

        lower = window_center - window_width // 2
        upper = window_center + window_width // 2
        pixel_array = np.clip(pixel_array.copy(), lower, upper)
        pixel_array = (pixel_array - lower) / (upper - lower)

        return pixel_array

    def process_slice(self, slice_info: pd.Series):
        """process slice with windowing, resize and tranforms"""

        # window
        if self.cfg.data.channels == "repeat":
            slice_path = RSNA_TRAIN_DIR / slice_info[RSNA_INSTANCE_PATH_COL]
            slice_array = self.read_dicom(slice_path)
            ct_slice = self.windowing(
                slice_array, 400, 1000
            )  # use PE window by default
            # create 3 channels after converting to Tensor
            # using torch.repeat won't take up 3x memory

        elif self.cfg.data.channels == "neighbor":
            slice_paths = [
                RSNA_TRAIN_DIR / slice_info[RSNA_PREV_INSTANCE_COL],
                RSNA_TRAIN_DIR / slice_info[RSNA_INSTANCE_PATH_COL],
                RSNA_TRAIN_DIR / slice_info[RSNA_NEXT_INSTANCE_COL],
            ]
            slice_arrays = np.array(
                [self.read_dicom(slice_path) for slice_path in slice_paths]
            )
            ct_slice = self.windowing(slice_arrays, 400, 1000)
            ct_slice = np.stack(ct_slice)

        else:
            slice_path = RSNA_TRAIN_DIR / slice_info[RSNA_INSTANCE_PATH_COL]
            slice_array = self.read_dicom(slice_path)
            ct_slice = [
                self.windowing(slice_array, -600, 1500),  # LUNG window
                self.windowing(slice_array, 400, 1000),  # PE window
                self.windowing(slice_array, 40, 400), # MEDIASTINAL window
            ]  
            ct_slice = np.stack(ct_slice)

        return ct_slice

    def read_from_hdf5(self, key, slice_idx=None, hdf5_path = RSNA_STUDY_HDF5):
        if self.hdf5_dataset is None: 
            self.hdf5_dataset = h5py.File(hdf5_path, 'r')
       
        if slice_idx is None: 
            arr = self.hdf5_dataset[key][:]
        else: 
            arr = self.hdf5_dataset[key][slice_idx]
        return arr

    def fix_slice_number(self, df: pd.DataFrame):

        num_slices = min(self.cfg.data.num_slices, df.shape[0])
        if self.cfg.data.sample_strategy == "random":
            slice_idx = np.random.choice(
                np.arange(df.shape[0]), replace=False, size=num_slices
            )
            slice_idx = list(np.sort(slice_idx))
            df = df.iloc[slice_idx, :]
        elif self.cfg.data.sample_strategy == "fix":
            df = df.iloc[:num_slices, :]
        else:
            raise Exception("Sampling strategy either 'random' or 'fix'")
        return df

    def fix_series_slice_number(self, series):

        num_slices = min(self.cfg.data.num_slices, series.shape[0])
        if self.cfg.data.sample_strategy == "random":
            slice_idx = np.random.choice(
                np.arange(series.shape[0]), replace=False, size=num_slices
            )
            slice_idx = list(np.sort(slice_idx))
            series = series[slice_idx, :]
        elif self.cfg.data.sample_strategy == "fix":
            series = series[:num_slices, :]
        else:
            raise Exception("Sampling strategy either 'random' or 'fix'")
        return series

    def fill_series_to_num_slicess(self, series, num_slices):
        x = torch.zeros(()).new_full((num_slices, *series.shape[1:]), 0.0)
        x[: series.shape[0]] = series
        return x

    def augment_series(self, series: np.array):
        """Series level augmentation"""

        if len(series.shape) == 3:
            series = np.expand_dims(series, 1)

        # crop volume slice-wise
        if self.cfg.transforms.RandomCrop is not None:
        # if not OmegaConf.is_none(self.cfg.transforms, "RandomCrop"):

            h = self.cfg.transforms.RandomCrop.height
            w = self.cfg.transforms.RandomCrop.width

            row_margin = max(0, series.shape[-2] - h)
            col_margin = max(0, series.shape[-1] - w)

            # Random crop during training, center crop during test inference
            row = (
                random.randint(0, row_margin)
                if self.split == "train"
                else row_margin // 2
            )
            col = (
                random.randint(0, col_margin)
                if self.split == "train"
                else col_margin // 2
            )
            series = series[:, :, col : col + h, row : row + w]

        # rotate
        if (self.cfg.transforms.Rotate is not None) and (
        # if (not OmegaConf.is_none(self.cfg.transforms, "Rotate")) and (
            self.split == "train"
        ):
            rotate_limit = self.cfg.transforms.Rotate.rotate_limit
            angle = random.randint(-rotate_limit, rotate_limit)

            series = rotate(series, angle, (-2, -1), reshape=False, cval=AIR_HU_VAL)

        return series
