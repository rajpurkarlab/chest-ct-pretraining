import os
import cv2
import torch
import numpy as np
import pandas as pd
import h5py
import random

from ..constants import *
from .dataset_base import PEDatasetBase
from omegaconf import OmegaConf


class PEDataset1D(PEDatasetBase):
    def __init__(self, cfg, split="train", transform=None):
        super().__init__(cfg, split)

        self.cfg = cfg
        self.df = pd.read_csv(RSNA_TRAIN_CSV)

        if split != "all":
            self.df = self.df[self.df[RSNA_SPLIT_COL] == split]
            #self.df = self.df[self.df[RSNA_INSTITUTION_SPLIT_COL] == split]

        # hdf5 path
        self.hdf5_path = self.cfg.data.hdf5_path
        if self.hdf5_path is None:
            raise Exception("Encoded slice HDF5 required")

        with h5py.File(self.hdf5_path, 'r') as f:
            study = list(f.keys())
            self.df = self.df[self.df[RSNA_STUDY_COL].isin(study)]

        if split == 'train':
            if not OmegaConf.is_none(cfg.data, 'sample_frac'):
                num_study = len(study) 
                num_sample = int(num_study * cfg.data.sample_frac)
                sampled_study = np.random.choice(study, num_sample, replace=False)
                self.df = self.df[self.df[RSNA_STUDY_COL].isin(sampled_study)]
            
        # get studies
        grouped_df = self.df.groupby(RSNA_STUDY_COL).head(1)[
            [RSNA_PE_TARGET_COL, RSNA_STUDY_COL]
        ]

        # use positive as 1
        self.pe_labels = [1 if t == 0 else 0 for t in grouped_df[RSNA_PE_TARGET_COL].tolist()]
        self.study = grouped_df[RSNA_STUDY_COL].tolist()


    def __getitem__(self, index):

        # read featurized series 
        study = self.study[index]
        x = self.read_from_hdf5(study, hdf5_path=self.hdf5_path)

        # fix number of slices
        x = self.fix_series_slice_number(x)

        # contextualize slices
        if self.cfg.data.contextualize_slice:
            x = self.contextualize_slice(x)

        # create torch tensor
        x = torch.from_numpy(x).float()

        # fill 
        x = self.fill_series_to_num_slicess(x, self.cfg.data.num_slices)
 
        # get traget
        y = self.pe_labels[index]
        y = torch.tensor(y).float().unsqueeze(-1)

        return x, y, study 

    def __len__(self):
        return len(self.study)

    def contextualize_slice(self, arr):

        # make new empty array
        new_arr = np.zeros((arr.shape[0], arr.shape[1] * 3), dtype=np.float32)

        # fill first third of new array with original features
        for i in range(len(arr)):
            new_arr[i, : arr.shape[1]] = arr[i]

        # difference between previous neighbor
        new_arr[1:, arr.shape[1] : arr.shape[1] * 2] = (
            new_arr[1:, : arr.shape[1]] - new_arr[:-1, : arr.shape[1]]
        )

        # difference between next neighbor
        new_arr[:-1, arr.shape[1] * 2 :] = (
            new_arr[:-1, : arr.shape[1]] - new_arr[1:, : arr.shape[1]]
        )

        return new_arr

    def get_sampler(self):

        neg_class_count = (np.array(self.pe_labels) == 0).sum()
        pos_class_count = (np.array(self.pe_labels) == 1).sum()
        class_weight = [1 / neg_class_count, 1 / pos_class_count]
        weights = [class_weight[i] for i in self.pe_labels]
 
        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

        return sampler


class LIDCDataset1D(PEDatasetBase):
    def __init__(self, cfg, split="train", transform=None):
        super().__init__(cfg, split)

        self.cfg = cfg
        self.df = pd.read_csv(LIDC_TRAIN_CSV)

        if split != "all":
            self.df = self.df[self.df[LIDC_SPLIT_COL] == split]

        # hdf5 path
        self.hdf5_path = self.cfg.data.hdf5_path
        if self.hdf5_path is None:
            raise Exception("Encoded slice HDF5 required")

        # TODO: 
        with h5py.File(self.hdf5_path, 'r') as f:
            study = list(f.keys())
            self.df = self.df[self.df[LIDC_STUDY_COL].isin(study)]

        if split == 'train':
            if not OmegaConf.is_none(cfg.data, 'sample_frac'):
                num_study = len(study) 
                num_sample = int(num_study * cfg.data.sample_frac)
                sampled_study = np.random.choice(study, num_sample, replace=False)
                self.df = self.df[self.df[LIDC_STUDY_COL].isin(sampled_study)]
        
        # get studies
        grouped_df = self.df.groupby(LIDC_STUDY_COL)[LIDC_NOD_SLICE_COL].max()
        grouped_df = grouped_df.reset_index()
        grouped_df = grouped_df.rename({
            LIDC_NOD_SLICE_COL: "nodule_present_in_study"
        }, axis=1)

        # use positive as 1
        self.pe_labels = grouped_df["nodule_present_in_study"].tolist()
        self.study = grouped_df[LIDC_STUDY_COL].tolist()
        print(len(self.study))


    def __getitem__(self, index):

        # read featurized series 
        study = self.study[index]
        x = self.read_from_hdf5(study, hdf5_path=self.hdf5_path)

        # fix number of slices
        x = self.fix_series_slice_number(x)

        # contextualize slices
        if self.cfg.data.contextualize_slice:
            x = self.contextualize_slice(x)

        # create torch tensor
        x = torch.from_numpy(x).float()

        # fill 
        x = self.fill_series_to_num_slicess(x, self.cfg.data.num_slices)
 
        # get traget
        y = self.pe_labels[index]
        y = torch.tensor(y).float().unsqueeze(-1)

        return x, y, study 

    def __len__(self):
        return len(self.study)

    def contextualize_slice(self, arr):

        # make new empty array
        new_arr = np.zeros((arr.shape[0], arr.shape[1] * 3), dtype=np.float32)

        # fill first third of new array with original features
        for i in range(len(arr)):
            new_arr[i, : arr.shape[1]] = arr[i]

        # difference between previous neighbor
        new_arr[1:, arr.shape[1] : arr.shape[1] * 2] = (
            new_arr[1:, : arr.shape[1]] - new_arr[:-1, : arr.shape[1]]
        )

        # difference between next neighbor
        new_arr[:-1, arr.shape[1] * 2 :] = (
            new_arr[:-1, : arr.shape[1]] - new_arr[1:, : arr.shape[1]]
        )

        return new_arr

    def get_sampler(self):

        neg_class_count = (np.array(self.pe_labels) == 0).sum()
        pos_class_count = (np.array(self.pe_labels) == 1).sum()
        class_weight = [1 / neg_class_count, 1 / pos_class_count]
        weights = [class_weight[i] for i in self.pe_labels]
 
        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

        return sampler



class PEDataset1DStanford(PEDatasetBase):
    def __init__(self, cfg, split="train", transform=None):
        super().__init__(cfg, split)

        self.cfg = cfg
        self.df = pd.read_csv(STANFORD_NO_RSNA_CSV)

        if split != "all":
            self.df = self.df[self.df["Split"] == split]

        # hdf5 path
        self.hdf5_path = self.cfg.data.hdf5_path
        if self.hdf5_path is None:
            raise Exception("Encoded slice HDF5 required")

        # TODO: 
        with h5py.File(self.hdf5_path, 'r') as f:
            study = list(f.keys())
            self.df = self.df[self.df['StudyInstanceUID'].isin(study)]

        if split == 'train':
            if not OmegaConf.is_none(cfg.data, 'sample_frac'):
                num_study = len(study) 
                num_sample = int(num_study * cfg.data.sample_frac)
                sampled_study = np.random.choice(study, num_sample, replace=False)
                self.df = self.df[self.df['StudyInstanceUID'].isin(sampled_study)]

        # get studies
        grouped_df = self.df.groupby('StudyInstanceUID').head(1)[
            ['negative_exam_for_pe', 'StudyInstanceUID']
        ]

        # use positive as 1
        self.pe_labels = grouped_df['negative_exam_for_pe'].tolist()
        self.study = grouped_df['StudyInstanceUID'].tolist()

    def __getitem__(self, index):

        # read featurized series 
        study = self.study[index]
        x = self.read_from_hdf5(study, hdf5_path=self.hdf5_path)

        # fix number of slices
        x = self.fix_series_slice_number(x)

        # contextualize slices
        if self.cfg.data.contextualize_slice:
            x = self.contextualize_slice(x)

        # create torch tensor
        x = torch.from_numpy(x).float()

        # fill 
        x = self.fill_series_to_num_slicess(x, self.cfg.data.num_slices)
 
        # get traget
        y = self.pe_labels[index]
        y = torch.tensor(y).float().unsqueeze(-1)

        return x, y, study 

    def __len__(self):
        return len(self.study)

    def contextualize_slice(self, arr):

        # make new empty array
        new_arr = np.zeros((arr.shape[0], arr.shape[1] * 3), dtype=np.float32)

        # fill first third of new array with original features
        for i in range(len(arr)):
            new_arr[i, : arr.shape[1]] = arr[i]

        # difference between previous neighbor
        new_arr[1:, arr.shape[1] : arr.shape[1] * 2] = (
            new_arr[1:, : arr.shape[1]] - new_arr[:-1, : arr.shape[1]]
        )

        # difference between next neighbor
        new_arr[:-1, arr.shape[1] * 2 :] = (
            new_arr[:-1, : arr.shape[1]] - new_arr[1:, : arr.shape[1]]
        )

        return new_arr

    def get_sampler(self):

        neg_class_count = (np.array(self.pe_labels) == 0).sum()
        pos_class_count = (np.array(self.pe_labels) == 1).sum()
        class_weight = [1 / neg_class_count, 1 / pos_class_count]
        weights = [class_weight[i] for i in self.pe_labels]
 
        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

        return sampler
