################################################################################
# TODO: only binary prediction of window level pe 
################################################################################
import numpy as np
import pandas as pd
import h5py
from pandas.core import series
import torch
import tqdm

from pe_models import utils
from ast import literal_eval
from collections import defaultdict
from ..constants import *
from .. import preprocess
from .dataset_base import PEDatasetBase

from timeit import default_timer as timer
from omegaconf import OmegaConf


class PEDataset3D(PEDatasetBase):
    def __init__(self, cfg, split="train", transform=None):
        super().__init__(cfg, split)

        self.df = pd.read_csv(RSNA_TRAIN_CSV)
        if split != "all":
            self.df = self.df[self.df[RSNA_SPLIT_COL] == split]
        self.all_series = list(self.df[RSNA_SERIES_COL].unique())

    def __getitem__(self, index):

        # get series name
        series_name = self.all_series[index]
        series_df = self.df[self.df[RSNA_SERIES_COL] == series_name].copy()
        series_df = series_df.sort_values("ImagePositionPatient_2")
        series_df = self.fix_slice_number(series_df)

        if self.cfg.data.use_hdf5: 
            study_name = series_df.iloc[0][RSNA_STUDY_COL]
            slice_idx = sorted(series_df[RSNA_INSTANCE_ORDER_COL])
            x = self.read_from_hdf5(study_name, slice_idx=slice_idx)

            # TODO: 3D dataset only support repeat channel right now
            x = self.windowing(x, 400, 1000)
            x = np.expand_dims(x, axis=1)
        else:
            x = np.stack([self.process_slice(row) for idx, row in series_df.iterrows()])
        
        x = self.augment_series(x)  # TODO: check placement & check if ablemutation support 3D
        x = torch.from_numpy(x).float()

        # if number of slice < max sampled slice
        if x.shape[0] < self.cfg.data.num_slices:
            x = self.fill_series_to_num_slicess(x, self.cfg.data.num_slices)
        x = np.transpose(x, (1, 0, 2, 3))

        # check dimention
        if x.shape[0] == 1:
            c, b, w, h = list(x.shape)
            x = x.expand(3, b, w, h)
        x = x.type(torch.FloatTensor)

        # get labels
        targets = RSNA_TARGET_TYPES[self.cfg.data.targets]
        y = series_df.iloc[0][targets].astype(float)
        y = torch.tensor(y)

        return x, y, series_name

    def __len__(self):
        return len(self.all_series)

    def get_sampler(self):

        df_study = self.df.groupby(RSNA_STUDY_COL).head(1)

        neg_class_count = (df_study[RSNA_PE_TARGET_COL] == 1).sum().item()
        pos_class_count = (df_study[RSNA_PE_TARGET_COL] == 0).sum().item()
        class_weight = [1 / neg_class_count, 1 / pos_class_count]
        weights = [class_weight[i] for i in df_study[RSNA_PE_TARGET_COL]]

        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

        return sampler


class PEDatasetWindow(PEDatasetBase):

    def __init__(self, cfg, split="train", transform=None):
        super().__init__(cfg, split)

        self.df_full = pd.read_csv(RSNA_TRAIN_CSV)
        self.cfg = cfg

        if OmegaConf.is_none(cfg.data, 'stride'):
            window_labels_csv = RSNA_DATA_DIR / f"rsna_window_{self.cfg.data.num_slices}_min_abnormal_{self.cfg.data.min_abnormal_slice}.csv"
        else: 
            window_labels_csv = RSNA_DATA_DIR / f"rsna_window_{self.cfg.data.num_slices}_min_abnormal_{self.cfg.data.min_abnormal_slice}_stride_{self.cfg.data.stride}.csv"
        if window_labels_csv.is_file(): 
            # literal_eval to convert str to list
            self.df = pd.read_csv(window_labels_csv, converters={
                RSNA_INSTANCE_PATH_COL: literal_eval,
                RSNA_PE_SLICE_COL: literal_eval, 
                RSNA_INSTANCE_COL: literal_eval, 
                RSNA_INSTANCE_ORDER_COL: literal_eval
            })
        else:
            print('='*80)
            print('\nCreating windowed dataset\n')
            print('-'*80)
            
            # defaults
            if OmegaConf.is_none(self.cfg.data, 'stride'):
                self.cfg.data.stride = 24
            if OmegaConf.is_none(self.cfg.data, 'min_abnormal_slice'):
                self.cfg.min_abnormal_slice = 4
            if OmegaConf.is_none(self.cfg.data, 'num_slices'):
                self.cfg.num_slices = 12 

            self.df = preprocess.rsna.process_window_df(
                self.df_full, self.cfg.data.num_slices, self.cfg.data.min_abnormal_slice, self.cfg.data.stride)

        if 'num_positive_slices' not in self.df.columns: 
            self.df['num_positive_slices'] = self.df['pe_present_on_image'].apply(lambda x: sum(x))


        if split != "all":
            self.df = self.df[self.df[RSNA_SPLIT_COL] == split]

        if cfg.data.sample_frac is not None:
            print('='*80)
            print('sampling stuff')
            print('='*80)
            all_studies = self.df[RSNA_STUDY_COL].tolist()
            num_studies = len(all_studies) 
            num_sample = int(num_studies * cfg.data.sample_frac)
            sampled_studies = all_studies[:num_sample]
            self.df = self.df[self.df[RSNA_STUDY_COL].isin(sampled_studies)]

        """
        if not OmegaConf.is_none(cfg.data, 'min_positive_slices'):
            #self.df['label'] = self.df['num_positive_slices'].apply(
            #    lambda x: 1 if x >= cfg.data.min_positive_slices else 0
            #)

            if 'num_positive_slices' not in self.df.columns: 
                self.df['num_positive_slices'] = self.df['pe_present_on_image'].apply(lambda x: sum(x))
            self.df = self.df[
                (self.df.num_positive_slices == 0) | 
                (self.df.num_positive_slices >= cfg.data.min_positive_slices)
            ]
        """

    def __getitem__(self, index):

        window = self.df.iloc[index]

        # read from hdf5 
        if self.cfg.data.use_hdf5: 
            study_name = window[RSNA_STUDY_COL]
            slice_idx = sorted(window[RSNA_INSTANCE_ORDER_COL])
            series = self.read_from_hdf5(study_name, slice_idx=slice_idx)

            # TODO: 3D dataset only support repeat channel right now
            series = self.windowing(series, 400, 1000)
        # read from raw datad
        else:    
            series = np.stack(
                [self.process_slice(pd.Series({RSNA_INSTANCE_PATH_COL: path})) for \
                    path in window[RSNA_INSTANCE_PATH_COL]]
            )

        series = self.augment_series(series)  # TODO: check placement & check if ablemutation support 3D

        # TODO: 
        series = series - 0.15897

        x = torch.from_numpy(series).float()
        if x.shape[0] < self.cfg.data.num_slices:
            x = self.fill_series_to_num_slicess(x, self.cfg.data.num_slices)
        x = torch.permute(x, (1, 0, 2, 3))
        
        # check dimention
        if x.shape[0] == 1:
            x = x.squeeze()
            x = x.expand(3, *list(x.shape))
        x = x.type(torch.FloatTensor)
        
        # get labels
        y = torch.tensor(window['label'])
        y = y.unsqueeze(-1).float()

        return x, y, (window[RSNA_STUDY_COL], (window['index']))

    def __len__(self):
        return len(self.df)

    def get_sampler(self):

        neg_class_count = (self.df['label'] == 0).sum().item()
        pos_class_count = (self.df['label'] == 1).sum().item()
        class_weight = [(1 / neg_class_count) * 0.7 , (1 / pos_class_count) * 0.3]
        #class_weight = [0.7, 0.3]
        weights = [class_weight[i] for i in self.df['label']]

        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

        return sampler


class PEDatasetWindowStanford(PEDatasetBase):

    def __init__(self, cfg, split="train", transform=None):
        super().__init__(cfg, split)

        self.df_full = pd.read_csv(STANFORD_NO_RSNA_CSV)

        window_labels_csv = STANFORD_DATA_DIR / f"stanford_window_{self.cfg.data.num_slices}_min_abnormal_{self.cfg.data.min_abnormal_slice}_stride_{self.cfg.data.stride}.csv"
        print(window_labels_csv)
        if window_labels_csv.is_file(): 
            # literal_eval to convert str to list
            self.df = pd.read_csv(window_labels_csv, converters={
                RSNA_INSTANCE_PATH_COL: literal_eval,
                RSNA_PE_SLICE_COL: literal_eval, 
                RSNA_INSTANCE_COL: literal_eval, 
                RSNA_INSTANCE_ORDER_COL: literal_eval, 
                RSNA_SLICE_ORDER_COL: literal_eval
            })
        else:
            print('='*80)
            print('\nCreating windowed dataset\n')
            print('-'*80)

            # defaults
            if OmegaConf.is_none(self.cfg.data, 'stride'):
                self.cfg.data.stride = 24
            if OmegaConf.is_none(self.cfg.data, 'min_abnormal_slice'):
                self.cfg.min_abnormal_slice = 4
            if OmegaConf.is_none(self.cfg.data, 'num_slices'):
                self.cfg.num_slices = 12 
            self.df = preprocess.stanford.process_window_df(
                self.df_full, self.cfg.data.num_slices, self.cfg.data.min_abnormal_slice, self.cfg.data.stride)


        if split != "all":
            self.df = self.df[self.df['Split'] == split]

        if not OmegaConf.is_none(cfg.data, 'min_positive_slices'):
            #self.df['label'] = self.df['num_positive_slices'].apply(
            #    lambda x: 1 if x >= cfg.data.min_positive_slices else 0
            #)
            self.df = self.df[
                (self.df.num_positive_slices == 0) | 
                (self.df.num_positive_slices >= cfg.data.min_positive_slices)
            ]

    def __getitem__(self, index):

        window = self.df.iloc[index]

        # read from hdf5 
        if self.cfg.data.use_hdf5: 
            study_name = window['StudyInstanceUID']
            slice_idx = sorted(window['SliceOrder'])
            series = self.read_from_hdf5(study_name, slice_idx=slice_idx, hdf5_path=STANFORD_NO_RSNA_HDF5)

            # TODO: 3D dataset only support repeat channel right now
            series = self.windowing(series, 400, 1000)
        # read from raw datad
        else:    
            series = np.stack(
                [self.process_slice(pd.Series({'InstancePath': path})) for \
                    path in window['InstancePath']]
            )

        series = self.augment_series(series)  # TODO: check placement & check if ablemutation support 3D

        # TODO: 
        series = series - 0.15897

        x = torch.from_numpy(series).float()
        if x.shape[0] < self.cfg.data.num_slices:
            x = self.fill_series_to_num_slicess(x, self.cfg.data.num_slices)
        x = torch.permute(x, (1, 0, 2, 3))
        
        # check dimention
        if x.shape[0] == 1:
            x = x.squeeze()
            x = x.expand(3, *list(x.shape))
        x = x.type(torch.FloatTensor)
        
        # get labels
        y = torch.tensor(window['Label'])
        y = y.unsqueeze(-1).float()

        return x, y, (window['StudyInstanceUID'], (window['index']))

    def __len__(self):
        return len(self.df)

    def get_sampler(self):

        neg_class_count = (self.df['Label'] == 0).sum().item()
        pos_class_count = (self.df['Label'] == 1).sum().item()
        class_weight = [(1 / neg_class_count) * 0.7 , (1 / pos_class_count) * 0.3]
        #class_weight = [0.7, 0.3]
        weights = [class_weight[i] for i in self.df['Label']]

        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

        return sampler

class LIDCDatasetWindow(PEDatasetBase):

    def __init__(self, cfg, split="train", transform=None):
        super().__init__(cfg, split)

        self.df_full = pd.read_csv(LIDC_TRAIN_CSV)

        window_labels_csv = LIDC_DATA_DIR / f"lidc_window_{self.cfg.data.num_slices}_min_abnormal_{self.cfg.data.min_abnormal_slice}.csv"
        if window_labels_csv.is_file(): 
            # literal_eval to convert str to list
            self.df = pd.read_csv(window_labels_csv, converters={
                LIDC_NOD_SLICE_COL: literal_eval, 
                LIDC_INSTANCE_COL: literal_eval, 
                LIDC_INSTANCE_ORDER_COL: literal_eval
            })
        else:
            print('='*80)
            print('\nCreating windowed dataset\n')
            print('-'*80)

            # defaults
            if OmegaConf.is_none(self.cfg.data, 'stride'):
                self.cfg.data.stride = 24
            if OmegaConf.is_none(self.cfg.data, 'min_abnormal_slice'):
                self.cfg.min_abnormal_slice = 4
            if OmegaConf.is_none(self.cfg.data, 'num_slices'):
                self.cfg.num_slices = 12 

            self.df = preprocess.lidc.process_window_df(
                self.df_full, self.cfg.data.num_slices, self.cfg.data.min_abnormal_slice)


        if split != "all":
            self.df = self.df[self.df[LIDC_SPLIT_COL] == split]

    def __getitem__(self, index):

        window = self.df.iloc[index]

        # read from hdf5 
        if self.cfg.data.use_hdf5: 
            study_name = window[LIDC_STUDY_COL]
            slice_idx = sorted(window[LIDC_INSTANCE_ORDER_COL])
            series = self.read_from_hdf5(study_name, slice_idx, hdf5_path=LIDC_STUDY_HDF5)

            # TODO: 3D dataset only support repeat channel right now
            series = self.windowing(series, -600, 1500)
        # read from raw datad
        else:
            raise NotImplementedError
            # series = np.stack(
            #     [self.process_slice(pd.Series({RSNA_INSTANCE_PATH_COL: path})) for \
            #         path in window[RSNA_INSTANCE_PATH_COL]]
            # )

        series = self.augment_series(series)  # TODO: check placement & check if ablemutation support 3D

        x = torch.from_numpy(series).float()
        if x.shape[0] < self.cfg.data.num_slices:
            x = self.fill_series_to_num_slicess(x, self.cfg.data.num_slices)
        x = torch.permute(x, (1, 0, 2, 3))
        
        # check dimention
        if x.shape[0] == 1:
            x = x.squeeze()
            x = x.expand(3, *list(x.shape))
        x = x.type(torch.FloatTensor)
        
        # get labels
        y = torch.tensor(window['label'])
        y = y.unsqueeze(-1).float()

        return x, y, (window[LIDC_STUDY_COL], (window['index']))

    def __len__(self):
        return len(self.df)

    def get_sampler(self):

        neg_class_count = (self.df['label'] == 0).sum().item()
        pos_class_count = (self.df['label'] == 1).sum().item()
        class_weight = [1 / neg_class_count, 1 / pos_class_count]
        weights = [class_weight[i] for i in self.df['label']]

        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )

        return sampler
