import torch
import numpy as np
import pandas as pd
import cv2

from ..constants import *
from .dataset_base import PEDatasetBase
from omegaconf import OmegaConf
from PIL import Image

class PEDataset2D(PEDatasetBase):
    def __init__(self, cfg, split="train", transform=None, study_name=None, instance_name=None):
        super().__init__(cfg, split)

        self.df = pd.read_csv(RSNA_TRAIN_CSV)
        self.transform = transform

        if study_name is not None: 
            self.df = self.df[self.df[RSNA_STUDY_COL] == study_name] 
        elif instance_name is not None: 
            self.df = self.df[self.df[RSNA_INSTANCE_COL].isin(instance_name)] 
        else: 
            if self.split != "all":
                self.df = self.df[self.df[RSNA_SPLIT_COL] == self.split]

            if self.split == "train":
                if self.cfg.data.positive_only:
                    print('\n using positive only')
                    self.df = self.df[self.df[RSNA_PE_TARGET_COL] == 0]
                if not OmegaConf.is_none(cfg.data, 'sample_frac'):
                    study = list(self.df[RSNA_STUDY_COL].unique())
                    num_study = len(study)
                    num_sample = int(num_study * cfg.data.sample_frac)
                    sampled_study = np.random.choice(study, num_sample, replace=False)
                    self.df = self.df[self.df[RSNA_STUDY_COL].isin(sampled_study)]
                    #self.df = self.df.sample(frac=cfg.data.sample_frac)
        print('='*80)
        print(f'Weighted sample: {cfg.data.weighted_sample}')
        print(f'Positive only: {cfg.data.positive_only}')
        print('='*80)

    def __getitem__(self, index):

        # get slice row
        instance_info = self.df.iloc[index]

        if self.cfg.data.use_hdf5: 
            study_name = instance_info[RSNA_STUDY_COL]
            slice_idx = instance_info[RSNA_INSTANCE_ORDER_COL]
            ct_slice = self.read_from_hdf5(study_name, slice_idx=slice_idx)

            ct_slice = self.windowing(ct_slice, 400, 1000)
        else: 
            ct_slice = self.process_slice(instance_info)

        # transform
        if ct_slice.shape[0] == 3:
            ct_slice = np.transpose(ct_slice, (1,2,0))
        else: 
            ct_slice = np.expand_dims(ct_slice, -1)

        if self.transform is not None:
            x = self.transform(image=ct_slice)["image"]
        else:
            x = torch.Tensor(ct_slice)
            x = x.permute(2,0,1)

        # check dimention
        if x.shape[0] == 1:  # for repeat
            c, w, h = list(x.shape)
            x = x.expand(3, w, h) 
        x = x.type(torch.FloatTensor)

        # get labels
        targets = RSNA_TARGET_TYPES[self.cfg.data.targets]
        y = instance_info[targets].astype(float)
        if y[RSNA_PE_SLICE_COL] == 0:  # set all labels to 0 if pe not on slice
            y.replace(1.0, 0.0)
        y = torch.tensor(y)

        # get series id
        instance_id = instance_info[RSNA_INSTANCE_COL]
        study_id = instance_info[RSNA_STUDY_COL]

        return x, y, f'{instance_id}-{study_id}'

    def __len__(self):
        return len(self.df)

    def get_sampler(self):

        neg_class_count = (self.df[RSNA_PE_SLICE_COL] == 0).sum().item()
        pos_class_count = (self.df[RSNA_PE_SLICE_COL] == 1).sum().item()
        class_weight = [1 / neg_class_count, 1 / pos_class_count]
        weights = [class_weight[i] for i in self.df[RSNA_PE_SLICE_COL]]

        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        return sampler


class LIDCDataset2D(PEDatasetBase):
    def __init__(self, cfg, split="train", transform=None, study_name=None):
        super().__init__(cfg, split)

        self.df = pd.read_csv(LIDC_TRAIN_CSV)
        self.transform = transform

        if study_name is not None: 
            self.df = self.df[self.df[LIDC_STUDY_COL] == study_name] 
        else: 
            if self.split != "all":
                self.df = self.df[self.df[LIDC_SPLIT_COL] == self.split]

            if self.split == "train":
                if self.cfg.data.positive_only:
                    raise NotImplementedError
                if not OmegaConf.is_none(cfg.data, 'sample_frac'):
                    study = list(self.df[LIDC_STUDY_COL].unique())
                    num_study = len(study)
                    num_sample = int(num_study * cfg.data.sample_frac)
                    sampled_study = np.random.choice(study, num_sample, replace=False)
                    self.df = self.df[self.df[LIDC_STUDY_COL].isin(sampled_study)]
                    #self.df = self.df.sample(frac=cfg.data.sample_frac)

        print('='*80)
        print(f'Weighted sample: {cfg.data.weighted_sample}')
        print(f'Positive only: {cfg.data.positive_only}')
        print('='*80)

    def __getitem__(self, index):

        # get slice row
        instance_info = self.df.iloc[index]

        if self.cfg.data.use_hdf5: 
            study_name = instance_info[LIDC_STUDY_COL]
            slice_idx = instance_info[LIDC_INSTANCE_ORDER_COL]
            ct_slice = self.read_from_hdf5(study_name, slice_idx=slice_idx, hdf5_path=LIDC_STUDY_HDF5)

            ct_slice = self.windowing(ct_slice, -600, 1500)
        else: 
            raise NotImplementedError

        # transform
        if ct_slice.shape[0] == 3:
            ct_slice = np.transpose(ct_slice, (1,2,0))
        else: 
            ct_slice = np.expand_dims(ct_slice, -1)

        if self.transform is not None:
            x = self.transform(image=ct_slice)["image"]

        else:
            x = torch.Tensor(ct_slice)
            x = x.permute(2,0,1)

        # TODO fix style 
        if not self.cfg.transforms.type == 'dino':
            # check dimention
            if x.shape[0] == 1:  # for repeat
                c, w, h = list(x.shape)
                x = x.expand(3, w, h) 
            x = x.type(torch.FloatTensor)

        # get labels
        y = torch.tensor(instance_info[LIDC_NOD_SLICE_COL])
        y = y.unsqueeze(-1).float()

        # get series id
        instance_id = instance_info[LIDC_INSTANCE_COL]
        study_id = instance_info[LIDC_STUDY_COL]

        return x, y, f'{instance_id}-{study_id}'

    def __len__(self):
        return len(self.df)

    def get_sampler(self):

        neg_class_count = (self.df[LIDC_NOD_SLICE_COL] == 0).sum().item()
        pos_class_count = (self.df[LIDC_NOD_SLICE_COL] == 1).sum().item()
        class_weight = [1 / neg_class_count, 1 / pos_class_count]
        weights = [class_weight[i] for i in self.df[LIDC_NOD_SLICE_COL]]

        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        return sampler


class PEDataset2DMAE(PEDataset2D):
    def __init__(self, cfg, split="train", transform=None, study_name=None):
        super().__init__(cfg, split, transform, study_name)

    def __getitem__(self, index):

        # get slice row
        instance_info = self.df.iloc[index]

        if self.cfg.data.use_hdf5: 
            study_name = instance_info[RSNA_STUDY_COL]
            slice_idx = instance_info[RSNA_INSTANCE_ORDER_COL]
            ct_slice = self.read_from_hdf5(study_name, slice_idx=slice_idx)

            ct_slice = self.windowing(ct_slice, 400, 1000)
        else: 
            ct_slice = self.process_slice(instance_info)

        # transform
        if ct_slice.shape[0] == 3:
            ct_slice = np.transpose(ct_slice, (1,2,0))
        else: 
            ct_slice = np.expand_dims(ct_slice, -1)

        if ct_slice.max() <= 1.0: 
            ct_slice *= 255
        ct_slice = Image.fromarray(np.uint8(ct_slice[:,:,0])).convert('RGB')        
        x, mask = self.transform(ct_slice)

        x = x.type(torch.FloatTensor)
        mask = mask.astype(int)

        # get series id
        instance_id = instance_info[RSNA_INSTANCE_COL]
        study_id = instance_info[RSNA_STUDY_COL]

        return x, mask, f'{instance_id}-{study_id}'

    def __len__(self):
        return len(self.df)



class PEDataset2DStanford(PEDatasetBase):
    def __init__(self, cfg, split="train", transform=None, study_name=None):
        super().__init__(cfg, split)

        self.df = pd.read_csv(STANFORD_NO_RSNA_CSV)
        self.transform = transform

        if self.split != "all":
            self.df = self.df[self.df['Split'] == self.split]

        if self.split == "train":
            if self.cfg.data.positive_only:
                self.df = self.df[self.df['negative_exam_for_pe'] == 0]
            if not OmegaConf.is_none(cfg.data, 'sample_frac'):
                study = list(self.df['StudyInstanceUID'].unique())
                num_study = len(study)
                num_sample = int(num_study * cfg.data.sample_frac)
                sampled_study = np.random.choice(study, num_sample, replace=False)
                self.df = self.df[self.df['StudyInstanceUID'].isin(sampled_study)]
                #self.df = self.df.sample(frac=cfg.data.sample_frac)

        print('='*80)
        print(f'Weighted sample: {cfg.data.weighted_sample}')
        print(f'Positive only: {cfg.data.positive_only}')
        print('='*80)

    def __getitem__(self, index):

        # get slice row
        instance_info = self.df.iloc[index]

        if self.cfg.data.use_hdf5: 
            study_name = instance_info['StudyInstanceUID']
            slice_idx = instance_info['SliceOrder']
            ct_slice = self.read_from_hdf5(study_name, slice_idx=slice_idx, hdf5_path=STANFORD_NO_RSNA_HDF5)

            ct_slice = self.windowing(ct_slice, 400, 1000)
        else: 
            ct_slice = self.process_slice(instance_info)

        # transform
        if ct_slice.shape[0] == 3:
            ct_slice = np.transpose(ct_slice, (1,2,0))
        else: 
            ct_slice = np.expand_dims(ct_slice, -1)

        if self.transform is not None:
            x = self.transform(image=ct_slice)["image"]

        else:
            x = torch.Tensor(ct_slice)
            x = x.permute(2,0,1)

        # check dimention
        if x.shape[0] == 1:  # for repeat
            c, w, h = list(x.shape)
            x = x.expand(3, w, h) 
        x = x.type(torch.FloatTensor)

        # get labels
        y = instance_info['pe_present_on_image'].astype(float)
        y = torch.tensor([y])

        # get series id
        instance_id = instance_info['SOPInstanceUID']
        study_id = instance_info['StudyInstanceUID']

        return x, y, f'{instance_id}*{study_id}'

    def __len__(self):
        return len(self.df)

    def get_sampler(self):

        neg_class_count = (self.df['pe_present_on_image'] == 0).sum().item()
        pos_class_count = (self.df['pe_present_on_image'] == 1).sum().item()
        class_weight = [1 / neg_class_count, 1 / pos_class_count]
        weights = [class_weight[i] for i in self.df['pe_present_on_image']]

        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        return sampler


class DemoDataset2D(PEDatasetBase):
    def __init__(self, cfg, split="train", transform=None): 
        super().__init__(cfg, split)

        self.df = pd.read_csv('./data/demo/demo.csv')
        self.transform = transform

        print('='*80)
        print(f'Weighted sample: {cfg.data.weighted_sample}')
        print(f'Positive only: {cfg.data.positive_only}')
        print('='*80)

    def __getitem__(self, index):

        # get slice row
        instance_info = self.df.iloc[index]

        ct_slice = self.process_slice(instance_info, './data/demo')

        # transform
        if ct_slice.shape[0] == 3:
            ct_slice = np.transpose(ct_slice, (1,2,0))
        else: 
            ct_slice = np.expand_dims(ct_slice, -1)

        if self.transform is not None:
            x = self.transform(image=ct_slice)["image"]
        else:
            x = torch.Tensor(ct_slice)
            x = x.permute(2,0,1)

        # check dimention
        if x.shape[0] == 1:  # for repeat
            c, w, h = list(x.shape)
            x = x.expand(3, w, h) 
        x = x.type(torch.FloatTensor)

        # get labels
        targets = RSNA_TARGET_TYPES[self.cfg.data.targets]
        y = instance_info[targets].astype(float).item()
        y = torch.tensor([y])

        # get series id
        instance_id = instance_info[RSNA_INSTANCE_COL]
        study_id = instance_info[RSNA_STUDY_COL]

        return x, y, f'{instance_id}-{study_id}'

    def __len__(self):
        return len(self.df)

    def get_sampler(self):

        neg_class_count = (self.df[RSNA_PE_SLICE_COL] == 0).sum().item()
        pos_class_count = (self.df[RSNA_PE_SLICE_COL] == 1).sum().item()
        class_weight = [1 / neg_class_count, 1 / pos_class_count]
        weights = [class_weight[i] for i in self.df[RSNA_PE_SLICE_COL]]

        weights = torch.Tensor(weights).double()
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        return sampler


