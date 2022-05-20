import pandas as pd
import mdai
import pydicom
import sys
import os
import glob
import tqdm
import numpy as np
import h5py

sys.path.append(os.getcwd())

from collections import defaultdict
from pe_models.constants import *
from pe_models import utils
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


def process_window_df(df:pd.DataFrame, num_slices:int=24, min_abnormal_slice:int=4, stride:int=6):

    window_labels_csv = STANFORD_DATA_DIR / f"stanford_window_{num_slices}_min_abnormal_{min_abnormal_slice}_stride_{stride}.csv"
    print(window_labels_csv)

    # count number of windows per slice
    #count_num_windows = lambda x: x // num_slices\
    #    + (1 if x % num_slices > 0 else 0)
    count_num_windows = lambda x: (x - num_slices) // stride
    df['NumWindows'] = df['NumSlices'].apply(
        count_num_windows)

    # get windows list
    df_study = df.groupby(['StudyInstanceUID']).head(1)
    window_labels = defaultdict(list)

    # studies
    for _, row in tqdm.tqdm(df_study.iterrows(), total=df_study.shape[0]):
        study_name = row['StudyInstanceUID']
        split = row['Split']

        study_df = df[df['StudyInstanceUID'] == study_name]
        study_df = study_df.sort_values("ImagePositionPatient_2")

        # windows
        for idx in range(row['NumWindows']):
            #start_idx = idx * num_slices
            #end_idx = (idx+1) * num_slice
            start_idx = idx * stride
            end_idx = (idx * stride) +  num_slices

            window_df = study_df.iloc[start_idx: end_idx]
            num_positives_slices = window_df['pe_present_on_image'].sum()
            label = 1 if num_positives_slices >= min_abnormal_slice else 0
            window_labels['StudyInstanceUID'].append(study_name)
            window_labels['index'].append(idx)
            window_labels['Label'].append(label)
            window_labels['Split'].append(split)
            window_labels['InstancePath'].append(window_df['InstancePath'].tolist())
            window_labels["ImagePositionPatient_2"].append(window_df["ImagePositionPatient_2"].tolist())
            window_labels['pe_present_on_image'].append(window_df['pe_present_on_image'].tolist())
            window_labels['SOPInstanceUID'].append(window_df['SOPInstanceUID'].tolist())
            window_labels['SliceOrder'].append(window_df['SliceOrder'].tolist())
            window_labels['num_positive_slices'].append(num_positives_slices)
    df = pd.DataFrame.from_dict(window_labels)
    df.to_csv(window_labels_csv)

    return df

