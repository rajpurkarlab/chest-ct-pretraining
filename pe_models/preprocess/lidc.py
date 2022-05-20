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

import cv2
import pylidc as pl


def process_window_df(df:pd.DataFrame, num_slices:int=24, min_abnormal_slice:int=4): 

    window_labels_csv = LIDC_DATA_DIR / f"lidc_window_{num_slices}_min_abnormal_{min_abnormal_slice}.csv"

    # count number of windows per slice
    count_num_windows = lambda x: x // num_slices\
        + (1 if x % num_slices > 0 else 0)
    df[LIDC_NUM_WINDOW_COL] = df[LIDC_NUM_SLICES_COL].apply(
        count_num_windows)
    
    # get windows list 
    df_study = df.groupby([LIDC_STUDY_COL]).head(1)
    window_labels = defaultdict(list)

    # studies
    for _, row in tqdm.tqdm(df_study.iterrows(), total=df_study.shape[0]):
        study_name = row[LIDC_STUDY_COL]
        split = row[LIDC_SPLIT_COL]
        study_df = df[df[LIDC_STUDY_COL] == study_name] 
        study_df = study_df.sort_values("ImagePositionPatient_2")

        # windows 
        for idx in range(row[LIDC_NUM_WINDOW_COL]):
            start_idx = idx * num_slices
            end_idx = (idx+1) * num_slices

            window_df = study_df.iloc[start_idx: end_idx]
            num_positives_slices = window_df[LIDC_NOD_SLICE_COL].sum()
            label = 1 if num_positives_slices >= min_abnormal_slice else 0
            window_labels[LIDC_STUDY_COL].append(study_name)
            window_labels['index'].append(idx)
            window_labels['label'].append(label)
            window_labels[LIDC_SPLIT_COL].append(split)
            window_labels["ImagePositionPatient_2"].append(window_df["ImagePositionPatient_2"].tolist())
            window_labels[LIDC_NOD_SLICE_COL].append(window_df[LIDC_NOD_SLICE_COL].tolist())
            window_labels[LIDC_INSTANCE_COL].append(window_df[LIDC_INSTANCE_COL].tolist())
            window_labels[LIDC_INSTANCE_ORDER_COL].append(window_df[LIDC_INSTANCE_ORDER_COL].tolist())
    df = pd.DataFrame.from_dict(window_labels) 
    df.to_csv(window_labels_csv)

    return df 


def process_dicom(dcm):
    pixel_array = dcm.pixel_array

    intercept = dcm.RescaleIntercept
    slope = dcm.RescaleSlope
    pixel_array = pixel_array * slope + intercept

    resize_shape = 256 # self.cfg.data.imsize
    pixel_array = cv2.resize(
        pixel_array, (resize_shape, resize_shape), interpolation=cv2.INTER_AREA
    )
    return pixel_array


def process_study_to_hdf5():
    records = []
    hdf5_fh = h5py.File(LIDC_STUDY_HDF5, "a")
    for scan in tqdm.tqdm(pl.query(pl.Scan).all()):
        dicoms = scan.load_all_dicom_images(verbose=False)
        if len(dicoms) == 0:
            continue
        series = np.stack([process_dicom(dcm) for dcm in dicoms])
        hdf5_fh.create_dataset(
            scan.study_instance_uid,
            data=series,
            dtype="float32",
            chunks=True
        )

        if len(scan.annotations) > 0:
            start_stop = np.stack([annot.bbox_matrix()[2] for annot in scan.annotations])
        for i, dcm in enumerate(dicoms):
            records.append({
                "PatientID": dcm.PatientID,
                "StudyInstanceUID": dcm.StudyInstanceUID,
                "SeriesInstanceUID": dcm.SeriesInstanceUID,
                "SOPInstanceUID": dcm.SOPInstanceUID,
                "image_index": i,
                "nodule_present_on_image": int(
                    sum((start_stop[:, 0] <= i) & (i <= start_stop[:, 1])) >= 3
                ) if len(scan.annotations) > 0 else 0,
                "InstanceNumber": dcm.InstanceNumber,
                "ImagePositionPatient_0": dcm.ImagePositionPatient[0],
                "ImagePositionPatient_1": dcm.ImagePositionPatient[1],
                "ImagePositionPatient_2": dcm.ImagePositionPatient[2],
                "ImageOrientationPatient_0": dcm.ImageOrientationPatient[0],
                "ImageOrientationPatient_1": dcm.ImageOrientationPatient[1],
                "ImageOrientationPatient_2": dcm.ImageOrientationPatient[2],
                "ImageOrientationPatient_3": dcm.ImageOrientationPatient[3],
                "ImageOrientationPatient_4": dcm.ImageOrientationPatient[4],
                "ImageOrientationPatient_5": dcm.ImageOrientationPatient[5],
                "PixelSpacing_0": dcm.PixelSpacing[0],
                "PixelSpacing_1": dcm.PixelSpacing[1],
                "RescaleIntercept": dcm.RescaleIntercept,
                "RescaleSlope": dcm.RescaleSlope,
                "WindowCenter": dcm.get("WindowCenter"),
                "WindowWidth": dcm.get("WindowWidth"),
            })
    hdf5_fh.close()

    pd.DataFrame.from_records(records).to_csv(LIDC_DICOM_CSV, index=False)


def add_split_to_label_df(
    label_df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    patient_col: str = LIDC_PATIENT_COL,
    split_col: str = LIDC_SPLIT_COL,
):
    patients = label_df[patient_col].unique()

    # split between train and val+test
    split_ratio = val_size + test_size
    train_patients, test_val_patients = train_test_split(
        patients, test_size=split_ratio, random_state=RANDOM_SEED
    )

    # split between val and test
    test_split_ratio = test_size / (val_size + test_size)
    val_patients, test_patients = train_test_split(
        test_val_patients, test_size=test_split_ratio, random_state=RANDOM_SEED
    )
    train_rows = label_df[patient_col].isin(train_patients)
    label_df.loc[train_rows, split_col] = "train"
    val_rows = label_df[patient_col].isin(val_patients)
    label_df.loc[val_rows, split_col] = "valid"
    test_rows = label_df[patient_col].isin(test_patients)
    label_df.loc[test_rows, split_col] = "test"

    return label_df


if __name__ == "__main__":
    if (not LIDC_STUDY_HDF5.is_file()) or (not LIDC_DICOM_CSV.is_file()):
        print('\n'+'='*80)
        print(f'\nParsing study HDF5 to {LIDC_STUDY_HDF5} and creating {LIDC_DICOM_CSV}')
        print('-'*80)
        process_study_to_hdf5()
    else: 
        print('='*80)
        print(f'\n{LIDC_STUDY_HDF5} and {LIDC_DICOM_CSV} already existed and processed')
        print('-'*80)

    if not LIDC_TRAIN_CSV.is_file():
        print('='*80)
        print(f'\nProcessing LIDC dataset metadata and save as {LIDC_TRAIN_CSV}')
        print('-'*80)

        lidc = pd.read_csv(LIDC_DICOM_CSV)

        # full dataset split
        lidc = add_split_to_label_df(lidc, split_col=LIDC_SPLIT_COL)
        for split in ["train", "valid", "test"]:
            print(
                f"Full split (patients) {split}: "
                + f"{lidc[lidc[LIDC_SPLIT_COL] == split][LIDC_PATIENT_COL].nunique()}"
            )

        # # add instance path
        # rsna[RSNA_INSTANCE_PATH_COL] = rsna.apply(
        #     lambda x: f"{x[RSNA_STUDY_COL]}/{x[RSNA_SERIES_COL]}/{x[RSNA_INSTANCE_COL]}.dcm",
        #     axis=1,
        # )

#         # create dataset and loader to extract metadata
#         dataset = Metadata(rsna)
#         loader = DataLoader(
#             dataset, batch_size=1, shuffle=False, num_workers=12, collate_fn=lambda x: x
#         )

#         # get metadata
#         meta = []
#         for data in tqdm.tqdm(loader, total=len(loader)):
#             meta += [data[0]]
#         meta_df = pd.concat(meta, axis=0, ignore_index=True)

        # get slice number
        unique_studies = pd.DataFrame(lidc[LIDC_STUDY_COL].value_counts()).reset_index()
        unique_studies.columns = [LIDC_STUDY_COL, LIDC_NUM_SLICES_COL]
        lidc = lidc.merge(unique_studies, on=LIDC_STUDY_COL)

#         # indicate neiborig slices based on patient position
#         study_dfs = []
#         for study_name in tqdm.tqdm(
#             rsna[RSNA_STUDY_COL].unique(), total=rsna[RSNA_STUDY_COL].nunique()
#         ):
#             study_df = rsna[rsna[RSNA_STUDY_COL] == study_name].copy()

#             # order study instances 
#             study_df = study_df.sort_values("ImagePositionPatient_2")
#             study_df[RSNA_INSTANCE_ORDER_COL] = np.arange(len(study_df))

#             # get neighbors paths 
#             instance_paths = study_df[RSNA_INSTANCE_PATH_COL].tolist()
#             instance_paths = [instance_paths[0]] + instance_paths + [instance_paths[-1]]
#             study_df[RSNA_PREV_INSTANCE_COL] = instance_paths[:-2]
#             study_df[RSNA_NEXT_INSTANCE_COL] = instance_paths[2:]

#             study_dfs.append(study_df)

#         rsna = pd.concat(study_dfs, axis=0, ignore_index=True)

        
        # create windowed dataset: default window_size=24, min_abnormal_slice: 4
        window_df = process_window_df(lidc) 

        lidc = lidc.to_csv(LIDC_TRAIN_CSV, index=False)
    else: 
        print('='*80)
        print(f'\n{LIDC_TRAIN_CSV} already existed and processed')
        print('-'*80)
