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


def process_window_df(df:pd.DataFrame, num_slices:int=24, min_abnormal_slice:int=4, stride:int=None): 

    window_labels_csv = RSNA_DATA_DIR / f"rsna_window_{num_slices}_min_abnormal_{min_abnormal_slice}_stride_{stride}.csv"

    # count number of windows per slice
    #count_num_windows = lambda x: x // num_slices\
    #    + (1 if x % num_slices > 0 else 0)
    count_num_windows = lambda x: (x - num_slices) // stride
    df[RSNA_NUM_WINDOW_COL] = df[RSNA_NUM_SLICES_COL].apply(
        count_num_windows)

    # get windows list 
    df_study = df.groupby([RSNA_STUDY_COL]).head(1)
    window_labels = defaultdict(list)

    # studies
    for _, row in tqdm.tqdm(df_study.iterrows(), total=df_study.shape[0]):
        study_name = row[RSNA_STUDY_COL]
        split = row[RSNA_SPLIT_COL]

        if RSNA_INSTITUTION_SPLIT_COL in df_study.columns:
            institution_split = row[RSNA_INSTITUTION_SPLIT_COL]
        else:
            institution_split = None 
        study_df = df[df[RSNA_STUDY_COL] == study_name] 
        study_df = study_df.sort_values("ImagePositionPatient_2")

        # windows 
        for idx in range(row[RSNA_NUM_WINDOW_COL]):
            start_idx = idx * stride
            end_idx = (idx * stride) + num_slices

            window_df = study_df.iloc[start_idx: end_idx]
            num_positives_slices = window_df[RSNA_PE_SLICE_COL].sum()
            label = 1 if num_positives_slices >= min_abnormal_slice else 0
            window_labels[RSNA_STUDY_COL].append(study_name)
            window_labels['index'].append(idx)
            window_labels['label'].append(label)
            window_labels[RSNA_SPLIT_COL].append(split)
            window_labels[RSNA_INSTITUTION_SPLIT_COL].append(institution_split)
            window_labels[RSNA_INSTANCE_PATH_COL].append(window_df[RSNA_INSTANCE_PATH_COL].tolist())
            window_labels["ImagePositionPatient_2"].append(window_df["ImagePositionPatient_2"].tolist())
            window_labels[RSNA_PE_SLICE_COL].append(window_df[RSNA_PE_SLICE_COL].tolist())
            window_labels[RSNA_INSTANCE_COL].append(window_df[RSNA_INSTANCE_COL].tolist())
            window_labels[RSNA_INSTANCE_ORDER_COL].append(window_df[RSNA_INSTANCE_ORDER_COL].tolist())
    df = pd.DataFrame.from_dict(window_labels) 
    df.to_csv(window_labels_csv)

    return df 


def process_study_to_hdf5(csv_path:str = RSNA_TRAIN_CSV):

    df = pd.read_csv(csv_path)

    # indicate neiborig slices based on patient position
    hdf5_fh = h5py.File(RSNA_STUDY_HDF5, 'a')
    for study_name in tqdm.tqdm(
        df[RSNA_STUDY_COL].unique(), total=df[RSNA_STUDY_COL].nunique()
    ):
        study_df = df[df[RSNA_STUDY_COL] == study_name].copy()

        # order study instances 
        study_df = study_df.sort_values("ImagePositionPatient_2")

        # save paths to hdf5 
        instance_paths = study_df[RSNA_INSTANCE_PATH_COL].tolist()
        series = np.stack([utils.read_dicom(RSNA_TRAIN_DIR / path, 256) for path in instance_paths])
        hdf5_fh.create_dataset(study_name, data=series, dtype='float32', chunks=True)

    # clean up
    hdf5_fh.close()



class Metadata(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        dcm = pydicom.dcmread(
            RSNA_TRAIN_DIR / row[RSNA_INSTANCE_PATH_COL], stop_before_pixels=True
        )

        metadata = {}
        for k in RSNA_DICOM_HEADERS:
            try:
                att = getattr(dcm, k)
                if k in ["InstanceNumber", "RescaleSlope", "RescaleIntercept"]:
                    metadata[k] = float(att)
                elif k in [
                    "PixelSpacing",
                    "ImagePositionPatient",
                    "ImageOrientationPatient",
                ]:
                    for ind, coord in enumerate(att):
                        metadata[f"{k}_{ind}"] = float(coord)
                else:
                    metadata[k] = str(att)
            except Exception as e:
                print(e)

        return pd.DataFrame(metadata, index=[0])


def add_split_to_label_df(
    label_df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    study_col: str = RSNA_STUDY_COL,
    split_col: str = RSNA_SPLIT_COL,
):
    patients = label_df[study_col].unique()

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
    train_rows = label_df[study_col].isin(train_patients)
    label_df.loc[train_rows, split_col] = "train"
    val_rows = label_df[study_col].isin(val_patients)
    label_df.loc[val_rows, split_col] = "valid"
    test_rows = label_df[study_col].isin(test_patients)
    label_df.loc[test_rows, split_col] = "test"

    return label_df


if __name__ == "__main__":

    if not RSNA_TRAIN_CSV.is_file():
        print('='*80)
        print('\nProcessing RSNA dataset metadata and save as {RSNA_TRAIN_CSV}')
        print('-'*80)

        # applying train/val/test split only on train csv (test.csv does not contain labels)
        rsna = pd.read_csv(RSNA_ORIGINAL_TRAIN_CSV)

        # full dataset split
        rsna[RSNA_ORIGINAL_SPLIT_COL] = "train"
        rsna = add_split_to_label_df(rsna, split_col=RSNA_SPLIT_COL)
        for split in ["train", "valid", "test"]:
            print(
                f"Full split {split}: "
                + f"{rsna[rsna.Split == split][RSNA_STUDY_COL].nunique()}"
            )

        # if raw rsna annotations are availible - extract stanford studies
        if RSNA_MDAI_JSON is not None:

            # read RSNA annotations
            mdai_client = mdai.Client(domain=MDAI_DOMAIN, access_token=MDAI_TOKEN)
            results = mdai.common_utils.json_to_dataframe(RSNA_MDAI_JSON)
            rsna_anno = results["annotations"]
            rsna_studies = rsna_anno[RSNA_STUDY_COL].unique()

            # extract overlapping studies
            stanford_df = pd.read_csv(STANFORD_PE_METADATA)
            stanford_overlapping_df = stanford_df[
                stanford_df.StudyInstanceUID.isin(rsna_studies)
            ]
            anon_stanford_studies = list(stanford_overlapping_df[RSNA_STUDY_COL].unique())
            rsna_stanford_df = rsna_anno[
                rsna_anno[RSNA_STUDY_COL].isin(anon_stanford_studies)
            ]

            # create mapping from stanford study UID to rsna hash
            rsna_stanford_mapping_df = pd.read_csv(RSNA_STANFORD_MAPPING)
            stanford_2_rsna = dict(
                zip(
                    rsna_stanford_mapping_df.SOPInstanceUID,
                    rsna_stanford_mapping_df.SOPInstanceUID_hash,
                )
            )
            rsna_stanford_df["anon_instance"] = rsna_stanford_df["SOPInstanceUID"].apply(
                lambda x: stanford_2_rsna[x]
                if (x in stanford_2_rsna and x is not None)
                else None
            )
            overlap_anon_instance = rsna_stanford_df["anon_instance"].unique()

            # split stanford vs other institutions
            stanford_studies = rsna[rsna[RSNA_INSTANCE_COL].isin(overlap_anon_instance)][
                RSNA_STUDY_COL
            ].unique()
            rsna_external_df = rsna[~rsna[RSNA_STUDY_COL].isin(stanford_studies)]
            rsna_stanford_df = rsna[rsna[RSNA_STUDY_COL].isin(stanford_studies)]
            rsna_stanford_df.loc[:, RSNA_INSTITUTION_COL] = "Stanford"
            rsna_external_df.loc[:, RSNA_INSTITUTION_COL] = "Other"

            # create split for data from other institutions
            rsna_external_df = add_split_to_label_df(
                rsna_external_df, split_col=RSNA_INSTITUTION_SPLIT_COL
            )
            rsna_stanford_df.loc[:, RSNA_INSTITUTION_SPLIT_COL] = "stanford_test"
            rsna = pd.concat([rsna_external_df, rsna_stanford_df])

            for split in ["train", "valid", "test", "stanford_test"]:
                print(
                    f"Institution split {split}: "
                    + f"{rsna[rsna[RSNA_INSTITUTION_SPLIT_COL] == split][RSNA_STUDY_COL].nunique()}"
                )

        # add instance path
        rsna[RSNA_INSTANCE_PATH_COL] = rsna.apply(
            lambda x: f"{x[RSNA_STUDY_COL]}/{x[RSNA_SERIES_COL]}/{x[RSNA_INSTANCE_COL]}.dcm",
            axis=1,
        )

        # create dataset and loader to extract metadata
        dataset = Metadata(rsna)
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=12, collate_fn=lambda x: x
        )

        # get metadata
        meta = []
        for data in tqdm.tqdm(loader, total=len(loader)):
            meta += [data[0]]
        meta_df = pd.concat(meta, axis=0, ignore_index=True)

        # get slice number
        unique_studies = pd.DataFrame(meta_df[RSNA_STUDY_COL].value_counts()).reset_index()
        unique_studies.columns = [RSNA_STUDY_COL, RSNA_NUM_SLICES_COL]
        meta_df = meta_df.merge(unique_studies, on=RSNA_STUDY_COL)

        # join with metadata with labels
        rsna = rsna.set_index([RSNA_STUDY_COL, RSNA_SERIES_COL, RSNA_INSTANCE_COL])
        meta_df = meta_df.set_index([RSNA_STUDY_COL, RSNA_SERIES_COL, RSNA_INSTANCE_COL])
        rsna = rsna.join(meta_df, how="left").reset_index()

        # indicate neiborig slices based on patient position
        study_dfs = []
        for study_name in tqdm.tqdm(
            rsna[RSNA_STUDY_COL].unique(), total=rsna[RSNA_STUDY_COL].nunique()
        ):
            study_df = rsna[rsna[RSNA_STUDY_COL] == study_name].copy()

            # order study instances 
            study_df = study_df.sort_values("ImagePositionPatient_2")
            study_df[RSNA_INSTANCE_ORDER_COL] = np.arange(len(study_df))

            # get neighbors paths 
            instance_paths = study_df[RSNA_INSTANCE_PATH_COL].tolist()
            instance_paths = [instance_paths[0]] + instance_paths + [instance_paths[-1]]
            study_df[RSNA_PREV_INSTANCE_COL] = instance_paths[:-2]
            study_df[RSNA_NEXT_INSTANCE_COL] = instance_paths[2:]

            study_dfs.append(study_df)

        rsna = pd.concat(study_dfs, axis=0, ignore_index=True)
        rsna.to_csv(RSNA_TRAIN_CSV, index=False)

        # create windowed dataset: default window_size=24, min_abnormal_slice: 4
        window_df = process_window_df(rsna) 
    else: 
        print('='*80)
        print(f'\n{RSNA_TRAIN_CSV} already existed and processed')
        print('-'*80)

    if not RSNA_STUDY_HDF5.is_file():
        print('\n'+'='*80)
        print(f'\nParsing study HDF5 to {RSNA_STUDY_HDF5}')
        print('-'*80)
        process_study_to_hdf5()
    else: 
        print('='*80)
        print(f'\n{RSNA_STUDY_HDF5} already existed and processed')
        print('-'*80)
