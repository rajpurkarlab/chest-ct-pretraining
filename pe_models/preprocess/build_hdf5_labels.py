import numpy as np
import pandas as pd
import cv2
import h5py
from tqdm import tqdm

import pylidc as pl

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

records = []
hdf5_fh = h5py.File("lidc_study.hdf5", "a")
for scan in tqdm(pl.query(pl.Scan).all()):
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

pd.DataFrame.from_records(records).to_csv("lidc_2d.csv", index=False)