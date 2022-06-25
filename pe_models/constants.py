from pathlib import Path

PROJECT_DATA_DIR = Path("<Path to project data directory>")
if not PROJECT_DATA_DIR.is_dir():
    print(
        "\nProject data directory not specified. Please update data path in "
        + "PROJECT_DATA_DIR in pe_models/constants.py "
    )
    PROJECT_DATA_DIR = Path("./data/")

## RSNA
RSNA_DATA_DIR = PROJECT_DATA_DIR / "rsna"  # check why isn't this just rsna
if not RSNA_DATA_DIR.is_dir():
    print(
        "\nPlease download the RSNA dataset from \n"
        + "    https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection\n"
        + f"and place the downloaded dataset in {PROJECT_DATA_DIR}"
    )
    RSNA_DATA_DIR = PROJECT_DATA_DIR / 'demo'

RSNA_TRAIN_DIR = RSNA_DATA_DIR / "train"
RSNA_ORIGINAL_TRAIN_CSV = RSNA_DATA_DIR / "train.csv"
RSNA_TRAIN_CSV = RSNA_DATA_DIR / "rsna_train_master.csv"
RSNA_TEST_DIR = RSNA_DATA_DIR / "test"
RSNA_TEST_CSV = RSNA_DATA_DIR / "test.csv"
RSNA_STUDY_HDF5 = RSNA_DATA_DIR / "rsna_study.hdf5" 

# Dataframe headers
RSNA_STUDY_COL = "StudyInstanceUID"
RSNA_SERIES_COL = "SeriesInstanceUID"
RSNA_INSTANCE_COL = "SOPInstanceUID"
RSNA_PREV_INSTANCE_COL = "PrevSOPInstanceUID"
RSNA_NEXT_INSTANCE_COL = "NextSOPInstanceUID"
RSNA_PE_TARGET_COL = "negative_exam_for_pe"
RSNA_ORIGINAL_SPLIT_COL = "OriginalSplit"
RSNA_SPLIT_COL = "Split"
RSNA_INSTITUTION_SPLIT_COL = "InstitutionSplit"
RSNA_INSTITUTION_COL = "Institution"
RSNA_NUM_SLICES_COL = "NumSlices"
RSNA_SLICE_ORDER_COL = "SliceOrder"
RSNA_NUM_WINDOW_COL = "NumWindows"
RSNA_INSTANCE_PATH_COL = "InstancePath"
RSNA_INSTANCE_ORDER_COL = "InstanceOrder"
RSNA_PE_SLICE_COL = "pe_present_on_image"
RSNA_TARGET_COLS = [
    "negative_exam_for_pe",
    "indeterminate",
    "rv_lv_ratio_gte_1",
    "rv_lv_ratio_lt_1",
    "leftsided_pe",
    "rightsided_pe",
    "chronic_pe",
    "acute_and_chronic_pe",
    "central_pe",
]
RSNA_SLICE_TARGET_COLS = [
    "pe_present_on_image",
    "indeterminate",
    "rv_lv_ratio_gte_1",
    "rv_lv_ratio_lt_1",
    "leftsided_pe",
    "rightsided_pe",
    "chronic_pe",
    "acute_and_chronic_pe",
    "central_pe",
]
RSNA_LOCATION_TARGET_COLS = [
    "pe_present_on_image",
    "leftsided_pe",
    "rightsided_pe",
    "central_pe",
]
RSNA_PE_PROPERTY_TARGET_COLS = [
    "pe_present_on_image",
    "leftsided_pe",
    "rightsided_pe",
    "central_pe",
    "chronic_pe",
    "acute_and_chronic_pe",
]

RSNA_TARGET_TYPES = {
    "rsna_targets": RSNA_TARGET_COLS,
    "rsna_slice_targets": RSNA_SLICE_TARGET_COLS,
    "rsna_location_targets": RSNA_LOCATION_TARGET_COLS,
    "rsna_pe_slice_target": [RSNA_PE_SLICE_COL],
    "rsna_pe_target": [RSNA_PE_TARGET_COL],
    "rsna_pe_property_targets": RSNA_PE_PROPERTY_TARGET_COLS,
}

RSNA_LOSS_WEIGHT = {
    "negative_exam_for_pe": 0.0736196319,
    "indeterminate": 0.09202453988,
    "rv_lv_ratio_gte_1": 0.2346625767,
    "rv_lv_ratio_lt_1": 0.0782208589,
    "leftsided_pe": 0.06257668712,
    "rightsided_pe": 0.06257668712,
    "chronic_pe": 0.1042944785,
    "acute_and_chronic_pe": 0.1042944785,
    "central_pe": 0.1877300613,
    "pe_present_on_image": 0.07361963,
}

RSNA_DICOM_HEADERS = [
    "SOPInstanceUID",
    "SeriesInstanceUID",
    "StudyInstanceUID",
    "InstanceNumber",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
    "RescaleIntercept",
    "RescaleSlope",
    "WindowCenter",
    "WindowWidth",
]

AIR_HU_VAL = -1000.0
RANDOM_SEED = 2


# Used to extract Stanford studies from RSNA
# Requires token and mapping
RSNA_MDAI_JSON = None
RSNA_STANFORD_MAPPING = None
MDAI_TOKEN = None
MDAI_DOMAIN = None
STANFORD_PE_METADATA = None


# Stanford Cohort
STANFORD_DATA_DIR = PROJECT_DATA_DIR / 'stanford' 
STANFORD_NO_RSNA_CSV = STANFORD_DATA_DIR / 'stanford_ctpe_no_rsna.csv'
STANFORD_NO_RSNA_HDF5 = STANFORD_DATA_DIR / 'stanford_ctpe_no_rsna.hdf5'

## LIDC
LIDC_DATA_DIR = PROJECT_DATA_DIR / "lidc"
LIDC_TRAIN_CSV = LIDC_DATA_DIR / "lidc_train.csv"
LIDC_STUDY_HDF5 = LIDC_DATA_DIR / "lidc_study.hdf5"
LIDC_DICOM_CSV = LIDC_DATA_DIR / "lidc_2d.csv"

LIDC_PATIENT_COL = "PatientID"
LIDC_STUDY_COL = "StudyInstanceUID"
LIDC_SERIES_COL = "SeriesInstanceUID"
LIDC_INSTANCE_COL = "SOPInstanceUID"
LIDC_NUM_WINDOW_COL = "NumWindows"
LIDC_NUM_SLICES_COL = "NumSlices"
LIDC_SPLIT_COL = "Split"
LIDC_NOD_SLICE_COL = "nodule_present_on_image"
LIDC_INSTANCE_ORDER_COL = "image_index" # or is this InstanceNumber
