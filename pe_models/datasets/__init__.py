from . import data_module, dataset_1d, dataset_2d, dataset_3d

ALL_DATASETS = {
    "1d": dataset_1d.PEDataset1D,
    "1d_stanford": dataset_1d.PEDataset1DStanford,
    "2d": dataset_2d.PEDataset2D,
    "2d_mae": dataset_2d.PEDataset2DMAE,
    "2d_stanford": dataset_2d.PEDataset2DStanford,
    "3d": dataset_3d.PEDataset3D,
    "window": dataset_3d.PEDatasetWindow,
    "window_stanford": dataset_3d.PEDatasetWindowStanford,
    "lidc-window": dataset_3d.LIDCDatasetWindow,
    "lidc-2d": dataset_2d.LIDCDataset2D,
    "lidc-1d": dataset_1d.LIDCDataset1D,
    "demo": dataset_2d.DemoDataset2D,
}
