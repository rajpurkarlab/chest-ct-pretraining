import numpy as np
import collections
import yaml
import pandas as pd
import requests
import pydicom
import cv2
import torch
import math

from matplotlib import pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score


def get_auroc(y, prob, keys):

    if type(y) == torch.Tensor:
        y = y.detach().cpu().numpy()
    if type(prob) == torch.Tensor:
        prob = prob.detach().cpu().numpy()

    auroc_dict = {}
    for i, k in enumerate(keys):
        y_cls = y[:, i]
        prob_cls = prob[:, i]

        if np.isnan(prob_cls).any():
            auroc_dict[k] = 0
        elif len(set(y_cls)) == 1:
            auroc_dict[k] = 0
        else:
            auroc_dict[k] = roc_auc_score(y_cls, prob_cls)
    auroc_dict["mean"] = np.mean([v for _, v in auroc_dict.items()])
    return auroc_dict


def get_auprc(y, prob, keys):

    if type(y) == torch.Tensor:
        y = y.detach().cpu().numpy()
    if type(prob) == torch.Tensor:
        prob = prob.detach().cpu().numpy()

    auprc_dict = {}
    for i, k in enumerate(keys):
        y_cls = y[:, i]
        prob_cls = prob[:, i]

        if np.isnan(prob_cls).any():
            auprc_dict[k] = 0
        elif len(set(y_cls)) == 1:
            auprc_dict[k] = 0
        else:
            auprc_dict[k] = average_precision_score(y_cls, prob_cls)
    auprc_dict["mean"] = np.mean([v for _, v in auprc_dict.items()])
    return auprc_dict


def flatten(d, parent_key="", sep="."):
    """flatten a nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_best_ckpt_path(ckpt_paths, ascending=False):
    """get best ckpt path from a list of ckpt paths

    ckpt_paths: JSON file with ckpt path to metric pair
    ascending: sort paths based on ascending or descending metrics
    """

    with open(ckpt_paths, "r") as stream:
        ckpts = yaml.safe_load(stream)

    ckpts_df = pd.DataFrame.from_dict(ckpts, orient="index").reset_index()
    ckpts_df.columns = ["path", "metric"]
    best_ckpt_path = (
        ckpts_df.sort_values("metric", ascending=ascending).head(1)["path"].item()
    )

    return best_ckpt_path


def read_dicom(file_path: str, imsize):
    """TODO: repeated between dataset base """

    # read dicom
    dcm = pydicom.dcmread(file_path)
    pixel_array = dcm.pixel_array

    # rescale
    intercept = dcm.RescaleIntercept
    slope = dcm.RescaleSlope
    pixel_array = pixel_array * slope + intercept

    # resize
    resize_shape = imsize 
    pixel_array = cv2.resize(
        pixel_array, (resize_shape, resize_shape), interpolation=cv2.INTER_AREA
    )

    return pixel_array

def windowing(pixel_array: np.array, window_center: int, window_width: int):
    """TODO: repeated between dataset base """

    lower = window_center - window_width // 2
    upper = window_center + window_width // 2
    pixel_array = np.clip(pixel_array.copy(), lower, upper)
    pixel_array = (pixel_array - lower) / (upper - lower)

    return pixel_array


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def visualize_examples(x, num_viz, save_dir):

    f, axarr = plt.subplots(num_viz, 3, figsize=(3, num_viz))
    plt.subplots_adjust(wspace=0, hspace=0)

    # make sure channel first
    if x.shape[1] == 3:
        x = np.transpose(x, (1, 0, 2, 3))
    factor = x.shape[1] // num_viz

    for i in range(num_viz):
        for j in range(3):
            image = x[j, i * factor, :, :]
            image = np.repeat(np.expand_dims(image, -1), 3, axis=-1)
            axarr[i][j].imshow(image)

    plt.setp(axarr, xticks=[], yticks=[])
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.05
    )
    plt.savefig(save_dir, dpi=100)
    print(f"Example images saved at: {save_dir}")


def linear_warmup_then_cosine(last_iter, warmup, max_iter, delay=None):
    if delay is not None:
        last_iter = max(0, last_iter - delay)

    if last_iter < warmup:
        # Linear warmup period
        return float(last_iter) / warmup
    elif last_iter < max_iter:
        # Cosine annealing
        return (1 + math.cos(math.pi * (last_iter - warmup) / max_iter)) / 2
    else:
        # Done
        return 0.