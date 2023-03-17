# Video pretraining advances 3D deep learning on chest CT tasks
This repository contains code to train and evaluate models on the RSNA PE dataset and the LIDC-IDRI dataset.

## Table of Contents
0. [System Requirements](#SystemRequirements)
0. [Installation](#Installation)
0. [Datasets](#Datasets)
0. [Usage](#Usage)
0. [Demo](#Demo)

## System Requirements

### Hardware requirements

The data processing steps requires only a standard computer with enough RAM to support the in-memory operations.

For training and testing models, a computer with sufficient GPU memory is recommended. 

### Software requirements
#### OS requirements
All models have been trained and tested on a Linux system (Ubuntu 16.04)

#### Python dependencies

All dependencies can be found in **environment.yml**


## Installation 

1. Please install [Anaconda](https://docs.conda.io/en/latest/miniconda.html) in order to create a Python environment.
2. Clone this repo (from the command-line: `git clone git@github.com:rajpurkarlab/2021-fall-chest-ct.git`).
3. Create the environment: `conda env create -f environment.yml`.
4. Activate the environment: `source activate pe_models`.
5. Install [PyTorch 1.7.1](https://pytorch.org/get-started/locally/) with the right CUDA version.

Installation should take less than 10 minutes with stable internet. 

## Datasets

### RSNA

Download dataset from: [RSNA PE Dataset](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection)

Make sure to update **PROJECT_DATA_DIR** in `pe_models/constants.py` with path to the directory that contains the RSNA dataset.

#### Preprocessing

Please download the pre-processed label file that contains data split and DICOM header infomation using this [link](https://stanfordmedicine.box.com/s/nlatp1dgg47qry1g7hhr0n87mlavj887) and place it in the RSNA data directory. 

Alternatively, you can create the pre-processed file by running:
```bash
$ python pe_models/preprocess/rsna.py
```

#### Test 
To ensure that the dataset is correct and that data are loading in the correct format, run the following unittest: 

```bash
$ python -W ignore -m unittest
```

Note that this might take a couple of minutes to complete. 

You can also visually inspect example inputs in `data/test/` after the unittest is complete. 

### LIDC

Download dataset from [TCIA Public Access](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) into a `PROJECT_DATA_DIR/lidc` folder.

#### Preprocessing

Install *pylidc* and set up your `~/.pylidcrc` file using the [official installation instructions](https://pylidc.github.io/install.html).

You can then create all the necessary pre-processed files by running:

```bash
$ python pe_models/preprocess/lidc.py
```

You can then set the `type` in an experiment YAML to `lidc-window` or `lidc-2d` to train on the LIDC dataset.

## Usage

To train a model, run the following: 

```bash
python run.py --config <path_to_config_file> --train
```

For more documentation, please run: 

```bash 
python run.py --help
```

To test a model, use the `--test` flag, making sure that either the `--checkpoint` flag is specified or that the config YAML contains a **checkpoint** entry:

```bash
python run.py --config <path_to_config_file> --checkpoint <path_to_ckpt> --test
```

To featurize all studies in a dataset (to run a 1d model for example), use the `--test_split all` flag

Example configs can be found in **./configs/**

### Run hyperparameter sweep with wandb

Example hyperparameter sweep configs for each model can be found in **./configs/**

```
wandb sweep <path_to_sweep_config>
wandb agent <sweep-id>
```
### Custom dataset: 
To train/test model on custom datasets: 
1. Please ensure that your data adhere to the same format as the RSNA/LIDC dataset. (See [Example](https://stanfordmedicine.box.com/s/nlatp1dgg47qry1g7hhr0n87mlavj887))
2. Create a dataloader similar to RSNA/LIDC in ./datasets and update ./datasets/__init__.py to include the name of your custom dataloader. 
3. Make sure the *data.type* in your config file points to the name of your dataloader. 

## Demo

To run train/test script on a simulated demo dataset, use: 

```
python run.py --config ./data/demo/resnet18_demo.yaml --checkpoint <path_to_ckpt> --test
```

You should expect the following results:

```
{'test/mean_auprc': 0.9107142686843872,
 'test/mean_auroc': 0.9166666865348816,
 'test/negative_exam_for_pe_auprc': 0.9107142686843872,
 'test/negative_exam_for_pe_auroc': 0.9166666865348816,
 'test_loss': 0.6920164227485657,
 'test_loss_epoch': 0.6920164227485657}
 ```
With a GPU, this should take less than 10 minutes to run. 
