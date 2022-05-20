# PE Models Benchmark 
Benchmarking models for PE detection 

## Usage

Start by [installing PyTorch 1.7.1](https://pytorch.org/get-started/locally/)
with the right CUDA version, then clone this repository and install the
dependencies.  

```bash
$ git clone git@github.com:marshuang80/pe_models_benchmark.git
$ cd pe_models_benchmark 
$ conda env create -f environment.yml
```

## Dataset 

Download dataset from: [RSNA PE Dataset](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection)

Make sure to udpate **PROJECT_DATA_DIR** in `pe_models/constants.py` with path to the directory that contains the RSNA dataset.

### Preprocessing

Please download the pre-processed label file that contains data split and DICOM header infomation using this [link](https://stanfordmedicine.box.com/s/nlatp1dgg47qry1g7hhr0n87mlavj887) and place it in the RSNA data directory. 

Alternatively, you can create the pre-processed file by running:
```bash
$ python pe_models/preprocess/rsna.py
```

### Test 
To ensure that the dataset is correct and that data are loading in the correct format, run the following unittest: 

```bash
$  python -W ignore -m unittest
```

Note that this might take a couple of minutes to complete. 

You can also visually inspect example inputs in `data/test/` after the unittest is complete. 

## Training the model

To train a model, run the following: 

```bash
python run.py --config <path_to_config_file> --train
```

For more documentation, please run: 

```bash 
python run.py --help
```

### Run hyperparameter sweep with wandb

Example hyperparameter sweep configs for each model can be found in **./configs/**

```
wandb sweep <path_to_sweep_config>
wandb agent <sweep-id>
```
