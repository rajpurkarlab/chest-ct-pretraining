import argparse
import torch
import pe_models
import datetime
import os
import numpy as np
import pandas as pd
import wandb
import yaml

from collections import defaultdict
from pathlib import Path
from dateutil import tz
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)


seed_everything(23)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="paths to base config")
    parser.add_argument(
        "--train", action="store_true", default=False, help="specify to train model")
    parser.add_argument(
        "--debug", action="store_true", default=False, help="specify to debug model")
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="specify to test model"
        "By default run.py trains a model based on config file",)
    parser.add_argument(
        "--checkpoint", type=str, default=None)
    parser.add_argument(
        "--cv_split", type=int, default=1, help="Cross validation split")
    parser.add_argument(
        "--top_n", type=int, default=1, help="test based on top N best ckpt")
    parser.add_argument(
        "--sample_frac", type=float, default=None, help="test based on top N best ckpt")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="test based on top N best ckpt")
    parser.add_argument(
        "--test_split", type=str, default='test', help="test based on top N best ckpt")

    parser = Trainer.add_argparse_args(parser)

    args, unknown = parser.parse_known_args()
    cli = [u.strip("--") for u in unknown]  # remove strings leading to flag

    # add command line argments to config
    cfg = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(cli)
    cli_flat = pe_models.utils.flatten(cli)
    cfg.hyperparameters = cli_flat  # hyperparameter defaults
    cfg.data.cv_split = args.cv_split
    if args.gpus is not None:
        cfg.lightning.trainer.gpus = str(args.gpus)


    cfg.data.sample_frac = args.sample_frac 
    if args.checkpoint is not None: 
        cfg.checkpoint = args.checkpoint
    if args.batch_size is not None: 
        cfg.train.batch_size = args.batch_size
    cfg.test_split = args.test_split

    # edit experiment name
    if not OmegaConf.is_none(cfg, "trial_name"):
        cfg.experiment_name = f"{cfg.experiment_name}_{cfg.trial_name}"
    if not OmegaConf.is_none(cfg.data, 'sample_frac'):
        cfg.experiment_name = f"{cfg.experiment_name}_frac{args.sample_frac}"
    if cfg.data.positive_only:
        cfg.experiment_name = f"{cfg.experiment_name}_positive_only"
    if cfg.data.weighted_sample:
        cfg.experiment_name = f"{cfg.experiment_name}_weighted_sample"


    if args.sample_frac is not None: 
        cfg.data.sample_frac = args.sample_frac
        cfg.experiment_name = f"{cfg.experiment_name}_frac{args.sample_frac}"

    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    cfg.extension = timestamp

    # add checkpoint 
    if args.checkpoint is not None: 
        cfg.checkpoint = args.checkpoint

    # check debug
    if args.debug:
        cfg.train.num_workers = 0 
        cfg.lightning.trainer.gpus = 1
        cfg.lightning.trainer.distributed_backend = None

    return cfg, args


def create_directories(cfg):

    # set directory names
    if cfg.phase == "pretrain":
        cfg.output_dir = f"./data/output/{cfg.experiment_name}/{cfg/deep/u/marshuang80/data.cv_split}/{cfg.extension}/pretrain"
        cfg.lightning.logger.name = (
            f"{cfg.experiment_name}/{cfg.data.cv_split}/{cfg.extension}"
        )
        cfg.lightning.checkpoint_callback.dirpath = f"./data/ckpt/{cfg.experiment_name}/{cfg/deep/u/marshuang80/data.cv_split}/{cfg.extension}/pretrain"
    else:
        cfg.output_dir = (
            f"./data/output/{cfg.experiment_name}/{cfg.data.cv_split}/{cfg.extension}"
        )
        cfg.lightning.logger.name = (
            f"{cfg.experiment_name}/{cfg.data.cv_split}/{cfg.extension}"
        )
        cfg.lightning.checkpoint_callback.dirpath = (
            f"./data/ckpt/{cfg.experiment_name}/{cfg.data.cv_split}/{cfg.extension}"
        )


    # create directories
    if not os.path.exists(cfg.lightning.logger.save_dir):
        os.makedirs(cfg.lightning.logger.save_dir)
    if not os.path.exists(cfg.lightning.checkpoint_callback.dirpath):
        print(cfg.lightning.checkpoint_callback.dirpath)
        os.makedirs(cfg.lightning.checkpoint_callback.dirpath)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    return cfg


def setup(cfg, test_split=False):

    # create output, logger and ckpt directories if split != test
    if not test_split:
        cfg = create_directories(cfg)

        # logging
        loggers = [pl_loggers.csv_logs.CSVLogger(cfg.output_dir)]
        if "logger" in cfg.lightning:

            logger_type = cfg.lightning.logger.pop("logger_type")
            logger_class = getattr(pl_loggers, logger_type)
            logger = logger_class(**cfg.lightning.logger)
            loggers.append(logger)
            cfg.lightning.logger.logger_type = logger_type

            """
            if logger_type == "WandbLogger":
                # set sweep defaults
                hyperparameter_defaults = cfg.hyperparameters
                run = logger.experiment
                run.config.setdefaults(hyperparameter_defaults)

                # update cfg with new sweep parameters
                run_config = [f"{k}={v}" for k, v in run.config.items()]
                run_config = OmegaConf.from_dotlist(run_config)
                cfg = OmegaConf.merge(cfg, run_config)  # update defaults to CLI

                # set best metric
                if cfg.lightning.checkpoint_callback.mode == "max":
                    goal = "maximize"
                else:
                    goal = "minimize"
                metric = cfg.lightning.checkpoint_callback.monitor
                wandb.define_metric(f"{metric}", summary="best", goal=goal)
            """

        # callbacks
        callbacks = [LearningRateMonitor(logging_interval="step")]
        if "checkpoint_callback" in cfg.lightning:
            checkpoint_callback = ModelCheckpoint(**cfg.lightning.checkpoint_callback)
            callbacks.append(checkpoint_callback)
        if "early_stopping_callback" in cfg.lightning:
            early_stopping_callback = EarlyStopping(
                **cfg.lightning.early_stopping_callback
            )
            callbacks.append(early_stopping_callback)

        # save config
        config_path = os.path.join(cfg.output_dir, "config.yaml")
        config_path_ckpt = os.path.join(
            cfg.lightning.checkpoint_callback.dirpath, "config.yaml"
        )
        with open(config_path, "w") as fp:
            OmegaConf.save(config=cfg, f=fp.name)
        with open(config_path_ckpt, "w") as fp:
            OmegaConf.save(config=cfg, f=fp.name)

    else:
        loggers = []
        callbacks = []
        checkpoint_callback = None

    # get datamodule
    dm = pe_models.builder.build_data_module(cfg)
    cfg.data.num_batches = len(dm.train_dataloader())

    # define lightning module
    model = pe_models.builder.build_lightning_model(cfg)

    # setup pytorch-lightning trainer
    trainer_args = argparse.Namespace(**cfg.lightning.trainer)
    trainer = Trainer.from_argparse_args(
        args=trainer_args, deterministic=True, callbacks=callbacks, logger=loggers
    )

    # auto learning rate finder
    if trainer_args.auto_lr_find is not False:
        lr_finder = trainer.tuner.lr_find(model, datamodule=dm)
        new_lr = lr_finder.suggestion()
        model.lr = new_lr
        print(f"learning rate updated to {new_lr}")

    return trainer, model, dm, checkpoint_callback


def save_best_checkpoints(checkpoint_callback, cfg, return_best=True):
    ckpt_paths = os.path.join(
        cfg.lightning.checkpoint_callback.dirpath, "best_ckpts.yaml"
    )
    checkpoint_callback.to_yaml(filepath=ckpt_paths)
    if return_best:
        ascending = cfg.lightning.checkpoint_callback.mode == "min"
        best_ckpt_path = pe_models.utils.get_best_ckpt_path(ckpt_paths, ascending)
        return best_ckpt_path


def find_best_ckpt(cfg, top_n):
    """Finding best ckpt for a wandb hyperparameter sweep"""

    output_dir = f"./data/output/{cfg.experiment_name}/{cfg.data.cv_split}"
    results = defaultdict(list)
    sweep_path = Path(output_dir)
    experiment_dirs = [p for p in sweep_path.iterdir() if p.is_dir()]

    for p in experiment_dirs:

        metrics_csv = p / "default" / "version_0" / "metrics.csv"
        config_file = p / "config.yaml"

        # run errored
        if not metrics_csv.exists():
            continue

        df = pd.read_csv(metrics_csv)
        try:
            curr_best_epoch = int(
                df.sort_values("val/mean_auroc", ascending=False).head(1)["epoch"].values[0]
            )
            curr_best_step = int(
                df.sort_values("val/mean_auroc", ascending=False).head(1)["step"].values[0]
            )
            curr_best_auroc = (
                df.sort_values("val/mean_auroc", ascending=False)
                .head(1)["val/mean_auroc"]
                .values[0]
            )
            curr_train_auroc = df[
                (df.epoch == curr_best_epoch) & (~df["train/mean_auroc"].isna())
            ].iloc[0]["train/mean_auroc"]
        except:
            # failed runs
            print(f"Unable to read {metrics_csv}")
            continue

        experiment_name = str(p).split("output")[1][1:]  # remote leading '/'
        results["experiment_name"].append(experiment_name)
        results["val_auroc"].append(curr_best_auroc)
        results["train_auroc"].append(curr_train_auroc)
        results["config_file"].append(config_file)
        results["results_path"].append(metrics_csv)
        results["best_ckpt"].append(
            str(
                Path("./data/ckpt")
                / experiment_name
                / f"epoch={curr_best_epoch}-step={curr_best_step}.ckpt"
            )
        )

    df = pd.DataFrame.from_dict(results)
    df = df.sort_values("val_auroc", ascending=False)
    loc = top_n - 1
    best = df.iloc[loc]
    best_train_auroc = best["train_auroc"]
    best_auroc = best["val_auroc"]
    best_ckpt = best["best_ckpt"]
    best_config_file = best["config_file"]
    best_result_path = best["results_path"]

    print(f"\nBest training AUROC: {best_train_auroc: .3f}")
    print(f"Best validation AUROC: {best_auroc: .3f}")
    print(f"Using checkpoint: {str(best_ckpt)}\n")
    best_cfg = OmegaConf.load(best_config_file)
    return str(best_ckpt), best_cfg.model, str(best_result_path)


if __name__ == "__main__":
    cfg, args = parse_configs()

    if args.train:
        trainer, model, dm, checkpoint_callback = setup(cfg)
        trainer.fit(model, dm)
        best_ckpt = save_best_checkpoints(checkpoint_callback, cfg, return_best=True)
        cfg.checkpoint = best_ckpt
        print(f"Best checkpoint path: {best_ckpt}")

    if args.test:
        if OmegaConf.is_none(cfg, "checkpoint"):
            cfg.checkpoint, cfg.model, cfg.save_dir = find_best_ckpt(cfg, args.top_n)
        
        print("="*80)
        print(cfg.checkpoint)
        print("="*80)
        cfg.output_dir = '/'.join(cfg.checkpoint.split('/')[:-1]).replace('ckpt','output')
        print(f'Output dir: {cfg.output_dir}')
        trainer, model, dm, checkpoint_callback = setup(cfg, test_split=True)
        trainer.test(model=model, datamodule=dm)
