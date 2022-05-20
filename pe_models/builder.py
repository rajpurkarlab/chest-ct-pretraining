import omegaconf
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import albumentations as A
import cv2

from . import models
from . import lightning
from . import datasets
from . import utils
from .loss import BinaryFocalLoss
from albumentations.pytorch import ToTensorV2
from .constants import *
from functools import partial
from omegaconf import OmegaConf


def build_data_module(cfg):
    return datasets.data_module.PEDataModule(cfg)


def build_dataset(cfg):
    if cfg.data.type.lower() in datasets.ALL_DATASETS:
        return datasets.ALL_DATASETS[cfg.data.type.lower()]
    else:
        raise NotImplementedError(
            f"Dataset not implemented for {cfg.data.type.lower()}"
        )


def build_lightning_model(cfg):


    if cfg.data.type == 'window' or cfg.data.type == 'lidc-window':
        model = lightning.PEWindowClassificationLightningModel
    else:
        model = lightning.PEClassificationLightningModel

    # TODO: ugly logic
    # mae vit uses it's ownd data loading 
    if OmegaConf.is_none(cfg, "checkpoint"):
        return model(cfg)
    else:
        checkpoint_path = cfg.checkpoint
        print('='*80)
        print(f'*** Using checkpoint: {checkpoint_path}')
        print('='*80)
        return model.load_from_checkpoint(checkpoint_path, cfg=cfg)


def build_model(cfg):
    num_class = len(RSNA_TARGET_TYPES[cfg.data.targets])
    return models.ALL_MODELS[cfg.model.type.lower()](cfg, num_class)


def build_optimizer(cfg, model):
    # different scheduler for fine-tune and encoder
    if not OmegaConf.is_none(cfg.model, "boundary_layer_name"): 
        print('Finetuning parameters')
        params = model.fine_tuning_parameters(
            cfg.model.boundary_layer_name, cfg.lightning.trainer.lr)
    else: 
        params = [p for p in model.parameters() if p.requires_grad]
    # optimizer_name = cfg.train.optimizer.pop("name")
    # optimizer = getattr(torch.optim, optimizer_name)
    # return optimizer(params, lr=cfg.lightning.trainer.lr, **cfg.train.optimizer)
    optimizer_name = cfg.train.optimizer.pop("name")
    optimizer_fn = getattr(torch.optim, optimizer_name)
    optimizer = optimizer_fn(params, lr=cfg.lightning.trainer.lr, **cfg.train.optimizer)
    cfg.train.optimizer.name = optimizer_name
    return optimizer


def build_scheduler(cfg, optimizer):

    if cfg.train.scheduler.name is not None:
        scheduler_name = cfg.train.scheduler.pop("name")
        monitor = cfg.train.scheduler.pop("monitor")
        interval = cfg.train.scheduler.pop("interval")
        frequency = cfg.train.scheduler.pop("frequency")

        if scheduler_name == 'CosineWarmup': 
            # If pretrained, delay for warmup steps to allow randomly initialized head to settle down
            if len(optimizer.param_groups) > 1:     
                ft_lambda_fn = partial(
                    utils.linear_warmup_then_cosine, 
                    delay=cfg.train.scheduler.lr_warmup_steps,
                    warmup=cfg.train.scheduler.lr_warmup_steps, 
                    max_iter=cfg.train.scheduler.lr_decay_step)
                reg_lambda_fn = partial(
                    utils.linear_warmup_then_cosine, 
                    warmup=cfg.train.scheduler.lr_warmup_steps, 
                    max_iter=cfg.train.scheduler.lr_decay_step)
                lr_fns = [ft_lambda_fn, reg_lambda_fn] if cfg.model.pretrained else reg_lambda_fn
            else: 
                lr_fns = partial(
                    utils.linear_warmup_then_cosine, 
                    warmup=cfg.train.scheduler.lr_warmup_steps, 
                    max_iter=cfg.train.scheduler.lr_decay_step)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fns) 
        else: 
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_name)
            scheduler = scheduler_class(optimizer, **cfg.train.scheduler)

        cfg.train.scheduler.name = scheduler_name
        cfg.train.scheduler.monitor = monitor 
        cfg.train.scheduler.interval = interval
        cfg.train.scheduler.frequency = frequency

    else:
        scheduler = None
        monitor = None
        interval = None
        frequency = None

    scheduler = {
        "scheduler": scheduler,
        "monitor": monitor,
        "interval": interval,
        "frequency": frequency,
    }

    return scheduler


def build_loss(cfg):
    # get loss function
    loss_fn_name = cfg.train.loss_fn.pop("name")
    if loss_fn_name == 'BinaryFocalLoss': 
        loss_fn = BinaryFocalLoss
    else: 
        loss_fn = getattr(nn, loss_fn_name)
    loss_function = loss_fn(**cfg.train.loss_fn)
    cfg.train.loss_fn.name = loss_fn_name
    return loss_function


def build_transformation(cfg, split):

    if OmegaConf.is_none(cfg, 'transforms'):
        return None
    elif cfg.data.type in ["3d", "window", "lidc-window", "window_stanford"]: # another exception for lidc-window
        return None

    transforms = []

    if split == "train":
        # handel shift scale rotate
        if "ShiftScaleRotate" in cfg.transforms:
            transforms.append(
                A.ShiftScaleRotate(
                    border_mode=cv2.BORDER_CONSTANT, 
                    **cfg.transforms.ShiftScaleRotate
                )
            )
        for transform_name, arguments in cfg.transforms.items():
            if transform_name == 'type': continue
            transforms.append(getattr(A, transform_name)(**arguments))
    else:
        if "RandomCrop" in cfg.transforms:
            transforms.append(A.CenterCrop(**cfg.transforms.RandomCrop))

    transforms += [ToTensorV2()]
    transforms = A.Compose(transforms)

    return transforms
