import numpy as np
import torch
import torch.nn.functional as F
import wandb
import json
import pandas as pd
import pickle
import os
import h5py 

from .. import builder
from .. import utils
from ..constants import *
from sklearn.metrics import average_precision_score, roc_auc_score
from pytorch_lightning.core import LightningModule


class PEClassificationLightningModel(LightningModule):
    """Pytorch-Lightning Module"""

    def __init__(self, cfg):
        """Pass in hyperparameters to the model"""
        # initalize superclass
        super().__init__()

        self.cfg = cfg
        self.model = builder.build_model(cfg)
        self.loss = builder.build_loss(cfg)
        self.target_names = RSNA_TARGET_TYPES[cfg.data.targets]

        if self.cfg.train.weighted_loss:
            print("Using weighted loss")
            self.loss_weights = torch.tensor(
                [RSNA_LOSS_WEIGHT[t] for t in self.target_names]
            )
        else:
            self.loss_weights = None

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.model)
        scheduler = builder.build_scheduler(self.cfg, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def training_epoch_end(self, training_step_outputs):
        return self.shared_epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        return self.shared_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        return self.shared_epoch_end(test_step_outputs, "test")

    def shared_step(self, batch, split, extract_features=False):
        """Similar to traning step"""

        #x, y, instance_id, _ = batch
        x, y, instance_id = batch
        logit, features = self.model(x, get_features=True)

        if self.loss_weights is not None:
            weight = self.loss_weights.to(y.device)
            loss = self.loss(logit, y).mean(0)
            loss = torch.sum(weight * loss)
        else:
            loss = self.loss(logit, y)

        self.log( f"{split}_loss", loss, on_epoch=True, on_step=True, logger=False, prog_bar=True,)

        return_dict = {"y": y, "loss": loss, "logit": logit, "instance_id": instance_id, "features": features.cpu().detach()}
        return return_dict

    def shared_epoch_end(self, step_outputs, split):

        instance_id = [ids for x in step_outputs for ids in x["instance_id"]]
        logit = torch.cat([x["logit"] for x in step_outputs])
        y = torch.cat([x["y"] for x in step_outputs])
        prob = torch.sigmoid(logit)
        features = torch.cat([x["features"] for x in step_outputs])
        
        if '-' in instance_id[0]:
            study_id  = [i.split('-')[1] for i in instance_id]
            instance_id  = [i.split('-')[0] for i in instance_id]

        # log auroc
        auroc_dict = utils.get_auroc(y, prob, self.target_names)
        for k, v in auroc_dict.items():
            self.log(f"{split}/{k}_auroc", v, on_epoch=True, logger=True, prog_bar=True)

        # log auprc
        auprc_dict = utils.get_auprc(y, prob, self.target_names)
        for k, v in auprc_dict.items():
            self.log(f"{split}/{k}_auprc", v, on_epoch=True, logger=True, prog_bar=True)

        if split == "test":
            # save results
            meta_dict = {"split": split}
            if not os.path.exists(self.cfg.output_dir): # I added this so it stops crashing
                os.makedirs(self.cfg.output_dir)
            results_csv = os.path.join(self.cfg.output_dir, "results.csv")
            auroc_dict = {f"{split}/{k}_auroc": [v] for k, v in auroc_dict.items()}
            auprc_dict = {f"{split}/{k}_auprc": [v] for k, v in auprc_dict.items()}
            results = {**meta_dict, **auroc_dict, **auprc_dict}

            results_df = pd.DataFrame.from_dict(results, orient="columns")
            if os.path.exists(results_csv):
                df = pd.read_csv(results_csv)
                results_df = pd.concat([df, results_df], ignore_index=True)
            results_df.to_csv(results_csv, index=False)
            print(f"\nResults saved at: {results_csv}")

            # save predictions
            y = [x[0] for x in list(y.cpu().detach().numpy())]
            prob = [x[0] for x in list(prob.cpu().detach().numpy())]
            prediction_dict = {
                "target": y,
                "prob": prob,
                "id": instance_id,
            }
            prediction_df = pd.DataFrame(prediction_dict)
            prediction_path = os.path.join(self.cfg.output_dir, "preds.csv")
            prediction_df.to_csv(prediction_path, index=False)
            print(f"\nPredictions saved at: {prediction_path}")

            # save features
            if self.cfg.data.type == '2d' or self.cfg.data.type == 'lidc-2d':
                _train_csv = RSNA_TRAIN_CSV if self.cfg.data.type == '2d' else LIDC_TRAIN_CSV
                _instance_col = RSNA_INSTANCE_COL if self.cfg.data.type == '2d' else LIDC_INSTANCE_COL
                _study_col = RSNA_STUDY_COL if self.cfg.data.type == '2d' else LIDC_STUDY_COL
                
                df_full = pd.read_csv(_train_csv)
                prediction_df.columns = ['target', 'prob', _instance_col]
                prediction_df = prediction_df.set_index(_instance_col)
                df_full = df_full.set_index(_instance_col) 
                df = prediction_df.join(df_full).reset_index()

                print('Instance-level performance') 
                print('--------------------------')
                #for split in ['train', 'valid', 'test']:
                for split in ['test']:
                    print(f'{split} AUROC: {roc_auc_score(df[df.Split == split].target, df[df.Split == split].prob)}')
                    print(f'{split} AUPRC: {average_precision_score(df[df.Split == split].target, df[df.Split == split].prob)}')                                                                               

                print('\n\n') 
                print('Study-level performance') 
                print('-----------------------')
                study_df = df[[_study_col, 'target', 'prob', 'Split']].groupby(_study_col).max().reset_index()                                                                               
                for split in ['train', 'valid', 'test']:
                    print(f'{split} AUROC: {roc_auc_score(study_df[study_df.Split == split].target, study_df[study_df.Split == split].prob)}') 
                    print(f'{split} AUROC: {average_precision_score(study_df[study_df.Split == split].target, study_df[study_df.Split == split].prob)}') 

                df = df.groupby(_study_col)

                id2feature = {k:v.numpy() for k,v in zip(instance_id, features)}

                features_path = os.path.join(self.cfg.output_dir, "features.hdf5")
                hdf5_fn = h5py.File(features_path, 'a')
                for study, grouped_df in df: 
                    grouped_df = grouped_df.sort_values(by=['InstanceOrder'])
                    features = np.stack([id2feature[ids] for ids in grouped_df[_instance_col].tolist()])
                    hdf5_fn.create_dataset(study, data=features, dtype='float32', chunks=True)
                hdf5_fn.close()
                print(f"\nFeatures saved at: {features_path}")
