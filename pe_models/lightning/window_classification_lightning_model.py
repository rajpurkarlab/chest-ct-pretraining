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
from collections import defaultdict


class PEWindowClassificationLightningModel(LightningModule):
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

        # first layer: split
        # second layer: study_2_pred, study_2_label
        # third layer: study list 
        self.results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))


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

        x, y, instance_id = batch
        if self.cfg.data.type == "2d" or self.cfg.data.type == "lidc-2d":
            instance_id = [[i.split("-")[1] for i in instance_id], [i.split("-")[0] for i in instance_id]]
        logit, features = self.model(x, get_features=True)

        if self.loss_weights is not None:
            weight = self.loss_weights.to(y.device)
            loss = self.loss(logit, y).mean(0)
            loss = torch.sum(weight * loss)
        else:
            loss = self.loss(logit, y)

        # map window prediction to study
        batch_pred = torch.sigmoid(logit.clone().detach()).cpu().numpy()
        batch_label = y.clone().detach().cpu().numpy()
        # ALEX TO ADD HERE
        for s,p,l in zip(instance_id[0], batch_pred, batch_label):
            self.results[split]['study_2_pred'][s].append(p)
            self.results[split]['study_2_label'][s].append(l)

        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=True,
            logger=False,
            prog_bar=True,
        )

        return_dict = {"y": y, "loss": loss, "logit": logit.detach(), "instance_id": instance_id, "features": features.cpu().detach()}
        return return_dict

    def shared_epoch_end(self, step_outputs, split):

        instance_id = [ids for x in step_outputs for ids in x["instance_id"]]
        logit = torch.cat([x["logit"] for x in step_outputs])
        y = torch.cat([x["y"] for x in step_outputs])
        prob = torch.sigmoid(logit)
        features = torch.cat([x["features"] for x in step_outputs])

        # log window auroc
        auroc_dict = utils.get_auroc(y, prob, self.target_names)
        for k, v in auroc_dict.items():
            self.log(f"{split}/window_{k}_auroc", v, on_epoch=True, logger=True, prog_bar=True)

        # log window auprc
        auprc_dict = utils.get_auprc(y, prob, self.target_names)
        for k, v in auprc_dict.items():
            self.log(f"{split}/window_{k}_auprc", v, on_epoch=True, logger=True, prog_bar=True)

        # study level prediction (based on max pred)
        study_y, study_prob, study_ids= [], [], []
        for study in self.results[split]['study_2_label'].keys():
            label = 1 if 1 in self.results[split]['study_2_label'][study] else 0
            pred = max(self.results[split]['study_2_pred'][study])
            study_y.append(label)
            study_prob.append(pred)
            study_ids.append(study)
        study_y = np.expand_dims(np.array(study_y), -1)
        study_prob = np.array(study_prob)

        # study level metrics
        auroc_dict = utils.get_auroc(study_y, study_prob, self.target_names)
        for k, v in auroc_dict.items():
            self.log(f"{split}/{k}_auroc", v, on_epoch=True, logger=True, prog_bar=True)
        auprc_dict = utils.get_auprc(study_y, study_prob, self.target_names)
        for k, v in auprc_dict.items():
            self.log(f"{split}/{k}_auprc", v, on_epoch=True, logger=True, prog_bar=True)

        # reset results
        self.results[split] = defaultdict(lambda: defaultdict(list))

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
            y = [a[0] for a in list(y.cpu().detach().numpy())]
            prob = [a[0] for a in list(prob.cpu().detach().numpy())]
            study_id = [item for sublist in instance_id[::2] for item in sublist]
            if self.cfg.data.type == '2d' or self.cfg.data.type == 'lidc-2d':
                window_idx = [item for sublist in instance_id[1::2] for item in sublist]
            else:
                window_idx = [item for sublist in instance_id[1::2] for item in list(sublist.cpu().numpy())]

            prediction_dict = {
                "target": y,
                "prob": prob,
                "study_id": study_id,
                "window_idx": window_idx
            }
            prediction_path = os.path.join(self.cfg.output_dir, "preds.csv")
            #pickle.dump(prediction_dict, open(prediction_path, "wb"))
            pred_df = pd.DataFrame.from_dict(prediction_dict)
            pred_df.to_csv(prediction_path)
            print(f"\nPredictions saved at: {prediction_path}")
            
            
            # save features
            if self.cfg.data.type == '2d' or self.cfg.data.type == 'lidc-2d':
                _train_csv = RSNA_TRAIN_CSV if self.cfg.data.type == '2d' else LIDC_TRAIN_CSV
                _instance_col = RSNA_INSTANCE_COL if self.cfg.data.type == '2d' else LIDC_INSTANCE_COL
                _study_col = RSNA_STUDY_COL if self.cfg.data.type == '2d' else LIDC_STUDY_COL
                
                df_full = pd.read_csv(_train_csv)
                prediction_df = pred_df.copy()
                prediction_df = prediction_df.drop("study_id", axis=1)
                prediction_df.columns = ['target', 'prob', _instance_col]
                prediction_df = prediction_df.set_index(_instance_col)
                df_full = df_full.set_index(_instance_col) 
                df = prediction_df.join(df_full).reset_index()

                print('Instance-level performance') 
                print('--------------------------')
                for split in ['train', 'valid', 'test']: # these are not the appropriate split names
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
                instance_study_id = [f"{x}-{y}" for x, y in zip(window_idx, study_id)]
                id2feature = {k:v.numpy() for k,v in zip(instance_study_id, features)}

                features_path = os.path.join(self.cfg.output_dir, "features.hdf5")
                hdf5_fn = h5py.File(features_path, 'a')
                for study, grouped_df in df: 
                    _image_order_col = "InstanceOrder" if self.cfg.data.type == '2d' else "image_index"
                    grouped_df = grouped_df.sort_values(by=[_image_order_col])
                    features = np.stack([
                        id2feature[ids] for ids in (grouped_df[_instance_col] + "-" + study).tolist()
                    ])
                    hdf5_fn.create_dataset(study, data=features, dtype='float32', chunks=True)
                hdf5_fn.close()
                print(f"\nFeatures saved at: {features_path}")
