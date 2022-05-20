import torch
import torch.nn as nn
from omegaconf import OmegaConf
from . import vision_backbones


class PEModel2D(nn.Module):
    def __init__(self, cfg, num_classes=1, **kwargs):
        super(PEModel2D, self).__init__()

        # define cnn model
        model_function = getattr(vision_backbones, cfg.model.model_name)
        self.model, self.feature_dim = model_function(pretrained=cfg.model.pretrained)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.cfg = cfg

        # freeze cnn
        if cfg.model.freeze_cnn:
            for param in self.model.parameters():
                param.requires_grad = False

        if not OmegaConf.is_none(cfg.model, 'checkpoint') and OmegaConf.is_none(cfg, 'checkpoint'):
            ckpt = torch.load(cfg.model.checkpoint) 

            if 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            elif 'model' in ckpt: 
                state_dict = ckpt['model']
            else: 
                raise Exception('ckpt key incorrect')
        
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            state_dict = {k:v for k,v in state_dict.items() if 'head' not in k}
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            msg = self.model.load_state_dict(state_dict, strict=False)         
            print('='*80)
            print(msg)
            print('='*80)

    def forward(self, x, get_features=False):
        x = self.model(x)
        pred = self.classifier(x)
        if get_features:
            return pred, x
        else:
            return pred

