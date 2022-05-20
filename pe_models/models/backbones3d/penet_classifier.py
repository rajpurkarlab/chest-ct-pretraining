"""
Adapted from: https://github.com/marshuang80/penet/blob/master/models/penet_classifier.py
"""

import math
import torch
import torch.nn as nn

from .layers.penet import *


class PENetClassifier(nn.Module):
    """PENet stripped down for classification.
    The idea is to pre-train this network, then use the pre-trained
    weights for the encoder in a full PENet.
    """

    def __init__(self, cfg, **kwargs):
        super(PENetClassifier, self).__init__()

        self.in_channels = 64
        self.model_depth = cfg.model.model_depth
        self.cardinality = cfg.model.cardinality
        self.num_channels = cfg.model.num_channels
        self.num_classes = cfg.model.num_classes

        self.in_conv = nn.Sequential(nn.Conv3d(self.num_channels, self.in_channels, kernel_size=7,
                                               stride=(1, 2, 2), padding=(3, 3, 3), bias=False),
                                     nn.GroupNorm(self.in_channels // 16, self.in_channels),
                                     nn.LeakyReLU(inplace=True))
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # Encoders
        if cfg.model.model_depth != 50:
            raise ValueError('Unsupported model depth: {}'.format(cfg.model.model_depth))
        encoder_config = [3, 4, 6, 3]
        total_blocks = sum(encoder_config)
        block_idx = 0

        self.encoders = nn.ModuleList()
        for i, num_blocks in enumerate(encoder_config):
            out_channels = 2 ** i * 128
            stride = 1 if i == 0 else 2
            encoder = PENetEncoder(self.in_channels, out_channels, num_blocks, self.cardinality,
                                  block_idx, total_blocks, stride=stride)
            self.encoders.append(encoder)
            self.in_channels = out_channels * PENetBottleneck.expansion
            block_idx += num_blocks

        self.classifier = GAPLinear(self.in_channels, cfg.model.num_classes)

        if cfg.model.init_method is not None:
            self._initialize_weights(cfg.model.init_method, focal_pi=0.01)

    def _initialize_weights(self, init_method, gain=0.2, focal_pi=None):
        """Initialize all weights in the network."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Linear):
                if init_method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=gain)
                elif init_method == 'xavier':
                    nn.init.xavier_normal_(m.weight, gain=gain)
                elif init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)
                else:
                    raise NotImplementedError('Invalid initialization method: {}'.format(self.init_method))
                if hasattr(m, 'bias') and m.bias is not None:
                    if focal_pi is not None and hasattr(m, 'is_output_head') and m.is_output_head:
                        # Focal loss prior (~0.01 prob for positive, see RetinaNet Section 4.1)
                        nn.init.constant_(m.bias, -math.log((1 - focal_pi) / focal_pi))
                    else:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm) and m.affine:
                # Gamma for last GroupNorm in each residual block gets set to 0
                init_gamma = 0 if hasattr(m, 'is_last_norm') and m.is_last_norm else 1
                nn.init.constant_(m.weight, init_gamma)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # Expand input (allows pre-training on RGB videos, fine-tuning on Hounsfield Units)
        if x.size(1) < self.num_channels:
            x = x.expand(-1, self.num_channels // x.size(1), -1, -1, -1)

        x = self.in_conv(x)
        x = self.max_pool(x)

        # Encoders
        for encoder in self.encoders:
            x = encoder(x)

        # Classifier
        x = self.classifier(x)

        return x

    def args_dict(self):
        """Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `PENet(**model_args)`.
        """
        model_args = {
            'model_depth': self.model_depth,
            'cardinality': self.cardinality,
            'num_classes': self.num_classes,
            'num_channels': self.num_channels
        }

        return model_args

    def load_pretrained(self, ckpt_path):
        """Load parameters from a pre-trained PENetClassifier from checkpoint at ckpt_path.
        Args:
            ckpt_path: Path to checkpoint for PENetClassifier.
        Adapted from:
            https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
        """
        try: 
            pretrained_dict = torch.load(ckpt_path)['model_state']
            model_dict = self.state_dict()

            # Filter out unnecessary keys
            pretrained_dict = {k[len('module.'):]: v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        except: 
            pretrained_dict = torch.load(ckpt_path)['state_dict']
            model_dict = self.state_dict()
            pretrained_dict = {k[len('module.'):]: v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['classifier.fc.weight', 'classifier.fc.bias']}


        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # Load the new state dict
        msg = self.load_state_dict(model_dict, strict=False)
        print(msg)

