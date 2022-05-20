import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import timm

from functools import partial
from .backbones3d import resnet3d, penet_classifier
# import sys
# sys.path.append("/deep/u/alexke/pe_models_benchmark/pe_models/")
from ..utils import download_file_from_google_drive
from .backbones2d import vit
from torchvision import models as models_2d
from omegaconf import OmegaConf

from .backbones3d.s3dg import S3D
from .backbones3d.resnet_2d3d import r2d3d50
from .backbones3d.r21d import R2Plus1DNet
from .backbones3d import video_swin_transformer as vst
from collections import OrderedDict


class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x
    
################################################################################
# Inception Family
################################################################################

def inception_v4(pretrained=True, **kwargs):
    model = timm.models.inception_v4(pretrained=pretrained)
    feature_dims = model.last_linear.in_features
    model.last_linear = Identity()
    return model, feature_dims

def inception_v3(pretrained=True, **kwargs):
    model = timm.models.inception_v3(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims


################################################################################
# EfficientNet Family
################################################################################
def efficientnet_b0(pretrained=True, **kwargs):
    model = timm.models.efficientnet.efficientnet_b0(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims

def efficientnet_b3(pretrained=True, **kwargs):
    model = timm.models.efficientnet.efficientnet_b3(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims

def efficientnet_b4(pretrained=True, **kwargs):
    model = timm.models.efficientnet.efficientnet_b4(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims

def efficientnet_b7(pretrained=True, **kwargs):
    model = timm.models.efficientnet.efficientnet_b7(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims

def efficientnet_b8(pretrained=True, **kwargs):
    model = timm.models.efficientnet.efficientnet_b8(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims


################################################################################
# ResNet Family
################################################################################

def resnet_18(pretrained=True):
    model = models_2d.resnet18(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims


def resnet_50(pretrained=True):
    model = models_2d.resnet50(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims


def resnet_101(pretrained=True):
    model = models_2d.resnet101(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims


################################################################################
# DenseNet Family
################################################################################

def densenet_121(pretrained=True):
    model = models_2d.densenet121(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims


def densenet_161(pretrained=True):
    model = models_2d.densenet161(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims


def densenet_169(pretrained=True):
    model = models_2d.densenet169(pretrained=pretrained)
    feature_dims = model.classifier.in_features
    model.classifier = Identity()
    return model, feature_dims


################################################################################
# ResNextNet Family
################################################################################

def resnext_50(pretrained=True):
    model = models_2d.resnext50_32x4d(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims


def resnext_101(pretrained=True):
    model = models_2d.resnext101_32x8d(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims


################################################################################
# Resnest family
################################################################################

def resnest_101(pretrained=True, **kwargs):
    model = timm.models.resnest.resnest101e(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims


def resnest_200(pretrained=True, **kwargs):
    model = timm.models.resnest.resnest200e(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims


def resnest_269(pretrained=True, **kwargs):
    model = timm.models.resnest.resnest269e(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims


################################################################################
# ResNet3D family
################################################################################

def resnet_3d_18(pretrained=True, **kwargs):
    model = resnet3d.generate_model(18, n_classes=1039)

    if pretrained:
        ckpt_path = "./data/ckpt/r3d18_KM_200ep.pth"
        if not os.path.isfile(ckpt_path):
            download_file_from_google_drive(
                "12FxrQY2hX-bINbmSrN9q2Z5zJguJhy6C", ckpt_path
            )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["state_dict"])

    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims


def resnet_3d_50(pretrained=True, **kwargs):
    model = resnet3d.generate_model(50, n_classes=1039)

    if pretrained:
        ckpt_path = "./data/ckpt/r3d50_KM_200ep.pth"
        if not os.path.isfile(ckpt_path):
            download_file_from_google_drive(
                "1fCKSlakRJ54b3pEWqgBmuJi0nF7HXQc0", ckpt_path
            )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["state_dict"])

    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims


def resnet_3d_101(pretrained=True, **kwargs):
    model = resnet3d.generate_model(101, n_classes=1039)

    if pretrained:
        ckpt_path = "./data/ckpt/r3d101_KM_200ep.pth"
        if not os.path.isfile(ckpt_path):
            download_file_from_google_drive(
                "1p80RJsghFIKBSLKgtRG94LE38OGY5h4y", ckpt_path
            )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["state_dict"])

    feature_dims = model.fc.in_features
    model.fc = Identity()
    return model, feature_dims


################################################################################
# Vision Transformers
################################################################################

def vit_small(patch_size=16, **kwargs):
    model = vit.VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.head = Identity()
    return model, model.embed_dim

def vit_small_16(pretrained=True, **kwargs):
    if pretrained: 
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
    else: 
        model = vit.VisionTransformer(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.head  = Identity() 
    return model, model.embed_dim

def vit_small_8(pretrained=True, **kwargs):
    if pretrained: 
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    else: 
        model = vit.VisionTransformer(
            patch_size=8, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.head  = Identity() 
    return model, model.embed_dim


def vit_base(patch_size=16, mask_ratio=None, **kwargs):
    model = vit.VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), mask_ratio=mask_ratio,
        **kwargs)
    model.head  = Identity() 
    return model, model.embed_dim

def vit_base_16(pretrained=True, **kwargs):
    if pretrained: 
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    else: 
        model = vit.VisionTransformer(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.head  = Identity() 
    return model, model.embed_dim

def vit_base_8(pretrained=True, **kwargs):
    if pretrained: 
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
    else: 
        model = vit.VisionTransformer(
            patch_size=8, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.head  = Identity() 
    return model, model.embed_dim

################################################################################
# PENet  
################################################################################

def penet(pretrained=True, **kwargs):
    model = penet_classifier.PENetClassifier(kwargs['cfg'])

    if pretrained:

        if OmegaConf.is_none(kwargs['cfg'].model, 'checkpoint'):
        
            ckpt_path = './data/ckpt/xnet_kin_90.pth.tar'
            if not os.path.isfile(ckpt_path):
                download_file_from_google_drive(
                     "1IyMmgjr3Wk2oyqKkv4_BjATJm_17vu5M", ckpt_path
                )
        else: 
            ckpt_path = kwargs['cfg'].model.checkpoint
        print(ckpt_path)
        ckpt = torch.load(ckpt_path)
        model.load_pretrained(ckpt_path)
        print('loaded pretrained models')

    feature_dims = model.classifier.fc.in_features
    model.classifier.fc = Identity()
    return model, feature_dims


################################################################################
# X3D  
################################################################################

def x3d(pretrained=True, **kwargs):
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load("facebookresearch/pytorchvideo", model='x3d_m', pretrained=pretrained)
    feature_dims = model._modules['blocks'][-1].proj.in_features
    model._modules['blocks'][-1].proj = Identity()
    return model, feature_dims


################################################################################
# MVIT
################################################################################

def mvit(pretrained=True, **kwargs):
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    model = torch.hub.load("facebookresearch/pytorchvideo", model='mvit_base_32x3', pretrained=pretrained)
    feature_dims = model.head.proj.in_features
    model.head.proj = Identity()
    return model, feature_dims


################################################################################
# CoCLR and VICC
################################################################################

def select_backbone(network, first_channel=3):
    param = {'feature_size': 1024}
    if network == 's3d':
        model = S3D(input_channel=first_channel)
    elif network == 's3dg':
        model = S3D(input_channel=first_channel, gating=True)
    elif network == 'r50':
        param['feature_size'] = 2048
        model = r2d3d50(input_channel=first_channel)
    elif network == 'r21d':
        param['feature_size'] = 512
        model = R2Plus1DNet()
    else: 
        raise NotImplementedError

    return model, param

class LinearClassifier(nn.Module):
    def __init__(self, num_class=101, 
                 network='resnet50', 
                 dropout=0.5, 
                 use_dropout=True, 
                 use_l2_norm=False,
                 use_final_bn=False):
        super(LinearClassifier, self).__init__()
        self.network = network
        self.num_class = num_class
        self.dropout = dropout
        self.use_dropout = use_dropout
        self.use_l2_norm = use_l2_norm
        self.use_final_bn = use_final_bn
        
        message = 'Classifier to %d classes with %s backbone;' % (num_class, network)
        if use_dropout: message += ' + dropout %f' % dropout
        if use_l2_norm: message += ' + L2Norm'
        if use_final_bn: message += ' + final BN'
        print(message)

        self.backbone, self.param = select_backbone(network)
        
        if use_final_bn:
            self.final_bn = nn.BatchNorm1d(self.param['feature_size'])
            self.final_bn.weight.data.fill_(1)
            self.final_bn.bias.data.zero_()
        
        if use_dropout:
            self.final_fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.param['feature_size'], self.num_class))
        else:
            self.final_fc = nn.Sequential(
                nn.Linear(self.param['feature_size'], self.num_class))
        self._initialize_weights(self.final_fc)
        
    def forward(self, block):
        (B, C, T, H, W) = block.shape
        feat3d = self.backbone(block)
        feat3d = F.adaptive_avg_pool3d(feat3d, (1,1,1)) # [B,C,1,1,1]
        feat3d = feat3d.view(B, self.param['feature_size']) # [B,C]

#         if self.use_l2_norm:
#             feat3d = F.normalize(feat3d, p=2, dim=1)
        
#         if self.use_final _bn:
#             logit = self.final_fc(self.final_bn(feat3d))
#         else:
#             logit = self.final_fc(feat3d)

        # return logit, feat3d
        return feat3d

    def _initialize_weights(self, module):
        for name, param in module.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.01)


def coclr_vicc_both(
    pretrain_path,
    pretrained=True,
    net="s3d",
    **kwargs,
):
    model = LinearClassifier(
                network=net, 
    )
                # num_class=args.num_class,
                # dropout=args.dropout,
                # use_dropout=args.use_dropout,
                # use_final_bn=args.final_bn,
                # use_l2_norm=args.final_norm)
    model.final_fc = Identity()

    if pretrained:
        ckpt = torch.load(pretrain_path, map_location="cpu")
        state_dict = ckpt["state_dict"]

        new_dict = {}
        for k,v in state_dict.items():
            k = k.replace('encoder_q.0.', 'backbone.')
            new_dict[k] = v
        state_dict = new_dict

        model.load_state_dict(state_dict, strict=False)
        print("=> loaded pretrained checkpoint '{}' (epoch {})".format(pretrain_path, ckpt['epoch']))
    
    feature_dims = model.param["feature_size"]
    return model, feature_dims


def coclr(*args, **kwargs): # seems weights aren't being loaded fully?
    ckpt_path = './data/ckpt/CoCLR-k400-rgb-128-s3d.pth.tar'
    if not os.path.isfile(ckpt_path):
        download_file_from_google_drive(
             "1_1HXcoQMpbRQQxvXi21_ytFuu5waEbUt", ckpt_path
        )
    return coclr_vicc_both(ckpt_path, *args, **kwargs)


def vicc(*args, **kwargs):
    ckpt_path = './data/ckpt/ViCC-single-stream-ucf101-RGB-s3d-epoch299.pth.tar'
    if not os.path.isfile(ckpt_path):
        download_file_from_google_drive(
             "1jHxjF4OmWECYIOc6KWG2F94uLYf1tPpK", ckpt_path
        )
    return coclr_vicc_both(ckpt_path, *args, **kwargs)


################################################################################
# Swin Transformer
################################################################################

def swin_transformer_3d(pretrained=True, **kwargs):
    model = vst.SwinTransformer3D( # this is the Swin-T config
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        patch_size=(2, 4, 4),
        window_size=(8, 7, 7), # pretty sure
        drop_path_rate=0.1,
    )
    
    if pretrained:
        ckpt_path = './data/ckpt/swin_tiny_patch244_window877_kinetics400_1k.pth'
        if not os.path.isfile(ckpt_path):
            download_file_from_google_drive(
                 "1OcZSYI1QFaScITWKQtncMEVtz0l0m189", ckpt_path
            )
        checkpoint = torch.load(ckpt_path)

        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'backbone' in k:
                name = k[9:]
                new_state_dict[name] = v 
        model.load_state_dict(new_state_dict, strict=False)
    
    i3dhead = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Dropout(p=0.5),
        nn.Flatten(),
        Identity(),
    )
    
    return nn.Sequential(model, i3dhead), 768


################################################################################
# Pytorch video models
################################################################################

def load_pytorchvideo_model(name, pretrained=True):
    return torch.hub.load("facebookresearch/pytorchvideo", model=name, pretrained=pretrained)


class PackPathway(nn.Module):
    def __init__(self):
        self.alpha = 4
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            2,
            torch.linspace(
                0, frames.shape[2] - 1, frames.shape[2] // self.alpha
            ).long().to(frames.device),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def slowfast_r50(pretrained=True, **kwargs):
    model = load_pytorchvideo_model("slowfast_r50", pretrained=pretrained)
    model.blocks[6].proj = Identity()
    model = nn.Sequential(PackPathway(), model)
    return model, 2304


def slow_r50(pretrained=True, **kwargs):
    model = load_pytorchvideo_model("slow_r50", pretrained=pretrained)
    model.blocks[5].proj = Identity()
    return model, 2048


def r2plus1d_r50(pretrained=True, **kwargs):
    model = load_pytorchvideo_model("r2plus1d_r50", pretrained=pretrained)
    model.blocks[5].proj = Identity()
    model.blocks[5].activation = Identity()
    return model, 2048


def csn_r101(pretrained=True, **kwargs):
    model = load_pytorchvideo_model("csn_r101", pretrained=pretrained)
    model.blocks[5].proj = Identity()
    return model, 2048
