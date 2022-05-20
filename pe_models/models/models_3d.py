import torch
import torch.nn as nn

from . import vision_backbones


class PEModel3D(nn.Module):
    def __init__(self, cfg, num_classes=1, **kwargs):
        super(PEModel3D, self).__init__()

        # define cnn model
        model_function = getattr(vision_backbones, cfg.model.model_name)
        self.model, self.feature_dim = model_function(
            pretrained=cfg.model.pretrained, num_frames=cfg.data.num_slices,
            cfg=cfg
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.cfg = cfg

        # freeze cnn
        if cfg.model.freeze_cnn:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, get_features=False):
        x = self.model(x)
        pred = self.classifier(x)

        if get_features:
            return pred, x
        else:
            return pred

    def fine_tuning_parameters(self, fine_tuning_boundary, fine_tuning_lr=0.0):
        """Get parameters for fine-tuning the model.
        Args:
            fine_tuning_boundary: Name of first layer after the fine-tuning layers.
            fine_tuning_lr: Learning rate to apply to fine-tuning layers (all layers before `boundary_layer`).
        Returns:
            List of dicts that can be passed to an optimizer.
        """

        def gen_params(boundary_layer_name, fine_tuning):
            """Generate parameters, if fine_tuning generate the params before boundary_layer_name.
            If unfrozen, generate the params at boundary_layer_name and beyond."""
            saw_boundary_layer = False
            for name, param in self.named_parameters():
                if name.startswith(boundary_layer_name):
                    saw_boundary_layer = True

                if saw_boundary_layer and fine_tuning:
                    return
                elif not saw_boundary_layer and not fine_tuning:
                    continue
                else:
                    yield param

        # Fine-tune the network's layers from encoder.2 onwards
        optimizer_parameters = [{'params': gen_params(fine_tuning_boundary, fine_tuning=True), 'lr': fine_tuning_lr},
                                {'params': gen_params(fine_tuning_boundary, fine_tuning=False)}]

        return optimizer_parameters 
