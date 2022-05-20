import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Focal loss for binary classification.
    Adapted from:
        https://gist.github.com/AdrienLE/bf31dfe94569319f6e47b2de8df13416#file-focal_dice_1-py
    """
    def __init__(self, gamma=2, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.take_mean = size_average

    def forward(self, logits, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == logits.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), logits.size()))

        max_val = (-logits).clamp(min=0)
        loss = logits - logits * target + max_val + ((-max_val).exp() + (-logits - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        inv_probs = F.logsigmoid(-logits * (target * 2 - 1))
        loss = (inv_probs * self.gamma).exp() * loss

        if self.take_mean:
            loss = loss.mean()

        return loss


class DINOLoss(nn.Module):
    """
    DINO loss, adapted from: 
        https://github.com/facebookresearch/dino/blob/main/main_dino.py
    """
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, verbose=True):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
                # TODO
                if not verbose: 
                    print('-------------  Loss Funcion -------------')
                    print(f'loss = {loss}')
                    print(f'total_loss = {total_loss}')
                    print(f'n loss = {n_loss_terms}') 
                    print(f'q = {q}')
                    print(f'{F.log_softmax(student_out[v], dim=-1)}')
                    print(f'{student_out[v]}')
                    import pdb; pdb.set_trace()
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        # TODO 
        if not verbose: 
            print(total_loss)
            print(n_loss_terms) 

        return total_loss, n_loss_terms

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        #dist.all_reduce(batch_center)
        #batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        batch_center = batch_center / (len(teacher_output))

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)