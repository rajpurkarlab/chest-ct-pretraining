################################################################################
# TODO: need testing
################################################################################
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Adapted from:
        https://github.com/GuanshuoXu/RSNA-STR-Pulmonary-Embolism-Detection/blob/main/trainall/2nd_level/seresnext101_192.py
    """

    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        print("=" * 80)
        print("Using attention")
        print("=" * 80)

        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(x.contiguous().view(-1, feature_dim), self.weight).view(
            -1, step_dim
        )

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * torch.unsqueeze(a, -1)

        return torch.sum(weighted_input, 1), self.weight


class RNNSequentialEncoder(nn.Module):
    """Model to encode series of encoded 2D CT slices using RNN

    Args:
        feature_size (int): number of features for input feature vector
        rnn_type (str): either lstm or gru
        hidden_size (int): number of hidden units
        bidirectional (bool): use bidirectional rnn
        num_layers (int): number of rnn layers
        dropout_prob (float): dropout probability
    """

    def __init__(
        self,
        feature_size: int,
        rnn_type: str = "lstm",
        hidden_size: int = 128,
        bidirectional: bool = True,
        num_layers: int = 1,
        dropout_prob: float = 0.0,
    ):

        super(RNNSequentialEncoder, self).__init__()

        self.feature_size = feature_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers

        if self.rnn_type not in ["LSTM", "GRU"]:
            raise Exception("RNN type has to be either LSTM or GRU")

        self.rnn = getattr(nn, rnn_type)(
            self.feature_size,
            self.hidden_size,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=self.dropout_prob,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        x = x.transpose(0, 1)
        x, _ = self.rnn(x)  # (Slice, Batch, Feature)
        x = x.transpose(0, 1)  # (Batch, Slice, Feature)
        return x


class PEModel1D(nn.Module):
    def __init__(self, cfg, num_classes=1):
        super(PEModel1D, self).__init__()

        # rnn input size 
        seq_input_size = cfg.data.feature_size 
        if cfg.data.contextualize_slice:
            seq_input_size = seq_input_size * 3

        # classifier input size 
        cls_input_size = cfg.model.seq_encoder.hidden_size
        if cfg.model.seq_encoder.bidirectional:
            cls_input_size = cls_input_size * 2

        self.seq_encoder = RNNSequentialEncoder(seq_input_size, **cfg.model.seq_encoder)

        if "attention" in cfg.model.aggregation:
            self.attention = Attention(cls_input_size, cfg.data.num_slices)

        self.batch_norm_layer = torch.nn.BatchNorm1d(cls_input_size)
        self.classifier = nn.Linear(cls_input_size, num_classes)
        self.cfg = cfg

    def forward(self, x, get_features=False):
        x = self.seq_encoder(x)
        x, w = self.aggregate(x)
        x = self.batch_norm_layer(x)
        pred = self.classifier(x)
        return pred, x

    def aggregate(self, x):

        if self.cfg.model.aggregation == "attention":
            return self.attention(x)
        elif self.cfg.model.aggregation == "mean":
            x = torch.mean(x, 1)
            return x, None
        elif self.cfg.model.aggregation == "max":
            x, _ = torch.max(x, 1)
            return x, None
        else:
            raise Exception(
                "Aggregation method should be one of 'attention', 'mean' or 'max'"
            )
