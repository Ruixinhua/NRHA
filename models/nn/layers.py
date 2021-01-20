import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class MultiHeadedAttention(nn.Module):
    """
    MultiheadedAttention:

    http://nlp.seas.harvard.edu/2018/04/03/attention.html#attention
    """

    def __init__(self, h, d_k, word_dim, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        # We assume d_v always equals d_k
        self.d_k = d_k
        self.h = h
        d_model = h * d_k
        self.linears = clones(nn.Linear(word_dim, d_model), 3)
        self.final = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [liner(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for liner, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.final(x), self.attn


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        shape = torch.Size([-1]) + x.size()[2:]
        x_reshape = x.contiguous().view(shape)  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class AttLayer(nn.Module):

    def __init__(self, word_emb_dim, attention_hidden_dim):
        super().__init__()
        # build attention network
        self.attention = nn.Sequential(
            nn.Linear(word_emb_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
            nn.Flatten(),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        attention_weight = torch.unsqueeze(self.attention(x), 2)
        y = torch.sum(x * attention_weight, dim=1)
        return y, attention_weight


class LayerNorm(nn.Module):
    """
    Construct a layernorm module (See citation for details).
    https://arxiv.org/abs/1607.06450
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MLP(nn.Module):
    """Multilayer perceptron torch module.
    Authors: Hongwei Zhang
    Parameters
    ----------
    input_size : int
        Size of packed_input.

    hidden_layers : iterable
        Hidden layer sizes.

    dropout : float
        Dropout rate.

    activation : str
        Name of activation function. ReLU, PReLU and Sigmoid are supported.
    """
    def __init__(self, input_size, hidden_layers,
                 dropout=0.0, batch_norm=True, activation='relu'):
        super(MLP, self).__init__()
        modules = OrderedDict()

        previous_size = input_size
        for index, hidden_layer in enumerate(hidden_layers):
            modules[f"dense{index}"] = nn.Linear(previous_size, hidden_layer)
            if batch_norm:
                modules[f"batchnorm{index}"] = nn.BatchNorm1d(hidden_layer)
            if activation:
                if activation.lower() == 'relu':
                    modules[f"activation{index}"] = nn.ReLU()
                elif activation.lower() == 'prelu':
                    modules[f"activation{index}"] = nn.PReLU()
                elif activation.lower() == 'sigmoid':
                    modules[f"activation{index}"] = nn.Sigmoid()
                else:
                    raise NotImplementedError(f"{activation} is not supported")
            if dropout:
                modules[f"dropout{index}"] = nn.Dropout(dropout)
            previous_size = hidden_layer
        self._sequential = nn.Sequential(modules)

    def forward(self, x):
        return self._sequential(x)
