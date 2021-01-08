# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import copy

import torch
import torch.nn.functional as F


__all__ = ["NRHATitle"]

import torch.nn as nn
from models.base import BaseModel
from models.layers import AttLayer


class NRHATitle(BaseModel):

    def clones(self, module, N):
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def __init__(self, hparams):
        super().__init__(hparams)
        self.attentions = self.clones(AttLayer(hparams.word_emb_dim // hparams.head_num, hparams.attention_hidden_dim),
                                      hparams.head_num)

    def sentence_encoder(self, y):
        # shape of y: [N, S, E]     N: batch size, S: sequence length, E: embedded size
        y = y.transpose(0, 1)
        y = self.news_self_att(y, y, y)[0].transpose(0, 1)
        y = F.dropout(y, p=self.hparams.dropout)
        # shape of y: [N, S, E]
        y = y.view(y.shape[0], y.shape[1], self.hparams.head_num, -1)
        # shape of y: [N, S, H, D]     E is head_num * head_dim, E = H * D
        y = y.transpose(1, 2).transpose(0, 1)
        # shape of y: [H, N, S, D]
        y = torch.stack([self_att(h)[0] for h, self_att in zip(y, self.attentions)])
        y = y.transpose(0, 1)
        # shape of y: [N, H, D]
        y = y.reshape(y.shape[0], self.hparams.word_emb_dim)
        # shape of y: [N, E]
        return y

    def news_encoder(self, sequences_input):
        y = self._embedding_layer(sequences_input)
        y = self.sentence_encoder(y)
        # shape of q: [N, S, E]     N: batch size, S: sequence length, E: embedded size
        # y = self.news_att_layer(y)
        return y
