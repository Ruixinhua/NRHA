import torch
import torch.nn as nn

from models.base import BaseModel
from models.nn.layers import clones, AttLayer, TimeDistributed

__all__ = ["NRHABase"]


class NRHABase(BaseModel):

    def __init__(self, hparams):
        super().__init__(hparams)
        out_dim = hparams.head_num * hparams.head_dim
        self.news_encode_layer = nn.Linear(hparams.word_emb_dim, out_dim)
        self.user_encode_layer = nn.Linear(out_dim, out_dim)
        self.news_attentions = clones(AttLayer(hparams.head_dim, hparams.attention_hidden_dim), hparams.head_num)
        self.user_attentions = clones(AttLayer(hparams.head_dim, hparams.attention_hidden_dim), hparams.head_num)

    def head_encoder(self, y, attentions):
        # shape of y: [N, S, H, D]     E is head_num * head_dim, E = H * D
        y = y.transpose(1, 2).transpose(0, 1)
        # shape of y: [H, N, S, D]
        y = torch.stack([att(h)[0] for h, att in zip(y, attentions)])
        y = y.transpose(0, 1)
        # shape of y: [N, H, D]
        y = y.reshape(y.shape[0], self.head_num * self.head_dim)
        # shape of y: [N, E]
        return y

    def sentence_encoder(self, y):
        y = self.news_encode_layer(y).view(y.size(0), -1, self.head_num, self.head_dim)
        return self.head_encoder(y, self.news_attentions)

    def news_encoder(self, sequences):
        y = self._embedding_layer(sequences)
        y = self.sentence_encoder(y)
        return y

    def user_encoder(self, his_input_title):
        # run the history news read by user
        y = TimeDistributed(self.news_encoder)(his_input_title)
        y = self.user_encode_layer(y).view(y.size(0), -1, self.head_num, self.head_dim)
        return self.head_encoder(y, self.user_attentions)
