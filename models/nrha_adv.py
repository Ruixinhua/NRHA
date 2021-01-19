import torch.nn as nn

from models.layers import TimeDistributed, MultiHeadedAttention, LayerNorm, clones, SublayerConnection, \
    PositionwiseFeedForward
from models.nrha_base import NRHABase


class NRHAAdv(NRHABase):

    def __init__(self, hparams):
        super().__init__(hparams)
        out_dim = hparams.head_num * hparams.head_dim
        self.feed_forward = PositionwiseFeedForward(out_dim, 512, self.dropout)
        self.sublayer = clones(SublayerConnection(out_dim, self.dropout), 2)
        # self.user_self_att = nn.GRU(out_dim, out_dim)

    def sentence_encoder(self, y):
        y = self.feed_forward(self.news_self_att(y)).view(y.size(0), -1, self.head_num, self.head_dim)
        return self.head_encoder(y, self.news_attentions)

    def news_encoder(self, sequences):
        y = self._embedding_layer(sequences)
        y = self.sentence_encoder(y)
        return y

    def user_encoder(self, his_input_title):
        # run the history news read by user
        y = TimeDistributed(self.news_encoder)(his_input_title)
        y = y.view(y.size(0), -1, self.head_num, self.head_dim)
        return self.head_encoder(y, self.news_attentions)
