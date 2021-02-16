import torch.nn as nn

from models.nn.layers import PositionwiseFeedForward, TimeDistributed, AttLayer
from models.nrha_base import NRHABase
from models.nrha_conv import NRHAConv


class NRHAAdv(NRHABase):

    def __init__(self, hparams):
        super().__init__(hparams)
        out_dim = hparams.head_num * hparams.head_dim
        self.news_encode_layer = nn.LSTM(self.word_emb_dim, out_dim, batch_first=True, bidirectional=True)
        self.user_encode_layer = nn.LSTM(out_dim * 2, out_dim, batch_first=True, bidirectional=True)
        self.news_att_layer = AttLayer(out_dim * 2, self.attention_hidden_dim)
        self.user_att_layer = AttLayer(out_dim * 2, self.attention_hidden_dim)
        # self.news_encode_layer = PositionwiseFeedForward(self.word_emb_dim, out_dim, dropout=self.dropout)
        # self.user_encode_layer = PositionwiseFeedForward(out_dim, out_dim, dropout=self.dropout)

    def sentence_encoder(self, y):
        y, (_, _) = self.news_encode_layer(y)
        y, _ = self.news_att_layer(y)
        # y = y.view(y.size(0), -1, self.head_num, self.head_dim)
        # return self.head_encoder(y, self.news_attentions)
        return y

    def user_encoder(self, his_input_title):
        # change size to (S, N, D): sequence length, batch size, word dimension
        y = TimeDistributed(self.news_encoder)(his_input_title)
        # change size back to (N, S, D)
        y, (_, _) = self.user_encode_layer(y)
        y = self.user_att_layer(y)[0]
        return y
