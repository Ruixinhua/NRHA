import torch
import torch.nn as nn
import torch.nn.functional as F
from models.nn.interest import Interest
from models.nn.layers import TimeDistributed, PositionwiseFeedForward
from models.nrha_base import NRHABase


class NRHAGRU(NRHABase):

    def __init__(self, hparams):
        super().__init__(hparams)
        out_dim = hparams.head_num * hparams.head_dim
        self.user_embedding = nn.Embedding(hparams.user_embedding_size, out_dim)
        self.feed_forward = PositionwiseFeedForward(out_dim, 512, self.dropout)
        self.interest_network = Interest(hparams.head_dim * hparams.head_num, gru_type=hparams.gru_type,
                                         att_hidden_layers=hparams.att_hidden_layers, att_dropout=hparams.att_dropout,
                                         att_activation=hparams.activation, use_negsampling=False)
        # self.user_self_att = nn.GRU(out_dim, out_dim)

    def sentence_encoder(self, y):
        y = self.feed_forward(self.news_encode_layer(y)).view(y.size(0), -1, self.head_num, self.head_dim)
        return self.head_encoder(y, self.news_attentions)

    def news_encoder(self, sequences):
        y = self._embedding_layer(sequences)
        y = self.sentence_encoder(y)
        return y

    def user_encoder(self, his_input):
        # run the history news read by user
        title, user_index, his_length = his_input
        long_u_emb = self.user_embedding(user_index)
        y = TimeDistributed(self.news_encoder)(title)
        user_present, _ = self.interest_network(long_u_emb, y, his_length)
        return user_present

    def forward(self, x):
        his_input_title, candidate_input_title, user_index, his_length = x
        user_present = self.user_encoder([his_input_title, user_index, his_length])
        news_present = TimeDistributed(self.news_encoder)(candidate_input_title)
        preds = torch.sum(news_present * user_present.unsqueeze(1), dim=-1)
        preds = F.softmax(preds, dim=-1)
        return preds
