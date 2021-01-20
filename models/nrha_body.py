import torch
from torchnlp.nn import Attention

from models.nn.layers import clones, AttLayer, TimeDistributed
from models.nrha_base import NRHABase


class NRHABody(NRHABase):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.news_attentions = clones(AttLayer(hparams.head_dim, hparams.attention_hidden_dim), hparams.head_num)
        self.user_attentions = clones(AttLayer(hparams.head_dim, hparams.attention_hidden_dim), hparams.head_num)
        self.title_body_att = Attention(hparams.head_dim)

    def sentence_encoder(self, y):
        y = self.news_encode_layer(y).view(y.size(0), -1, self.head_num, self.head_dim)
        return self.head_encoder(y, self.news_attentions)

    def news_encoder(self, sequences):
        y = self._embedding_layer(sequences)
        title, body = y[:, :self.news_attr["title"][0]], y[:, self.news_attr["title"][0]:]
        body = body.reshape(body.shape[0], self.body_shape[0], self.body_shape[1], self.word_emb_dim)
        body = body.transpose(0, 1)
        q = self.sentence_encoder(title).reshape(-1, self.head_num, self.head_dim)
        q = q.reshape(q.shape[0] * self.head_num, 1, self.head_dim)
        y = torch.stack([self.sentence_encoder(sent) for sent in body]).transpose(0, 1)
        y = y.reshape(-1, self.body_shape[0], self.head_num, self.head_dim)
        y = y.reshape(y.shape[0] * self.head_num, self.body_shape[0], self.head_dim)
        y, _ = self.title_body_att(q, y)
        y = y.reshape(-1, self.head_num * self.head_dim)
        return y

    def user_encoder(self, his_input_title):
        # run the history news read by user
        y = TimeDistributed(self.news_encoder)(his_input_title)
        y = self.user_encode_layer(y).view(y.size(0), -1, self.head_num, self.head_dim)
        return self.head_encoder(y, self.user_attentions)
