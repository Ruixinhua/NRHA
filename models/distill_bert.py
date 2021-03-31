import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from models.base import BaseModel
from models.nn.layers import AttLayer


class DistillBert(BaseModel):

    def __init__(self, hparams):

        super().__init__(hparams)
        config = AutoConfig.from_pretrained(self.model_name)
        config.n_layers = 1
        self.distilbert = AutoModel.from_pretrained(self.model_name, config=config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        self.news_att_layer = AttLayer(config.dim, self.attention_hidden_dim)
        self.user_att_layer = AttLayer(config.dim, self.attention_hidden_dim)

    def news_encoder(self, sequences, head_mask=None):
        x, attention_mask = sequences
        distilbert_output = self.distilbert(input_ids=x,
                                            attention_mask=attention_mask,
                                            head_mask=head_mask)

        hidden_state = distilbert_output[0]
        # take the first cls as the feature of news
        y = self.pre_classifier(hidden_state[:, 0])
        y = nn.ReLU()(y)
        y = self.dropout(y)
        return y

    def time_distributed(self, x, mask):
        # calculate news features across time series
        x_shape = torch.Size([-1]) + x.size()[2:]
        x_reshape = x.contiguous().view(x_shape)
        mask_reshape = mask.contiguous().view(x_shape)
        y = self.news_encoder((x_reshape, mask_reshape))
        y = y.contiguous().view(x.size(0), -1, y.size(-1))
        return y

    def user_encoder(self, his_input):
        his_input_title, his_title_mask = his_input
        # change size to (S, N, D): sequence length, batch size, word dimension
        y = self.time_distributed(his_input_title, his_title_mask)
        # change size back to (N, S, D)
        y = self.user_att_layer(y)[0]
        return y

    def forward(self, x):
        his_input_title, candidate_input_title, his_title_mask, candidate_title_mask = x
        user_present = self.user_encoder((his_input_title, his_title_mask))
        news_present = self.time_distributed(candidate_input_title, candidate_title_mask)
        # equal to Dot(axis=-1)([x, y])
        preds = torch.sum(news_present * user_present.unsqueeze(1), dim=-1)
        preds = F.softmax(preds, dim=-1)
        return preds
