# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import AttLayer, TimeDistributed, MultiHeadedAttention
import pytorch_lightning as pl
from utils.loss import CategoricalLoss


class BaseModel(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.__dict__.update(hparams.params)
        self.hparams.update(hparams.params)
        self.save_hyperparameters()
        self.word2vec_embedding = np.load(self.word_emb_file)
        if self.embedding == "elmo":
            from allennlp.modules.elmo import Elmo
            self.embedding_layer = Elmo(self.options_file, self.weight_file, num_output_representations=4)
        else:
            self.embedding_layer = nn.Embedding(self.word2vec_embedding.shape[0], self.word_emb_dim).from_pretrained(
                torch.FloatTensor(self.word2vec_embedding), freeze=False)
        self.news_att_layer = AttLayer(self.head_num * self.head_dim, self.attention_hidden_dim)
        self.user_att_layer = AttLayer(self.head_num * self.head_dim, self.attention_hidden_dim)
        self.news_encode_layer = MultiHeadedAttention(self.head_num, self.head_dim, self.word_emb_dim)
        self.user_encode_layer = MultiHeadedAttention(self.head_num, self.head_dim, self.head_num * self.head_dim)
        self.dropouts = nn.Dropout(self.dropout)
        # for fast evaluation

    def _embedding_layer(self, sequences):
        if self.embedding == "elmo":
            # only using the last layer results
            sequences = self.embedding_layer(sequences)["elmo_representations"]
            sequences = self.dropouts(sequences[-1])
        else:
            sequences = self.dropouts(self.embedding_layer(sequences))
        return sequences

    def news_encoder(self, sequences):
        y = self._embedding_layer(sequences)
        y = self.news_encode_layer(y, y, y)[0]
        y = self.dropouts(y)
        y = self.news_att_layer(y)[0]
        return y

    def user_encoder(self, his_input_title):
        # change size to (S, N, D): sequence length, batch size, word dimension
        y = TimeDistributed(self.news_encoder)(his_input_title)
        # change size back to (N, S, D)
        y = self.user_encode_layer(y, y, y)[0]
        y = self.user_att_layer(y)[0]
        return y

    def forward(self, x):
        his_input_title, pred_input_title = x
        user_present = self.user_encoder(his_input_title)
        news_present = TimeDistributed(self.news_encoder)(pred_input_title)
        # equal to Dot(axis=-1)([x, y])
        preds = torch.sum(news_present * user_present.unsqueeze(1), dim=-1)
        preds = F.softmax(preds, dim=-1)
        return preds

    def predict(self, x):
        his_input_title, pred_input_title_one = x
        user_present = self.user_encoder(his_input_title)
        pred_title_one_reshape = torch.reshape(pred_input_title_one, (self.title_size,))
        news_present_one = self.news_encoder(pred_title_one_reshape)
        # equal to Dot(axis=-1)([x, y])
        preds_one = torch.sum(news_present_one * user_present.unsqueeze(1), dim=-1)
        preds_one = F.sigmoid(preds_one)
        return preds_one

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        sequences, y = batch
        y_hat = self(sequences)
        loss = CategoricalLoss()(y_hat, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
