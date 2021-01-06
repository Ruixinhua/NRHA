# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import AttLayer, TimeDistributed
import pytorch_lightning as pl
from utils.loss import CategoricalLoss
from utils.metrics import cal_metric


class NRMSModel(pl.LightningModule):

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
        self.news_att_layer = AttLayer(self.word_emb_dim, self.attention_hidden_dim)
        self.user_att_layer = AttLayer(self.word_emb_dim, self.attention_hidden_dim)
        self.news_self_att = nn.MultiheadAttention(self.word_emb_dim, self.head_num)
        self.user_self_att = nn.MultiheadAttention(self.word_emb_dim, self.head_num)
        # for fast evaluation
        self.news_vectors, self.user_vectors = {}, {}

    def _embedding_layer(self, sequences):
        if self.embedding == "elmo":
            # only using the last layer results
            sequences = self.embedding_layer(sequences)["elmo_representations"]
            sequences = F.dropout(sequences[-1], p=self.dropout)
        else:
            sequences = F.dropout(self.embedding_layer(sequences), p=self.dropout)
        return sequences

    def news_encoder(self, sequences):
        y = self._embedding_layer(sequences).transpose(0, 1)
        y = self.news_self_att(y, y, y)[0]
        y = F.dropout(y, p=self.dropout).transpose(0, 1)
        y = self.news_att_layer(y)
        return y

    def user_encoder(self, his_input_title):
        # change size to (S, N, D): sequence length, batch size, word dimension
        y = TimeDistributed(self.news_encoder)(his_input_title).transpose(0, 1)
        # change size back to (N, S, D)
        y = self.user_self_att(y, y, y)[0].transpose(0, 1)
        y = self.user_att_layer(y)
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

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            index, news = batch
            news_vec = self.news_encoder(news)
            self.news_vectors.update(dict(zip(index.cpu().tolist(), news_vec.cpu().numpy())))
        elif dataloader_idx == 1:
            index, clicked_news = batch
            user_vec = self.user_encoder(clicked_news)
            self.user_vectors.update(dict(zip(index.cpu().tolist(), user_vec.cpu().numpy())))
        else:
            # calculate for only on instance
            candidate_index, imp_index, y = batch
            candidate_index = [i.cpu().tolist()[0] for i in candidate_index]
            y = [i.cpu().tolist()[0] for i in y]
            imp_index = imp_index.cpu().tolist()[0]
            candidate_vector = np.stack([self.news_vectors[i] if i in self.news_vectors else self.news_vectors[0]
                                         for i in candidate_index])
            user_vector = self.user_vectors[imp_index]
            pred = np.dot(candidate_vector, user_vector).tolist()
            return y, pred

    def validation_epoch_end(self, outputs):
        y, pred = [], []
        for out in outputs[2]:
            y.append(out[0])
            pred.append(out[1])
        res = cal_metric(y, pred, self.metrics)
        log = [self.log(k, v) for k, v in res.items()]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
