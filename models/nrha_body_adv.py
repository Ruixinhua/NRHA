import torch.nn as nn
from models.nrha_body import NRHABody


class NRHABodyAdv(NRHABody):

    def __init__(self, hparams):
        super().__init__(hparams)
        out_dim = hparams.head_num * hparams.head_dim
        padding = (hparams.kernel_size - 1) // 2
        assert 2 * padding == hparams.kernel_size - 1, "Kernel size must be an odd number"
        self.news_encode_layer = nn.Sequential(
            nn.Conv1d(hparams.word_emb_dim, out_dim, hparams.kernel_size, padding=padding),
            nn.ReLU()
        )

    def sentence_encoder(self, y):
        y = self.news_encode_layer(y.transpose(1, 2)).transpose(1, 2)
        y = y.view(y.size(0), -1, self.head_num, self.head_dim)
        return self.head_encoder(y, self.news_attentions)
