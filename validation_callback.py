import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from torch.utils.data.dataloader import DataLoader

from data_loader.base_dataset import BaseDataset
from data_loader.test_dataset import NewsDataset, UserDataset
from utils.metrics import cal_metric


class ValidationCallback(Callback):

    def __init__(self, news_file, behaviors_file, hparams, converter, test_set=False, val_check_interval=10):
        self.dataset = BaseDataset(news_file, behaviors_file, hparams, converter, test_set)
        self.news_dataloader = DataLoader(NewsDataset(self.dataset), hparams.batch_size)
        self.user_dataloader = DataLoader(UserDataset(self.dataset), hparams.batch_size)
        self.val_check_interval = val_check_interval
        self.metrics = hparams.metrics
        self.news_vectors, self.user_vectors = {}, {}
        super(ValidationCallback, self).__init__()

    def validation(self, model):
        for batch in self.news_dataloader:
            index, news = batch
            news_vec = model.news_encoder(news.to(model.device))
            self.news_vectors.update(dict(zip(index, news_vec.cpu().numpy())))
        for batch in self.user_dataloader:
            index, clicked_news = batch
            user_vec = model.user_encoder(clicked_news.to(model.device))
            self.user_vectors.update(dict(zip(index, user_vec.cpu().numpy())))
        group_y, group_pred = [], []
        for candidate, imp_index, y in zip(self.dataset.impr_news, self.dataset.impr_indexes, self.dataset.labels):
            # calculate for only on instance
            pred = np.dot(np.stack([self.news_vectors[i] for i in candidate]), self.user_vectors[imp_index])
            group_y.append(y)
            group_pred.append(pred)
        res = cal_metric(group_y, group_pred, self.metrics)
        print([model.log(k, v) for k, v in res.items()])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == self.val_check_interval:
            trainer.model.eval()
            with torch.no_grad():
                self.validation(trainer.model)

    def on_init_start(self, trainer):
        print('Starting to init trainer!')

    def on_init_end(self, trainer):
        print('trainer is init now')

    def on_train_end(self, trainer, pl_module):
        print('do something when training ends')

