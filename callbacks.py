import zipfile
import numpy as np
import torch
import os
from pytorch_lightning.callbacks import Callback
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from data_loader.base_dataset import BaseDataset
from data_loader.test_dataset import NewsDataset, UserDataset
from utils.metrics import cal_metric


class BaseCallback(Callback):
    def __init__(self, news_file, behaviors_file, hparams, converter, test_set):
        self.dataset = BaseDataset(news_file, behaviors_file, hparams, converter, test_set)
        self.news_dataloader = DataLoader(NewsDataset(self.dataset), hparams.batch_size)
        self.user_dataloader = DataLoader(UserDataset(self.dataset), hparams.batch_size)
        self.news_vectors, self.user_vectors = {}, {}
        super(BaseCallback, self).__init__()

    def run_news_users(self, model):
        for batch in tqdm(self.news_dataloader):
            "Run news data"
            index, news = batch
            news_vec = model.news_encoder(news.to(model.device))
            self.news_vectors.update(dict(zip(index.cpu().tolist(), news_vec.cpu().numpy())))
        for batch in tqdm(self.user_dataloader):
            "Run users data"
            index, clicked_news = batch
            user_vec = model.user_encoder(clicked_news.to(model.device))
            self.user_vectors.update(dict(zip(index.cpu().tolist(), user_vec.cpu().numpy())))


class ValidationCallback(BaseCallback):
    def __init__(self, news_file, behaviors_file, hparams, converter, ckpt_dir, check_interval=10):
        super(ValidationCallback, self).__init__(news_file, behaviors_file, hparams, converter, False)
        self.val_check_interval = check_interval
        self.metrics, self.best_auc, self.ckpt_dir = hparams.metrics, 0.0, ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def validation(self, model):
        self.run_news_users(model)
        group_y, group_pred = [], []
        for candidate, index, y in tqdm(zip(self.dataset.impr_news, self.dataset.impr_indexes, self.dataset.labels)):
            # calculate for only on instance
            pred = np.dot(np.stack([self.news_vectors[i] for i in candidate]), self.user_vectors[index])
            group_y.append(y)
            group_pred.append(pred)
        res = cal_metric(group_y, group_pred, self.metrics)
        print(res)
        return res

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0: return
        if (batch_idx % self.val_check_interval) == 0:
            pl_module.eval()
            with torch.no_grad():
                res = self.validation(pl_module)
                if res["group_auc"] > self.best_auc:
                    self.best_auc = res["group_auc"]
                    cur_model_path = os.path.join(self.ckpt_dir, f"group_auc=={res['group_auc']}.ckpt")
                    trainer.save_checkpoint(cur_model_path)
                    trainer.save_checkpoint(os.path.join(self.ckpt_dir, "best_model.ckpt"))
            pl_module.train()


class TestCallback(BaseCallback):
    def __init__(self, news_file, behaviors_file, hparams, converter, ckpt_dir, pred_dir):
        super(TestCallback, self).__init__(news_file, behaviors_file, hparams, converter, True)
        self.pred_dir = pred_dir
        self.best_model_path = os.path.join(ckpt_dir, "best_model.ckpt")
        os.makedirs(self.pred_dir, exist_ok=True)

    def inference(self, model):
        model.eval()
        with torch.no_grad():
            self.run_news_users(model)
            group_imp_indexes, group_pred = [], []
            for candidate, imp_index in tqdm(zip(self.dataset.impr_news, self.dataset.impr_indexes)):
                # calculate for only on instance
                pred = np.dot(np.stack([self.news_vectors[i] for i in candidate]), self.user_vectors[imp_index])
                group_pred.append(pred)
                group_imp_indexes.append(imp_index)
            with open(os.path.join(self.pred_dir, 'prediction.txt'), 'w') as f:
                for impr_index, preds in tqdm(zip(group_imp_indexes, group_pred)):
                    impr_index += 1
                    pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
                    pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
                    f.write(' '.join([str(impr_index), pred_rank]) + '\n')
            f = zipfile.ZipFile(os.path.join(self.pred_dir, 'prediction.zip'), 'w', zipfile.ZIP_DEFLATED)
            f.write(os.path.join(self.pred_dir, 'prediction.txt'), arcname='prediction.txt')
            f.close()

    def on_fit_end(self, trainer, pl_module):
        print("Inference beginning")
        model = pl_module.load_from_checkpoint(checkpoint_path=self.best_model_path)
        self.inference(model)
