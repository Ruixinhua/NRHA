import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from data_loader.base_dataset import BaseDataset


class UserDataset(Dataset):

    def __init__(self, dataset: BaseDataset):
        self.histories = dataset.histories
        self.uindexes = dataset.uindexes
        self.his_length = dataset.his_length
        self.news_matrix = dataset.news_matrix
        self.hparams = dataset.hparams

    def __getitem__(self, index):
        # get the matrix of corresponding news with index
        clicked_news = [self.news_matrix[attr][self.histories[index]] for attr in self.news_matrix.keys()]
        clicked_news = torch.tensor(np.concatenate(clicked_news, axis=1), dtype=torch.long)
        if self.hparams.embedding == "distill_bert":
            # pass mask list to the model
            # clicked index mask
            news_mask = torch.tensor(np.where(clicked_news == 0, 0, 1), dtype=torch.int8)
            clicked_news = (clicked_news, news_mask)
        return index, clicked_news, self.uindexes[index], self.his_length[index]

    def __len__(self):
        return len(self.histories)


class NewsDataset(Dataset):

    def __init__(self, dataset: BaseDataset):
        self.hparams = dataset.hparams
        self.news_matrix = dataset.news_matrix

    def __getitem__(self, index):
        # get the matrix of corresponding news with index
        news = [self.news_matrix[attr][index] for attr in self.news_matrix.keys()]
        input_feat = torch.tensor(np.concatenate(news, axis=0), dtype=torch.long)
        if self.hparams.embedding == "distill_bert":
            # pass mask list to the model
            # clicked index mask
            news_mask = torch.tensor(np.where(input_feat == 0, 0, 1), dtype=torch.int8)
            input_feat = (input_feat, news_mask)
        return index, input_feat

    def __len__(self):
        return len(self.news_matrix[list(self.news_matrix.keys())[0]])
