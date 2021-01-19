import numpy as np
import torch

from data_loader.base_dataset import BaseDataset
from utils.helpers import newsample


class TrainingDataset(BaseDataset):
    
    def __init__(self, news_file, behaviors_file, hparams, converter, npratio=-1):
        super().__init__(news_file, behaviors_file, hparams, converter, False, npratio)
        
        self._load_index()

    def _load_index(self):
        attrs = ["labels", "impression_index", "user_index", "candidate_index", "clicked_index", "history_length"]
        self.impression_index = {attr: [] for attr in attrs}
        indexes = np.arange(len(self.labels))
        if self.npratio > 0:
            np.random.shuffle(indexes)

        for i in indexes:
            for impression in self.parser_one_line(i):
                for key in self.impression_index.keys():
                    self.impression_index[key].append(impression[key])

    def parser_one_line(self, line):
        """Parse one behavior sample into feature values.
        if npratio is larger than 0, return negative sampled result.

        Args:
            line (int): sample index.

        Returns:
            list: Parsed results including label, impression id , user id,
            candidate_{atr}_index, clicked_{atr}_index.
        """
        impr_label = self.labels[line]
        impr = self.impr_news[line]
        if self.npratio > 0:
            poss = []
            negs = []

            for news, click in zip(impr, impr_label):
                # divide click and un-click news
                if click == 1:
                    poss.append(news)
                else:
                    negs.append(news)

            for p in poss:
                label = [1] + [0] * self.npratio
                impr_index = self.impr_indexes[line]
                user_index = self.uindexes[line]
                n = [p] + newsample(negs, self.npratio)
                impression = {"labels": label, "impression_index": impr_index, "user_index": user_index,
                              "candidate_index": n, "clicked_index": line, "history_length": self.his_length[line]}
                yield impression

        else:
            # use all impression news as training data
            for news, label in zip(impr, impr_label):
                label = [label]
                impr_index = self.impr_indexes[line]
                user_index = self.uindexes[line]
                impression = {"labels": label, "impression_index": impr_index, "user_index": user_index,
                              "candidate_index": news, "clicked_index": line, "history_length": self.his_length[line]}
                yield impression

    def __getitem__(self, index):
        impression = {"clicked_index": [], "candidate_index": []}
        for attr in self.news_matrix.keys():
            # get a single impression from impression index
            candidate_index = self.news_matrix[attr][self.impression_index["candidate_index"][index]]
            clicked_index = self.news_matrix[attr][self.histories[self.impression_index["clicked_index"][index]]]
            impression["candidate_index"].append(candidate_index)
            impression["clicked_index"].append(clicked_index)
        input_feat = [torch.tensor(np.concatenate(impression[k], axis=1), dtype=torch.long) for k in impression.keys()]
        if self.hparams.model == "nrha_gru":
            input_feat.append(torch.tensor(self.impression_index["user_index"][index], dtype=torch.long))
            input_feat.append(torch.tensor(self.impression_index["history_length"][index], dtype=torch.long))
        label = torch.tensor(self.impression_index["labels"][index], dtype=torch.long)
        return input_feat, label

    def __len__(self):
        return len(self.impression_index["labels"])
