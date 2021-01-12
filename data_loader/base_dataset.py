import json
import os

from torch.utils.data.dataset import Dataset
from utils.helpers import load_dict, word_tokenize, init_matrix


class BaseDataset(Dataset):

    def __init__(self, news_file, behaviors_file, hparams, converter, test_set=False, npratio=-1):
        self.hparams, self.npratio, self.test_set = hparams, npratio, test_set
        self.flatten_article = hparams.flatten_article if hasattr(hparams, "flatten_article") else None
        self.converter = converter

        self.word_dict = load_dict(hparams.word_dict_file)
        self.uid2index = load_dict(hparams.user_dict_file)
        # initial data of corresponding news attributes
        self.news = {attr: [[""]] for attr in hparams.news_attr.keys()}
        self.nid2index = {}

        self._load_news(news_file)
        self._load_behaviors(behaviors_file, test_set)

    def _load_news(self, news_file):
        body_file = os.path.join(os.path.dirname(news_file), "msn.json")
        use_body = hasattr(self.hparams, "body_shape")
        articles = json.load(open(body_file)) if os.path.exists(body_file) and use_body else None
        self.body = [[""]] if use_body and self.flatten_article else [[[""]]]
        with open(news_file, "r", encoding="utf-8") as rd:
            for index in rd:
                # news id, category, subcategory, title, abstract, url
                nid, vert, subvert, title, ab, url, title_entity, abs_entity = index.strip("\n").split("\t")
                entities = self.load_entity(title_entity) if len(title_entity) > 2 else self.load_entity(abs_entity)
                entities = entities if len(entities) else title
                news_dict = {"title": title, "entity": entities, "vert": vert, "subvert": subvert, "abstract": ab}
                if nid in self.nid2index:
                    continue
                # add news attribution
                for attr in self.news.keys():
                    if attr in news_dict:
                        self.news[attr].append(word_tokenize(news_dict[attr]))

                # add news body
                if articles:
                    news_id = url.split("/")[-1].split(".")[0]
                    if news_id not in articles:
                        article = [""] if self.flatten_article else [[""]]
                    else:
                        if self.flatten_article:
                            # flatten body, load an article in a list
                            article = word_tokenize(" ".join(articles[news_id]))
                            article = article if article else [""]
                        else:
                            # load an article in multiply lists
                            article = [word_tokenize(sent) for sent in articles[news_id]]
                            article = article if article else [[""]]
                    self.body.append(article)
                self.nid2index[nid] = len(self.nid2index) + 1
        rd.close()
        self._init_matrix()

    def _init_matrix(self):
        # initial index matrix of news attributes
        self.news_matrix = {
            f"{attr}_index": init_matrix(self.converter(self.news[attr]),  # convert word to index
                                         [len(self.news[attr])] + self.hparams.news_attr[attr])  # shape
            for attr in self.news.keys()
        }
        # initial body index matrix
        if hasattr(self.hparams, "body_shape"):
            if self.flatten_article:
                body = self.converter(self.body)
            else:
                body = [self.converter(article) for article in self.body]
            shape = [len(body)] + self.hparams.body_shape
            body_matrix = init_matrix(body, shape)
            if self.flatten_article:
                body_matrix = body_matrix.reshape([shape[0]] + [shape[1] * shape[2]] + shape[3:])
            self.news_matrix["body_index"] = body_matrix

    def _load_behaviors(self, behaviors_file, test_set=False, col_spl="\t"):
        """"
        Create four global variants:
        histories: The index of news clicked by user in history (clicked)
        impr_news: The index of news in this impression (candidate)
        """
        # initial behaviors attributes
        self.histories, self.impr_news, self.labels, self.impr_indexes, self.uindexes = [], [], [], [], []
        with open(behaviors_file, "r", encoding="utf-8") as rd:
            impr_index = 0
            for index in rd:
                uid, time, history, impr = index.strip("\n").split(col_spl)[-4:]

                history = [self.nid2index[i] for i in history.split()]
                history = [0] * (self.hparams.his_size - len(history)) + history[:self.hparams.his_size]

                impr_news = [self.nid2index[i.split("-")[0]] for i in impr.split()]
                if not test_set:
                    self.labels.append([int(i.split("-")[1]) for i in impr.split()])
                if uid not in self.uid2index:
                    self.uid2index[uid] = len(self.uid2index) + 1
                uindex = self.uid2index[uid]
                self.histories.append(history)
                self.impr_news.append(impr_news)
                self.impr_indexes.append(impr_index)
                self.uindexes.append(uindex)
                impr_index += 1
        rd.close()

    @staticmethod
    def load_entity(entity):
        return " ".join([" ".join(e["SurfaceForms"]) for e in json.loads(entity)])

    def __getitem__(self, index):
        # get the matrix of corresponding news with index
        return self.impr_news[index], index, self.labels[index]

    def __len__(self):
        return len(self.impr_news)
