import argparse
import os
import json

from models.base import BaseModel
from models.nrha_adv import NRHAAdv
from models.nrha_base import NRHABase
from models.nrha_body import NRHABody
from models.nrha_body_adv import NRHABodyAdv
from models.nrha_body_bilstm import NRHABodyBiLSTM
from models.nrha_conv import NRHAConv
from models.nrha_gru import NRHAGRU
from models.nrha_mlp import NRHAMLP
from models.nrha_test import NRHATest
from models.nrha_title import NRHATitle
from utils.download_utils import download_resources, get_mind_data_set
from utils.params_utils import prepare_hparams

epochs = 8
seed = 42
data_root_path = json.load(open("config.json"))["data_root_path"] if os.path.exists("config.json") else None


def set_data_root_path(new_path):
    global data_root_path
    data_root_path = new_path


def get_data_path(mind_type="small"):
    return os.path.join(data_root_path, mind_type)


def get_path(name, mind_type="small"):
    # set up the path of dataset
    news_file = os.path.join(get_data_path(mind_type), name, r'news.tsv')
    behaviors_file = os.path.join(get_data_path(mind_type), name, r'behaviors.tsv')
    return news_file, behaviors_file


def get_emb_path():
    return os.path.join(data_root_path, "utils", "embedding.npy")


def get_user_dict_path():
    return os.path.join(data_root_path, "utils", "uid2index.pkl")


def get_dict_file():
    return os.path.join(data_root_path, "utils", "word_dict.pkl")


def get_yaml_path(yaml_name=r"nrha.yaml"):
    return os.path.join(data_root_path, "utils", yaml_name)


def download_data(mind_type="small"):
    mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(mind_type)
    data_path = get_data_path(mind_type)
    train_news_file, _ = get_path("train", mind_type)
    valid_news_file, _ = get_path("valid", mind_type)
    test_news_file, _ = get_path("test", mind_type)
    if not os.path.exists(train_news_file):
        download_resources(mind_url, os.path.join(data_path, 'train'), mind_train_dataset)
    if not os.path.exists(valid_news_file):
        download_resources(mind_url, os.path.join(data_path, 'valid'), mind_dev_dataset)
    if mind_type == "large":
        if not os.path.exists(test_news_file):
            download_resources(mind_url, os.path.join(data_path, 'test'), mind_dev_dataset)
    if not os.path.exists(get_yaml_path()):
        utils_url = r'https://recodatasets.blob.core.windows.net/newsrec/'
        download_resources(utils_url, os.path.join(data_root_path, 'utils'), mind_utils)


def get_params(yaml_path, **kwargs):
    return prepare_hparams(yaml_path, word_emb_file=get_emb_path(), word_dict_file=get_dict_file(), epochs=epochs,
                           show_step=10, user_dict_file=get_user_dict_path(), **kwargs)


def get_argument():
    parse = argparse.ArgumentParser(description="Training process")
    parse.add_argument("--log", "-l", dest="log", metavar="FILE", help="log file", default="test")
    parse.add_argument("--configure", "-c", dest="config", metavar="FILE", help="yaml file",
                       default=r"config/nrha.yaml")
    parse.add_argument("--gup_num", "-g", dest="gpus", metavar="INT", default=1, help="The number of gpu")
    parse.add_argument("--model_class", "-m", dest="model_class", metavar="TEXT", default="nrha")
    parse.add_argument("--mind_type", "-t", dest="mind_type", metavar="TEXT", default="demo")
    parse.add_argument("--head_num", "-n", dest="head_num", metavar="INT", default=20)
    parse.add_argument("--head_dim", "-d", dest="head_dim", metavar="INT", default=20)
    parse.add_argument("--batch_size", "-b", dest="batch_size", metavar="INT", default=32)
    parse.add_argument("--resume", "-r", dest="resume", metavar="INT", default=0, help="resume from best")
    parse.add_argument("--train_mode", "-i", dest="mode", metavar="INT", default=0,
                       help="0: default test; 1: head number test; 2: body shape test")
    parse.add_argument("--num_workers", "-w", dest="num_workers", metavar="INT", default=0)
    return parse.parse_args()


def get_model_class(model_class):
    if model_class == "nrha":
        return NRHATitle
    elif model_class == "nrha_base":
        return NRHABase
    elif model_class == "nrha_adv":
        return NRHAAdv
    elif model_class == "nrha_conv":
        return NRHAConv
    elif model_class == "nrha_gru":
        return NRHAGRU
    elif model_class == "nrha_body":
        return NRHABody
    elif model_class == "nrha_body_bilstm":
        return NRHABodyBiLSTM
    elif model_class == "nrha_body_adv":
        return NRHABodyAdv
    elif model_class == "nrha_mlp":
        return NRHAMLP
    elif model_class == "nrha_test":
        return NRHATest
    else:
        return BaseModel
