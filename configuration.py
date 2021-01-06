import os
import json

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


def get_params(yaml_name, **kwargs):
    if yaml_name:
        yaml_path = os.path.join(data_root_path, "utils", yaml_name)
    else:
        yaml_path = get_yaml_path()
    return prepare_hparams(yaml_path, word_emb_file=get_emb_path(), word_dict_file=get_dict_file(), epochs=epochs,
                           show_step=10, user_dict_file=get_user_dict_path(), **kwargs)
