import pickle
import random
import re
import yaml
import numpy as np
from transformers import AutoTokenizer


def load_dict(file_path):
    """ load pickle file
    Args:
        file path (str): file path

    Returns:
        (obj): pickle load obj
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_yaml(filename):
    """Load a yaml file.

    Args:
        filename (str): Filename.

    Returns:
        dict: Dictionary.
    """
    try:
        with open(filename, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        return config
    except FileNotFoundError:  # for file not found
        raise
    except Exception as e:  # for other exceptions
        raise IOError("load {0} error!".format(filename))


def word_tokenize(sent):
    """ Split sentence into word list using regex.
    Args:
        sent (str): Input sentence

    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []


def newsample(news, ratio):
    """ Sample ratio samples from news list.
    If length of news is less than ratio, pad zeros.

    Args:
        news (list): packed_input news list
        ratio (int): sample number

    Returns:
        list: output of sample list.
    """
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)


def init_matrix(data, shape):
    """
    Initial the data matrix as shape.
    Args:
        data (list): packed_input data
        shape (tuple): the shape of matrix

    Returns:
        numpy matrix of data
    """
    matrix = np.zeros(shape, dtype=np.int32)
    if len(shape) == 1:
        for index in range(min(shape[0], len(data))):
            matrix[index] = data[index]
        return matrix
    else:
        for index in range(min(shape[0], len(data))):
            matrix[index] = init_matrix(data[index], shape[1:])
    return matrix


class Converter:
    def __init__(self, hparams):
        converter_type = hparams.embedding
        if converter_type == "elmo":
            from allennlp.modules.elmo import batch_to_ids
            self.converter = batch_to_ids
        elif converter_type == "word2vec":
            # load dictionary for word2vec embedding
            self.word_dict = load_dict(hparams.word_dict_file)
            self.converter = self.convert
        elif converter_type == "distill_bert":
            self.tokenizer = AutoTokenizer.from_pretrained(hparams.model_name)
            self.max_seq_len = hparams.max_seq_len
            self.converter = self.encode

    def encode(self, articles):
        x_encoded = [self.tokenizer.encode(" ".join(x), add_special_tokens=True,
                                           max_length=self.max_seq_len, truncation=True)
                     for x in articles]
        return x_encoded

    def convert(self, article):
        """
        convert article to index
        Args:
            article (list(list)): It can be a batch of data or an article

        Returns:
        word index list
        """
        return [[self.word_dict[word] if word in self.word_dict else 0 for word in sent] for sent in article]
