from utils.helpers import load_yaml


class HParams:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.params = sorted(self.__dict__.items(), key=lambda i: i[0])

    def to_string(self):
        return ",".join([f"({k}, {v})" for k, v in self.params])

    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        self.params = sorted(self.__dict__.items(), key=lambda i: i[0])
        return self


def flat_config(config):
    """Flat config loaded from a yaml file to a flat dict.

    Args:
        config (dict): Configuration loaded from a yaml file.

    Returns:
        dict: Configuration dictionary.
    """
    f_config = {}
    category = config.keys()
    for cate in category:
        for key, val in config[cate].items():
            f_config[key] = val
    return f_config


def prepare_hparams(yaml_file=None, **kwargs):
    """Prepare the trainer hyperparameters and check that all have the correct value.

    Args:
        yaml_file (str): YAML file as configuration.

    Returns:
        obj: Hyperparameter object in TF (tf.contrib.training.HParams).
    """
    if yaml_file is not None:
        config = load_yaml(yaml_file)
        config = flat_config(config)
    else:
        config = {}

    config.update(kwargs)

    return create_hparams(config)


def create_hparams(flags):
    """Create the trainer hyperparameters.

    Args:
        flags (dict): Dictionary with the trainer requirements.

    Returns:
        obj: Hyperparameter object in TF (tf.contrib.training.HParams).
    """
    return HParams(
        # data
        data_format=flags.get("data_format", None),
        iterator_type=flags.get("iterator_type", None),
        support_quick_scoring=flags.get("support_quick_scoring", False),
        entityEmb_file=flags.get("entityEmb_file", None),
        entityIdDict_file=flags.get("entityIdDict_file", None),
        subvertDict_file=flags.get("subvertDict_file", None),
        # models
        model_type=flags.get("model_type", "nrms"),
        title_size=flags.get("title_size", None),
        body_size=flags.get("body_size", None),
        word_emb_dim=flags.get("word_emb_dim", None),
        word_size=flags.get("word_size", None),
        user_num=flags.get("user_num", None),
        vert_num=flags.get("vert_num", None),
        subvert_num=flags.get("subvert_num", None),
        his_size=flags.get("his_size", None),
        npratio=flags.get("npratio"),
        dropout=flags.get("dropout", 0.0),
        attention_hidden_dim=flags.get("attention_hidden_dim", 200),
        # query
        entity_id_size=flags.get("entity_id_size", None),
        entity_emb_dim=flags.get("entity_emb_dim", None),
        # nrms
        head_num=flags.get("head_num", 4),
        head_dim=flags.get("head_dim", 100),
        # naml
        cnn_activation=flags.get("cnn_activation", None),
        dense_activation=flags.get("dense_activation", None),
        filter_num=flags.get("filter_num", 200),
        window_size=flags.get("window_size", 3),
        vert_emb_dim=flags.get("vert_emb_dim", 100),
        subvert_emb_dim=flags.get("subvert_emb_dim", 100),
        # lstur
        gru_unit=flags.get("gru_unit", 400),
        type=flags.get("type", "ini"),
        # npa
        user_emb_dim=flags.get("user_emb_dim", 50),
        # train
        learning_rate=flags.get("learning_rate", 0.001),
        loss=flags.get("loss", None),
        optimizer=flags.get("optimizer", "adam"),
        epochs=flags.get("epochs", 10),
        batch_size=flags.get("batch_size", 1),
        log_file=flags.get("log_file", "log.txt"),
        # show info
        show_step=flags.get("show_step", 1),
        metrics=flags.get("metrics", None),
    ).update(**flags)
