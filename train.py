import itertools
import os
from callbacks import ValidationCallback
from configuration import get_path, get_params, get_argument, get_model_class
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from data_loader.training_dataset import TrainingDataset
from torch.utils.data.dataloader import DataLoader
from utils.helpers import Converter


def run(option=None):
    saved_dir = f"{args.mind_type}/{args.model_class}"
    option = option or args.log
    ckpt_dir = f"saved/checkpoint/{saved_dir}/{option}"
    best_model_path = os.path.join(ckpt_dir, "best_model.ckpt")
    resume_path = best_model_path if os.path.exists(best_model_path) and int(args.resume) else None
    valid_callback = ValidationCallback(valid_news_file, valid_behaviors_file, hparams, converter, ckpt_dir, interval)
    tb_logger = pl_loggers.TensorBoardLogger(f"saved/logs/{saved_dir}")
    trainer = pl.Trainer(gpus=int(args.gpus), accelerator=accelerator, max_epochs=epochs, deterministic=True,
                         logger=tb_logger, callbacks=[valid_callback], resume_from_checkpoint=resume_path)
    trainer.fit(model_class(hparams), train_dataloader)
    group_auc = [float(file.split("==")[1].replace(".ckpt", "")) for file in os.listdir(ckpt_dir) if "==" in file]
    best_auc = max(group_auc)
    for file in os.scandir(ckpt_dir):
        if "best_model" in file.name:
            continue
        auc = float(file.name.split("==")[1].replace(".ckpt", ""))
        if auc < best_auc:
            os.remove(file.path)


def test_options(options, option_name):
    for option in options:
        hparams.update(**{option_name: option})
        print(hparams)
        # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
        run(option)


if  __name__ == "__main__":
    args = get_argument()
    hparams = get_params(args.config, head_num=int(args.head_num), head_dim=int(args.head_dim), model=args.model_class,
                         batch_size=int(args.batch_size))
    accelerator = "ddp" if int(args.gpus) > 1 else None
    print(args)
    train_news_file, train_behaviors_file = get_path("train", mind_type=args.mind_type)
    valid_news_file, valid_behaviors_file = get_path("valid", mind_type=args.mind_type)
    test_news_file, test_behaviors_file = get_path("test", mind_type=args.mind_type)

    converter = Converter(hparams.embedding, word_dict_file=hparams.word_dict_file).converter
    train_dataset = TrainingDataset(train_news_file, train_behaviors_file, hparams, converter, npratio=hparams.npratio)
    train_dataloader = DataLoader(train_dataset, hparams.batch_size, num_workers=0)
    # TODO: modify value
    interval, epochs = len(train_dataloader) / 3, 8
    # TODO: modify name
    pl.trainer.seed_everything(40)
    model_class = get_model_class(args.model_class)
    hparams.update(**{"user_embedding_size": len(train_dataset.uid2index)})

    gru_types = ["GRU", "AIGRU", "AGRU", "AUGRU"]
    head_nums = [10, 20]
    test_options(head_nums, "head_num")
# run()
