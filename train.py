from callbacks import TestCallback, ValidationCallback
from configuration import get_path, get_params
import argparse
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from data_loader.training_dataset import TrainingDataset
from torch.utils.data.dataloader import DataLoader

from models.base import BaseModel
from models.nrha_title import NRHATitle
from utils.helpers import get_converter


parse = argparse.ArgumentParser(description="Training process")
parse.add_argument("--log", "-l", dest="log", metavar="FILE", help="log file", default="test")
parse.add_argument("--configure", "-c", dest="config", metavar="FILE", help="yaml file", default=r"config/nrha.yaml")
parse.add_argument("--gup_num", "-g", dest="gpus", metavar="INT", default=1, help="The number of gpu")
parse.add_argument("--model_class", "-m", dest="model_class", metavar="TEXT", default="nrha")
parse.add_argument("--mind_type", "-t", dest="mind_type", metavar="TEXT", default="large")
parse.add_argument("--resume", "-r", dest="resume", metavar="INT", help="whether resume or not", default=0)
args = parse.parse_args()
train_news_file, train_behaviors_file = get_path("train", mind_type=args.mind_type)
valid_news_file, valid_behaviors_file = get_path("valid", mind_type=args.mind_type)
test_news_file, test_behaviors_file = get_path("test", mind_type=args.mind_type)
config = {"yaml_name": args.config, "log": args.log, "gpus": int(args.gpus),
          "model_class": args.model_class, "mind_type": args.mind_type}
accelerator = "ddp" if config["gpus"] > 1 else None
hparams = get_params(args.config)
converter = get_converter(hparams.embedding, word_dict_file=hparams.word_dict_file)
train_dataset = TrainingDataset(train_news_file, train_behaviors_file, hparams, converter, npratio=hparams.npratio)
train_dataloader = DataLoader(train_dataset, hparams.batch_size)
# TODO: modify value
interval, epochs = len(train_dataloader) // 8, 5
saved_dir = f"{config['mind_type']}/{config['model_class']}"
ckpt_dir = f"saved/checkpoint/{saved_dir}/{args.log}"
pred_dir = f"saved/prediction/{saved_dir}/{args.log}"
valid_callback = ValidationCallback(valid_news_file, valid_behaviors_file, hparams, converter, ckpt_dir, interval)
# TODO: modify name
tb_logger = pl_loggers.TensorBoardLogger(f"saved/logs/{saved_dir}", name=args.log)
model = NRHATitle(hparams)
# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
trainer = pl.Trainer(gpus=config["gpus"], accelerator=accelerator, max_epochs=epochs, deterministic=True,
                     logger=tb_logger, callbacks=[valid_callback])
trainer.fit(model, train_dataloader)
test_callback = TestCallback(test_news_file, test_behaviors_file, hparams, converter, ckpt_dir, pred_dir)
test_callback.on_fit_end(trainer, model)
