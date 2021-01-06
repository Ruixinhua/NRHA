# set up configuration
from pytorch_lightning.callbacks import ModelCheckpoint

from configuration import get_path, get_params
import argparse
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from data_loader.base_dataset import BaseDataset
from data_loader.test_dataset import UserDataset, NewsDataset
from data_loader.training_dataset import TrainingDataset
from torch.utils.data.dataloader import DataLoader

from models.nrms import NRMSModel
from utils.helpers import get_converter
from validation_callback import ValidationCallback

parse = argparse.ArgumentParser(description="Training process")
parse.add_argument("--log", "-l", dest="log", metavar="FILE", help="log file", default="test")
parse.add_argument("--configure", "-c", dest="config", metavar="FILE", help="yaml file", default=r"nrha.yaml")
parse.add_argument("--device_id", "-d", dest="device_id", metavar="INT", default=0)
parse.add_argument("--model_class", "-m", dest="model_class", metavar="TEXT", default="nrms")
parse.add_argument("--mind_type", "-t", dest="mind_type", metavar="TEXT", default="small")
parse.add_argument("--resume", "-r", dest="resume", metavar="INT", help="whether resume or not", default=0)
args = parse.parse_args()

train_news_file, train_behaviors_file = get_path("train", mind_type=args.mind_type)
valid_news_file, valid_behaviors_file = get_path("valid", mind_type=args.mind_type)
config = {"yaml_name": args.config, "log": args.log, "device_id": int(args.device_id),
          "model_class": args.model_class, "mind_type": args.mind_type}
hparams = get_params(args.config)
converter = get_converter(hparams.embedding, word_dict_file=hparams.word_dict_file)
train_dataset = TrainingDataset(train_news_file, train_behaviors_file, hparams, converter, npratio=hparams.npratio)
train_dataloader = DataLoader(train_dataset, hparams.batch_size)
valid_callback = ValidationCallback(valid_news_file, valid_behaviors_file, hparams, converter)
model = NRMSModel(hparams)
saved_dir = f"{config['mind_type']}/{config['model_class']}"
checkpoint_callback = ModelCheckpoint(monitor="group_auc", filename="{group_auc:.4f}", save_top_k=3, mode="max",
                                      dirpath=f"saved/checkpoint/{saved_dir}/{args.log}")
tb_logger = pl_loggers.TensorBoardLogger(f"saved/logs/{saved_dir}", name=args.log)
# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
trainer = pl.Trainer(gpus=1, max_epochs=5, deterministic=True, val_check_interval=100, logger=tb_logger,
                     callbacks=[checkpoint_callback, valid_callback])
trainer.fit(model, train_dataloader)
