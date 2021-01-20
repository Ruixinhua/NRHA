import os

from callbacks import ValidationCallback
from configuration import get_path, get_params, get_argument, get_model_class
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers

from data_loader.training_dataset import TrainingDataset
from torch.utils.data.dataloader import DataLoader
from utils.helpers import get_converter

args = get_argument()
hparams = get_params(args.config, head_num=int(args.head_num), head_dim=int(args.head_dim), model=args.model_class,
                     batch_size=int(args.batch_size))
accelerator = "ddp" if int(args.gpus) > 1 else None
print(args)
train_news_file, train_behaviors_file = get_path("train", mind_type=args.mind_type)
valid_news_file, valid_behaviors_file = get_path("valid", mind_type=args.mind_type)
test_news_file, test_behaviors_file = get_path("test", mind_type=args.mind_type)

converter = get_converter(hparams.embedding, word_dict_file=hparams.word_dict_file)
train_dataset = TrainingDataset(train_news_file, train_behaviors_file, hparams, converter, npratio=hparams.npratio)
train_dataloader = DataLoader(train_dataset, hparams.batch_size)
# TODO: modify value
interval, epochs = len(train_dataloader) // 3, 8
saved_dir = f"{args.mind_type}/{args.model_class}"
ckpt_dir = f"saved/checkpoint/{saved_dir}/{args.log}"
pred_dir = f"saved/prediction/{saved_dir}/{args.log}"
best_model_path = os.path.join(ckpt_dir, "best_model.ckpt")
resume_path = best_model_path if os.path.exists(best_model_path) and args.resume else None
valid_callback = ValidationCallback(valid_news_file, valid_behaviors_file, hparams, converter, ckpt_dir, interval)
# TODO: modify name
tb_logger = pl_loggers.TensorBoardLogger(f"saved/logs/{saved_dir}", name=args.log)
pl.trainer.seed_everything(40)
model_class = get_model_class(args.model_class)
hparams.update(**{"user_embedding_size": len(train_dataset.uid2index)})
# most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
trainer = pl.Trainer(gpus=int(args.gpus), accelerator=accelerator, max_epochs=epochs, deterministic=True,
                     logger=tb_logger, callbacks=[valid_callback], resume_from_checkpoint=resume_path)
trainer.fit(model_class(hparams), train_dataloader)
