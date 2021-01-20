import os
import torch

from callbacks import TestCallback
from configuration import get_path, get_params, get_argument, get_model_class
from models.base import BaseModel
from models.nrha_title import NRHATitle
from utils.helpers import get_converter

args = get_argument()
hparams = get_params(args.config, head_num=int(args.head_num), head_dim=int(args.head_dim))
test_news_file, test_behaviors_file = get_path("test", mind_type=args.mind_type)
converter = get_converter(hparams.embedding, word_dict_file=hparams.word_dict_file)
saved_dir = f"{args.mind_type}/{args.model_class}"
ckpt_dir = f"saved/checkpoint/{saved_dir}/{args.log}"
pred_dir = f"saved/prediction/{saved_dir}/{args.log}"
model_class = get_model_class(args.model_class)
model = model_class(hparams).load_from_checkpoint(os.path.join(ckpt_dir, "best_model.ckpt"), hparams=hparams)
test_callback = TestCallback(test_news_file, test_behaviors_file, hparams, converter, ckpt_dir, pred_dir)
print("Inference Begin")
test_callback.inference(model.to(torch.device("cuda")))
print("End of inference")
