import os

from callbacks import TestCallback
from configuration import get_path, get_params
import argparse
from models.base import BaseModel
from utils.helpers import get_converter


print("Hello world")
parse = argparse.ArgumentParser(description="Training process")
parse.add_argument("--log", "-l", dest="log", metavar="FILE", help="log file", default="test")
parse.add_argument("--configure", "-c", dest="config", metavar="FILE", help="yaml file", default=r"config/nrha.yaml")
parse.add_argument("--model_class", "-m", dest="model_class", metavar="TEXT", default="nrha")
parse.add_argument("--mind_type", "-t", dest="mind_type", metavar="TEXT", default="large")
args = parse.parse_args()
config = {"yaml_name": args.config, "log": args.log, "model_class": args.model_class, "mind_type": args.mind_type}
hparams = get_params(args.config)
test_news_file, test_behaviors_file = get_path("test", mind_type=args.mind_type)
converter = get_converter(hparams.embedding, word_dict_file=hparams.word_dict_file)
saved_dir = f"{config['mind_type']}/{config['model_class']}"
ckpt_dir = f"saved/checkpoint/{saved_dir}/{args.log}"
pred_dir = f"saved/prediction/{saved_dir}/{args.log}"

model = BaseModel(hparams).load_from_checkpoint(os.path.join(ckpt_dir, "best_model.ckpt"), hparams=hparams)
test_callback = TestCallback(test_news_file, test_behaviors_file, hparams, converter, ckpt_dir, pred_dir)
print("Inference Begin")
test_callback.inference(model)
