import os
import torch

from callbacks import TestCallback, ValidationCallback
from configuration import get_path, get_params, get_argument, get_model_class
from utils.helpers import Converter

args = get_argument()
hparams = get_params(args.config, head_num=int(args.head_num), head_dim=int(args.head_dim), model=args.model_class)
print(hparams)
test_news_file, test_behaviors_file = get_path("test", mind_type=args.mind_type)
converter = Converter(hparams).converter
saved_dir = f"{args.mind_type}/{args.model_class}"
ckpt_dir = f"saved/checkpoint/{saved_dir}/{args.log}"
pred_dir = f"saved/prediction/{saved_dir}/{args.log}"
best_model_path = os.path.join(ckpt_dir, "best_model.ckpt")
if args.resume and os.path.exists(args.resume):
    best_model_path = args.resume
load_path = best_model_path
if not os.path.exists(load_path):
    raise FileNotFoundError
test_methods = ["linear", "conv", "multi_head", "lstm", "bilstm", "empty"]
att_methods = ["layer_att", "head_att"]
hparams.update(**{"news_layer": test_methods[int(args.news_layer)],
                  "user_layer": test_methods[int(args.user_layer)],
                  "att_method": att_methods[0]})
hparams.update(**{"n_layers": args.n_layers})
model_class = get_model_class(args.model_class)
model = model_class(hparams).load_from_checkpoint(load_path, hparams=hparams)
test_callback = TestCallback(test_news_file, test_behaviors_file, hparams, converter, ckpt_dir, pred_dir)
# valid_news_file, valid_behaviors_file = get_path("valid", mind_type=args.mind_type)
# valid_callback = ValidationCallback(valid_news_file, valid_behaviors_file, hparams, converter, ckpt_dir, 0)
print("Inference Begin")
model.eval()
# with torch.no_grad():
#     validate on validation set
#     print(valid_callback.validation(model.to(torch.device("cuda"))))
test_callback.inference(model.to(torch.device("cuda")))
print("End of inference")
