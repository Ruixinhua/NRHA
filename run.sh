# -t: dataset type; -m: model type; -n: head number; -d: dimensional number; -c: configuration file; -b: batch size; -r: resume path or use current best model
python train.py -t demo -m nrha_body -n 10 -d 30 -c config/nrha_body.yaml -b 16 -r 1
# -l: directory name of saved path
python inference.py -t large -l body -c config/nrha_body.yaml -m nrha_body -n 10 -d 30