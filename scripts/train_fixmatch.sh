#!/bin/bash
#SBATCH --output=./train1.log  # send stdout to outfile
#SBATCH -p debug
#SBATCH --gres=gpu:2

cd ..
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:./
ssh -N -f -L localhost:5004:localhost:5005 melnychukv@zandalar.dbs.ifi.lmu.de
python3 ./runnables/train_fixmatch.py -m data.source='CIFAR10' data.n_labelled=250 optimizer.weight_decay=0.0005 data.batch_size.train=64 exp.max_epochs=40000 model.ema_decay=0.999 model.drop_rate=0.25 model.drop_type=UniformDropout model.certainty_strategy=SoftMax