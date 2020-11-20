#!/bin/bash
#SBATCH --output=./train_uniform.log  # send stdout to outfile
#SBATCH -p debug
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1

cd ..
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:./
ssh -N -f -L localhost:5004:localhost:5005 melnychukv@zandalar.dbs.ifi.lmu.de
python3 ./runnables/train_fixmatch.py -m data.source='CIFAR10' data.n_labelled=250 optimizer.weight_decay=0.0005 data.batch_size.train=64 exp.max_epochs=512 model.ema_decay=0.999 model.drop_rate=0.25 model.drop_type=AfterBNDropout model.certainty_strategy=SoftMax model.threshold=0.95