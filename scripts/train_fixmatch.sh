#!/bin/bash
#SBATCH --output=./train.log  # send stdout to outfile
#SBATCH -p debug
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH -w worker-2

cd ..
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:./
ssh -N -f -L localhost:5005:localhost:5005 melnychukv@zandalar.dbs.ifi.lmu.de
python3 ./runnables/train_fixmatch.py -m data.source='STL10' data.n_labelled=1000 optimizer.weight_decay=0.0005 model.drop_rate=0.0 model.drop_type=Dropout model.certainty_strategy=Entropy model.multi_strategy=False exp.log_ul_statistics=False model.ema_decay=0.999 exp.precision=16 data.val_ratio=0.0 exp.checkpoint=False model.threshold=0.85 model.wrn.depth=37 model.wrn.widen_factor=2 model.wrn.scale=4 data.weak_aug.crop_size=96 exp.max_epochs=512
#python3 ./runnables/train_fixmatch.py -m data.source='CIFAR100' data.n_labelled=10000 optimizer.weight_decay=0.001 model.drop_rate=0.0 model.drop_type=Dropout model.certainty_strategy=Entropy model.multi_strategy=False exp.log_ul_statistics=False model.ema_decay=0.999 exp.precision=16 data.val_ratio=0.0 exp.checkpoint=False model.threshold=0.85 model.wrn.depth=28 model.wrn.widen_factor=8