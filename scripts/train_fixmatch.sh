#!/bin/bash
#SBATCH --output=./train.log  # send stdout to outfile
#SBATCH -p debug
#SBATCH --gres=gpu:2

cd ..
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:./
#module load cuda42/toolkit
ssh -N -f -L localhost:5000:localhost:5000 ubuntu@10.195.1.189
python3 ./runnables/train_fixmatch.py data.source='CIFAR10' data.n_labelled=250 optimizer.weight_decay=0.0005 data.batch_size.train=64 exp.max_epochs=20000