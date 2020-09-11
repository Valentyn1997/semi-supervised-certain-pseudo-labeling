#!/bin/bash
#SBATCH --output=./train1.log  # send stdout to outfile
#SBATCH -p debug


cd ..
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:./
ssh -N -f -L localhost:5000:localhost:5000 ubuntu@10.195.1.189
apt install sshfs
sshfs ubuntu@10.195.1.189:/home/ubuntu/semi_supervised_certain_pseudo_labeling/mlruns /home/ubuntu/semi_supervised_certain_pseudo_labeling/mlruns  # Mounting remote folder via ssh
python3 ./runnables/train_fixmatch.py data.source='CIFAR10' data.n_labelled=250 optimizer.weight_decay=0.0005 data.batch_size.train=64 exp.max_epochs=20000