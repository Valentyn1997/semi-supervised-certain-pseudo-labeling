semi-supervised-certain-pseudo-labeling
==============================

Employing uncertanty estimation in pseudo-labelling for semi-supervised image classification methods (Fix-Match). As [vanilla Fix-Match](https://arxiv.org/abs/2001.07685) can suffer from overconfident pseudo-labels, we propose to employ active learning techniques to choose more confidently pseudo-labelled images from unlabelled pool.

Uncertainty estimators:
- [MC Dropout + Wide-Resnet](https://arxiv.org/abs/1506.02142) -- failed, as Dropout is non-compatible with Batch Normalisation (https://arxiv.org/pdf/1801.05134.pdf). Alternative versions with:
    - Uniform Noise Dropout
    - After Batch Normalisations Dropout
- [Spectral Normalisation + Wide-Resnet](https://arxiv.org/pdf/2102.11582.pdf) -- distinguishing epistemic and aleatoric uncertainty.

Uncertainty strategies:
- Entropy
- Mutual Information
- SofMax outputs (max, top-k)
- Log-likelihood of Gaussian mixture, fitted on the last ResNet layer


## MLFLOW server

Starting (from the project's root):
`mlflow server --default-artifact-root='/home/ubuntu/semi_supervised_certain_pseudo_labeling/mlruns/'`

Remote connection via ssh:
`ssh -N -f -L localhost:5000:localhost:5000 <user>@10.195.1.189`

Than one can access [localhost:5000](http://localhost:5000)

## Running Scripts
First one needs to make the virtual environment and install all the requirements:
```console
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Then one can run experiments:
```console
PYTHONPATH=. python3 runnables/train_fixmatch.py data.source='CIFAR10' data.n_labelled=250 optimizer.weight_decay=0.0005
PYTHONPATH=. python3 runnables/train_fixmatch.py data.source='CIFAR100' data.n_labelled=400 optimizer.weight_decay=0.001
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
