import logging
from scipy.stats import pearsonr, spearmanr
import torch
import random
import numpy as np
from omegaconf import DictConfig
from copy import deepcopy
from pytorch_lightning.callbacks import ModelCheckpoint


logger = logging.getLogger(__name__)


def set_seed(args: DictConfig):
    random.seed(args.exp.seed)
    np.random.seed(args.exp.seed)
    torch.manual_seed(args.exp.seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.exp.seed)


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    # modify saving
    def _save_model(self, filepath, trainer, pl_module):
        self.model.best_model = deepcopy(self.model.ema_model)
        if self.model.hparams.exp.logging:
            self.model.trainer.logger.log_metrics({'best_epoch': self.model.trainer.current_epoch + 1},
                                                  step=self.model.trainer.global_step)


def simple_accuracy(logits, labels):
    preds = np.argmax(logits, axis=1)
    return (preds == labels).mean()
