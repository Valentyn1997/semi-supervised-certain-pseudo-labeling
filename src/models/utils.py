from pytorch_lightning.core.step_result import TrainResult
import pandas as pd
import torch
import math
import numpy as np
from src.utils import simple_accuracy
from copy import deepcopy
from torch.optim.lr_scheduler import LambdaLR


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.ema_model.eval()

        self.alpha = alpha
        self.ema_has_module = hasattr(self.ema_model, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.ema_model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema_model.named_buffers()]
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def step(self):
        needs_module = hasattr(self.model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = self.model.state_dict()
            esd = self.ema_model.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.alpha + (1. - self.alpha) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])


class UnlabelledStatisticsLogger:

    def __init__(self, level='image', save_frequency=500, artifacts_path=None):
        self.level = level
        self.batch_dfs = []
        self.save_frequency = save_frequency
        self.artifacts_path = artifacts_path
        self.logging_df = pd.DataFrame()

    def log_statistics(self,
                       u_scores: torch.tensor,
                       u_targets: torch.tensor,
                       u_pseudo_targets: torch.tensor,
                       u_ids: torch.tensor,
                       current_epoch: int,
                       strategy_name=None):
        if self.level == 'batch':
            raise NotImplementedError()
            # Needs to be rewriten to consider u_scores

            # certain_ul_targets = u_targets[thresholding_mask == 1.0].cpu().numpy()
            # all_ul_targets = u_targets.cpu().numpy()

            # result.log('certain_ul_acc', certain_ul_acc, on_epoch=False, on_step=True, sync_dist=True)
            # result.log('all_ul_acc', all_ul_acc, on_epoch=False, on_step=True, sync_dist=True)
            # result.log('max_probs', u_scores.mean(), on_epoch=False, on_step=True, sync_dist=True)
            # result.log('n_certain', thresholding_mask.sum(), on_epoch=False, on_step=True, sync_dist=True)

        elif self.level == 'image':
            batch_df = pd.DataFrame(index=range(len(u_ids)))
            batch_df['image_id'] = u_ids.tolist()
            batch_df['score'] = u_scores.tolist()
            batch_df['correctness'] = (u_pseudo_targets == u_targets).tolist()
            batch_df['epoch'] = current_epoch
            if strategy_name is not None:
                batch_df['strategy'] = strategy_name
            self.batch_dfs.append(batch_df)

    def on_epoch_end(self, current_epoch):
        if self.level:
            for batch_df in self.batch_dfs:
                self.logging_df = self.logging_df.append(batch_df, ignore_index=True)
            self.batch_dfs = []

        if self.level == 'image' and current_epoch % self.save_frequency == 0:
            epochs_range = self.logging_df['epoch'].min(), self.logging_df['epoch'].max()
            csv_path = f'{self.artifacts_path}/epochs_{epochs_range[0]:05d}_{epochs_range[1]:05d}.csv'
            self.logging_df.to_csv(csv_path, index=False)
            self.logging_df = pd.DataFrame()


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)