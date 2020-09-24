from pytorch_lightning.core.step_result import TrainResult
import pandas as pd
import torch
from baal.bayesian import MCDropoutConnectModule
import numpy as np
from src.utils import simple_accuracy


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999, weight_decay=None):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.weight_decay = weight_decay

        params = self.model.parent_module.state_dict() if isinstance(self.model, MCDropoutConnectModule) \
            else self.model.state_dict()
        ema_params = self.ema_model.state_dict()

        for key in params.keys():
            if key in ema_params:
                ema_params[key].data.copy_(params[key].data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha

        params = self.model.parent_module.state_dict() if isinstance(self.model, MCDropoutConnectModule) \
            else self.model.state_dict()
        ema_params = self.ema_model.state_dict()

        for key in params.keys():
            if key in ema_params:
                if len(params[key].shape) > 0:
                    ema_params[key].mul_(self.alpha)
                    ema_params[key].add_(params[key] * one_minus_alpha)

        if self.weight_decay is not None:
            for param in params.values():
                if len(param.shape) > 0:
                    param.mul_(1 - self.weight_decay)


class UnlabelledStatisticsLogger:

    def __init__(self, level='image', save_frequency=500, artifacts_path=None):
        self.level = level
        self.batch_dfs = []
        self.save_frequency = save_frequency
        self.artifacts_path = artifacts_path
        self.logging_df = pd.DataFrame()

    def log_statistics(self, result: TrainResult,
                       thresholding_mask: torch.tensor,
                       u_scores: torch.tensor,
                       u_targets: torch.tensor,
                       u_pseudo_targets: torch.tensor,
                       u_ids: torch.tensor,
                       current_epoch: int):
        if self.level == 'batch':
            # Needs to be rewriten to consider u_scores

            # certain_ul_targets = u_targets[thresholding_mask == 1.0].cpu().numpy()
            # all_ul_targets = u_targets.cpu().numpy()

            # result.log('certain_ul_acc', certain_ul_acc, on_epoch=False, on_step=True, sync_dist=True)
            # result.log('all_ul_acc', all_ul_acc, on_epoch=False, on_step=True, sync_dist=True)
            result.log('max_probs', u_scores.mean(), on_epoch=False, on_step=True, sync_dist=True)
            result.log('n_certain', thresholding_mask.sum(), on_epoch=False, on_step=True, sync_dist=True)

        elif self.level == 'image':
            batch_df = pd.DataFrame(index=range(len(u_ids)))
            batch_df['image_id'] = u_ids.tolist()
            batch_df['score'] = u_scores.tolist()
            batch_df['correctness'] = (u_pseudo_targets == u_targets).tolist()
            batch_df['epoch'] = current_epoch
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
