from pytorch_lightning.core.step_result import TrainResult
import pandas as pd
import torch
import numpy as np
from src.utils import simple_accuracy


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(list(self.model.state_dict().values()), list(self.ema_model.state_dict().values())):
            if len(param.shape) > 0:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)


class UnlabelledStatisticsLogger:

    def __init__(self, level='image', save_frequency=500, artifacts_path=None):
        self.level = level
        self.batch_dfs = []
        self.save_frequency = save_frequency
        self.artifacts_path = artifacts_path
        self.logging_df = pd.DataFrame()

    def log_statistics(self, result: TrainResult,
                       thresholding_mask: torch.tensor,
                       thresholding_score: torch.tensor,
                       uw_logits: torch.tensor,
                       u_targets: torch.tensor,
                       u_pseudo_targets: torch.tensor,
                       u_ids: torch.tensor,
                       current_epoch: int):
        if self.level == 'batch':
            certain_logits_uw = uw_logits[thresholding_mask == 1.0].cpu().numpy()
            certain_ul_targets = u_targets[thresholding_mask == 1.0].cpu().numpy()
            all_logits_uw = uw_logits.cpu().numpy()
            all_ul_targets = u_targets.cpu().numpy()

            certain_ul_acc = simple_accuracy(certain_logits_uw, certain_ul_targets)
            certain_ul_acc = torch.tensor(0.0) if np.isnan(certain_ul_acc) else torch.tensor(certain_ul_acc)

            all_ul_acc = simple_accuracy(all_logits_uw, all_ul_targets)
            all_ul_acc = torch.tensor(all_ul_acc)

            result.log('certain_ul_acc', certain_ul_acc, on_epoch=False, on_step=True, sync_dist=True)
            result.log('all_ul_acc', all_ul_acc, on_epoch=False, on_step=True, sync_dist=True)
            result.log('max_probs', thresholding_score.mean(), on_epoch=False, on_step=True, sync_dist=True)
            result.log('n_certain', thresholding_mask.sum(), on_epoch=False, on_step=True, sync_dist=True)

        elif self.level == 'image':
            batch_df = pd.DataFrame(index=range(len(u_ids)))
            batch_df['image_id'] = u_ids.tolist()
            batch_df['score'] = thresholding_score.tolist()
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
