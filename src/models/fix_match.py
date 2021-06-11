import importlib

import torchvision.utils as vutils
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from copy import deepcopy
from pytorch_lightning.core.step_result import TrainResult, EvalResult
import pandas as pd

from src.models.dropouts import uniform_dropout
from src.models.certainty_strategy import AbstractStrategy, BALDCertainty
from src.utils import simple_accuracy
from src.models.utils import WeightEMA, UnlabelledStatisticsLogger, get_cosine_schedule_with_warmup
from src.models.backbones import WideResNet
from src.models.running_gmm import RunningGMM
from src.models.certainty_strategy import MultiStrategies
from src.data.ssl_datasets import SLLDatasetsCollection, FixMatchCompositeTrainDataset


class FixMatch(LightningModule):
    def __init__(self, args: DictConfig, datasets_collection: SLLDatasetsCollection, artifacts_path: str = None):
        super().__init__()
        self.lr = None  # Placeholder for auto_lr_find
        self.artifacts_path = artifacts_path
        self.datasets_collection = datasets_collection
        self.hparams = args  # Will be logged to mlflow

        if self.hparams.model.drop_type == 'Dropout':
            self.model = WideResNet(depth=args.model.wrn.depth, widen_factor=args.model.wrn.widen_factor,
                                    drop_rate=self.hparams.model.drop_rate,
                                    scale=self.hparams.model.wrn.scale,
                                    spectral_normalise=self.hparams.model.spectral_norm,
                                    num_classes=len(datasets_collection.classes))
        elif self.hparams.model.drop_type == 'DropConnect':
            self.model = WideResNet(depth=args.model.wrn.depth, widen_factor=args.model.wrn.widen_factor,
                                    drop_rate=0.0,
                                    scale=self.hparams.model.wrn.scale,
                                    weight_dropout=self.hparams.model.drop_rate,
                                    spectral_normalise=self.hparams.model.spectral_norm,
                                    num_classes=len(datasets_collection.classes))
        elif self.hparams.model.drop_type == 'AlphaDropout':
            self.model = WideResNet(depth=args.model.wrn.depth, widen_factor=args.model.wrn.widen_factor,
                                    drop_rate=self.hparams.model.drop_rate,
                                    scale=self.hparams.model.wrn.scale,
                                    spectral_normalise=self.hparams.model.spectral_norm,
                                    num_classes=len(datasets_collection.classes), dropout_method=F.alpha_dropout)
        elif self.hparams.model.drop_type == 'AfterBNDropout':
            self.model = WideResNet(depth=args.model.wrn.depth, widen_factor=args.model.wrn.widen_factor,
                                    num_classes=len(datasets_collection.classes),
                                    scale=self.hparams.model.wrn.scale,
                                    spectral_normalise=self.hparams.model.spectral_norm,
                                    after_bn_drop_rate=self.hparams.model.drop_rate)
        elif self.hparams.model.drop_type == 'UniformDropout':
            self.model = WideResNet(depth=args.model.wrn.depth, widen_factor=args.model.wrn.widen_factor,
                                    drop_rate=self.hparams.model.drop_rate,
                                    scale=self.hparams.model.wrn.scale,
                                    spectral_normalise=self.hparams.model.spectral_norm,
                                    num_classes=len(datasets_collection.classes), dropout_method=uniform_dropout)
        else:
            raise NotImplementedError

        self.ema_model = deepcopy(self.model)
        self.best_model = self.ema_model  # Placeholder for checkpointing
        self.ema_optimizer = WeightEMA(self.model, self.ema_model, alpha=self.hparams.model.ema_decay)
        self.ul_w_logger = UnlabelledStatisticsLogger(level=self.hparams.exp.log_ul_w_statistics,
                                                      save_frequency=self.hparams.exp.log_statistics_freq,
                                                      artifacts_path=self.artifacts_path,
                                                      name='unlabelled_weak')
        self.ul_s_logger = UnlabelledStatisticsLogger(level=self.hparams.exp.log_ul_s_statistics,
                                                      save_frequency=self.hparams.exp.log_statistics_freq,
                                                      artifacts_path=self.artifacts_path,
                                                      name='unlabelled_strong')
        self.val_logger = UnlabelledStatisticsLogger(level=self.hparams.exp.log_val_statistics,
                                                     save_frequency=self.hparams.exp.log_statistics_freq,
                                                     artifacts_path=self.artifacts_path,
                                                     name='val')

        # Initialisation of certainty strategy
        strategy_class_name = self.hparams.model.certainty_strategy
        module = importlib.import_module("src.models.certainty_strategy")
        strategy = getattr(module, strategy_class_name)()
        assert isinstance(strategy, AbstractStrategy)
        self.strategy = strategy

        # Setting initial threshold to 1.0, if choose_threshold_on_val
        if self.hparams.model.choose_threshold_on_val:
            self.hparams.model.threshold = 1.0

        # GMM & Aleatoric uncertainty
        if self.hparams.model.features_gmm:
            self.running_gmm = RunningGMM(self.model.channels, len(self.datasets_collection.classes))


    def prepare_data(self):
        self.train_dataset = FixMatchCompositeTrainDataset(self.datasets_collection.train_l_dataset,
                                                           self.datasets_collection.train_ul_dataset,
                                                           self.hparams.model.mu,
                                                           self.hparams.data.steps_per_epoch * self.hparams.data.batch_size.train,
                                                           self.datasets_collection.val_dataset if self.hparams.model.choose_threshold_on_val else None)
        if self.hparams.data.val_ratio > 0.0:
            self.val_dataset = self.datasets_collection.val_dataset
        else:
            self.val_dataset = self.datasets_collection.test_dataset
        self.test_dataset = self.datasets_collection.test_dataset
        self.hparams.data_size = DictConfig({
            'train': {'lab': len(self.train_dataset.l_dataset), 'unlab': len(self.train_dataset.ul_dataset)},
            'val': len(self.val_dataset),
            'test': len(self.test_dataset)
        })

    def configure_optimizers(self):
        if self.lr is not None:  # After auto_lr_find
            self.hparams.optimizer.lr = self.lr

        no_decay = ['bias', 'bn']
        grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.hparams.optimizer.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = SGD(grouped_parameters,
                        lr=self.hparams.optimizer.lr,
                        momentum=self.hparams.optimizer.momentum,
                        nesterov=self.hparams.optimizer.nesterov)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    self.hparams.optimizer.warmup_steps,
                                                    len(self.train_dataloader()) * self.hparams.exp.max_epochs)
        lr_dict = {
            'scheduler': scheduler,
            'interval': 'step'
        }
        return [optimizer], [lr_dict]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams.data.batch_size.train, num_workers=5,
                          drop_last=self.hparams.exp.drop_last_batch)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.hparams.data.batch_size.val, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, shuffle=False, batch_size=self.hparams.data.batch_size.test, num_workers=4)

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.ema_optimizer.step()

    def forward(self, batch, model=None, return_feature_map=False):
        if model is None:
            logits, feature_map = self.model(batch)
        elif model == 'best':
            logits, feature_map = self.best_model(batch)
        elif model == 'ema':
            logits, feature_map = self.ema_model(batch)
        else:
            raise NotImplementedError()

        if return_feature_map:
            return logits, feature_map
        else:
            return logits

    @staticmethod
    def _split_batch(composite_batch):
        l_targets = composite_batch[0][0][1]
        l_images = composite_batch[0][0][0]

        u_targets = torch.cat([item[1] for item in composite_batch[1]])  # For later checks
        u_ids = torch.cat([item[2] for item in composite_batch[1]])
        uw_images = torch.cat([item[0][0] for item in composite_batch[1]])
        us_images = torch.cat([item[0][1] for item in composite_batch[1]])

        if len(composite_batch) > 2:
            v_targets = composite_batch[2][0][1]
            v_images = composite_batch[2][0][0]
            v_ids = composite_batch[2][0][2]
        else:
            v_targets, v_images, v_ids = None, None, None

        return l_targets, l_images, u_targets, u_ids, uw_images, us_images, v_targets, v_images, v_ids

    def training_step(self, composite_batch, batch_ind):
        l_targets, l_images, u_targets, u_ids, uw_images, us_images, v_targets, v_images, v_ids \
            = self._split_batch(composite_batch)

        inputs = torch.cat((l_images, us_images, uw_images)).type_as(l_images)
        logits, feature_map = self(inputs, return_feature_map=True)  # !!! IMPORTANT FOR SIMULTANEOUS BATCHNORM UPDATE !!!

        # Supervised loss
        batch_size = l_images.shape[0]
        l_logits = logits[:batch_size]
        l_loss = F.cross_entropy(l_logits, l_targets, reduction='mean')

        # GMM / Aleatoric uncertainty
        if self.hparams.model.features_gmm:
            l_feature_map = feature_map[:batch_size].detach()
            self.running_gmm.update_running(l_feature_map, l_targets)
            _, uw_feature_map = feature_map[batch_size:].detach().chunk(2)
            uw_log_probs = self.running_gmm.log_prob(uw_feature_map)

        # Unsupervised loss
        us_logits, uw_logits = logits[batch_size:].chunk(2)
        uw_logits = uw_logits.detach()
        with torch.no_grad():
            # Unlabelled batch
            if self.strategy.is_ensemble:
                self.model.disable_batch_norm_update()
                uw_logits_list = [self(uw_images).detach() for _ in range(self.hparams.model.T - 1)]  # First pass already done
                uw_logits = torch.stack(uw_logits_list + [uw_logits])
                assert not (uw_logits[0] == uw_logits[1]).all()  # Checking if logits are actually different
                self.model.enable_batch_norm_update()

        # get the value of the max, and the index (between 1 and 10 for CIFAR10 for example)
        uw_pseudo_soft_targets = torch.softmax(uw_logits, dim=-1)
        u_scores, u_pseudo_targets = self.strategy.get_certainty_and_label(softmax_outputs=uw_pseudo_soft_targets)
        assert not any(torch.isnan(u_scores))

        # Validation / strongly augmented logging
        with torch.no_grad():
            self.model.disable_batch_norm_update()
            if self.strategy.is_ensemble:
                v_logits_list = [self(v_images).detach() for _ in range(self.hparams.model.T)]
                v_logits = torch.stack(v_logits_list)

                us_logits_list = [self(us_images).detach() for _ in range(self.hparams.model.T)]
                us_logits_ = torch.stack(us_logits_list)

            else:
                us_logits_ = self(us_images).detach()
                if self.hparams.model.choose_threshold_on_val:
                    v_logits = self(v_images).detach()

            self.model.enable_batch_norm_update()

        if self.hparams.model.choose_threshold_on_val:
            v_pseudo_soft_targets = torch.softmax(v_logits, dim=-1)
            v_scores, v_pseudo_targets = self.strategy.get_certainty_and_label(softmax_outputs=v_pseudo_soft_targets)

        us_pseudo_soft_targets = torch.softmax(us_logits_, dim=-1)
        us_scores, us_pseudo_targets = self.strategy.get_certainty_and_label(softmax_outputs=us_pseudo_soft_targets)

        # Statistics logging
        self.ul_w_logger.log_statistics(u_scores, u_targets, u_pseudo_targets, u_ids, current_epoch=self.trainer.current_epoch)
        self.ul_s_logger.log_statistics(us_scores, u_targets, us_pseudo_targets, u_ids, current_epoch=self.trainer.current_epoch)

        if self.hparams.model.choose_threshold_on_val:
            self.val_logger.log_statistics(v_scores, v_targets, v_pseudo_targets, v_ids, current_epoch=self.trainer.current_epoch,
                                           current_globalstep=self.trainer.global_step)

        # Unlabelled masking

        # Basic (Aleatoric uncertainty) -- Softmax Entropy
        mask = u_scores.ge(self.hparams.model.threshold).float()

        # Epistemic uncertainty
        if self.hparams.model.features_gmm:
            if self.hparams.dynamic_log_prob_threshold:
                log_prob_threshold = self.running_gmm.log_prob(l_feature_map).mean()
            else:
                log_prob_threshold = self.hparams.model.log_prob_threshold
            log_lik_mask = uw_log_probs.ge(log_prob_threshold)

            if self.hparams.model.mask_operation == 'or':
                mask = torch.logical_or(mask, log_lik_mask).float()
            elif self.hparams.model.mask_operation == 'and':
                mask = torch.logical_and(mask, log_lik_mask).float()
            else:
                raise NotImplementedError()

            if self.hparams.model.u_update_gmm:
                self.running_gmm.update_running(uw_feature_map[log_lik_mask], u_pseudo_targets[log_lik_mask])

        # Extra mask for BALD uncertainty
        if isinstance(self.strategy, BALDCertainty):
            conf_scores = self.strategy.get_model_confidence(softmax_outputs=uw_pseudo_soft_targets)
            conf_mask = conf_scores.ge(self.hparams.model.conf_threshold)
            mask = torch.logical_and(mask, conf_mask).float()

        u_loss = (F.cross_entropy(us_logits, u_pseudo_targets, reduction='none') * mask).mean()

        # Train loss / labelled accuracy
        loss = l_loss + self.hparams.model.lambda_u * u_loss

        result = TrainResult(minimize=loss)
        result.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        result.log('train_loss_l', l_loss, on_epoch=True, on_step=False, sync_dist=True)
        result.log('train_loss_ul', u_loss, on_epoch=True, on_step=False, sync_dist=True)
        result.log('train_mean_mask', mask.mean(), on_epoch=True, on_step=False, sync_dist=True)
        if self.hparams.model.features_gmm:
            result.log('train_mean_log_lik_ul', uw_log_probs.mean(), on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
            result.log('train_std_log_lik_ul', uw_log_probs.std(), on_epoch=True, on_step=False, sync_dist=True)
            if self.hparams.dynamic_log_prob_threshold:
                result.log('log_prob_threshold', uw_log_probs.mean(), on_epoch=True, on_step=False, sync_dist=True,
                           prog_bar=True)
        if self.hparams.exp.log_pl_accuracy:
            result.log('train_pl_accuracy', (u_pseudo_targets[mask.bool()] == u_targets[mask.bool()]).float().mean(),
                       on_epoch=True, on_step=False, sync_dist=True)
        return result

    def validation_step(self, batch, batch_ind):
        images, targets, ids = batch
        logits = self(images, model='ema')
        loss = F.cross_entropy(logits, targets, reduction='mean')
        result = EvalResult(early_stop_on=loss, checkpoint_on=loss)
        result.log('val_loss', loss, sync_dist=True)
        result.log_dict(self.calculate_metrics(logits, targets, prefix='val'), sync_dist=True)
        return result

    def test_step(self, batch, batch_ind):
        images, targets, ids = batch
        logits = self(images, model='best')
        loss = F.cross_entropy(logits, targets, reduction='mean')
        result = EvalResult()
        result.log('test_loss', loss, sync_dist=True)
        result.log_dict(self.calculate_metrics(logits, targets, prefix='test'), sync_dist=True)
        return result

    @staticmethod
    def calculate_metrics(logits, targets, prefix):
        return {prefix + '_acc': simple_accuracy(logits, targets)}

    def on_train_batch_end(self, batch, batch_idx, dataloader_idx) -> None:
        if self.hparams.model.choose_threshold_on_val and \
                (self.trainer.global_step + 1) % self.hparams.model.choose_threshold_on_val_freq == 0:
            self.hparams.model.threshold = self.val_logger.get_optimal_threshold(self.hparams.model.choose_threshold_on_val_freq,
                                                                                 accuracy=self.hparams.model.choose_threshold_on_val_acc)

            if self.hparams.exp.logging:
                self.trainer.logger.log_metrics({'threshold': self.hparams.model.threshold}, step=self.trainer.global_step)


    def on_epoch_end(self) -> None:
        self.ul_w_logger.on_epoch_end(current_epoch=self.trainer.current_epoch + 1)
        self.ul_s_logger.on_epoch_end(current_epoch=self.trainer.current_epoch + 1)
        self.val_logger.on_epoch_end(current_epoch=self.trainer.current_epoch + 1)

    def on_train_start(self) -> None:
        if self.hparams.exp.log_artifacts:  # Plotting one batch of images
            composite_batch = next(iter(self.train_dataloader()))

            l_images = composite_batch[0][0][0]
            uw_images = torch.cat([item[0][0] for item in composite_batch[1]])
            us_images = torch.cat([item[0][1] for item in composite_batch[1]])

            img = vutils.make_grid(l_images, padding=5, normalize=True)
            img_path = f'{self.artifacts_path}/train_labelled.png'
            vutils.save_image(img, img_path)

            img = vutils.make_grid(us_images, padding=5, normalize=True, nrow=16)
            img_path = f'{self.artifacts_path}/train_unlabelled_strong.png'
            vutils.save_image(img, img_path)

            img = vutils.make_grid(uw_images, padding=5, normalize=True, nrow=16)
            img_path = f'{self.artifacts_path}/train_unlabelled_weak.png'
            vutils.save_image(img, img_path)


class MultiStrategyFixMatch(FixMatch):

    def __init__(self, args: DictConfig, datasets_collection: SLLDatasetsCollection, artifacts_path: str = None):
        super().__init__(args, datasets_collection, artifacts_path)

        # Re-initialisation of strategies
        self.decision_strategy = self.strategy.__class__.__name__
        self.strategy = MultiStrategies()

    def on_train_batch_end(self, batch, batch_idx, dataloader_idx) -> None:
        if self.hparams.model.choose_threshold_on_val and \
                (self.trainer.global_step + 1) % self.hparams.model.choose_threshold_on_val_freq == 0:
            self.hparams.model.threshold = self.val_logger.get_optimal_threshold(self.hparams.model.choose_threshold_on_val_freq,
                                                                                 strategy_name=self.decision_strategy,
                                                                                 accuracy=self.hparams.model.choose_threshold_on_val_acc)

            if self.hparams.exp.logging:
                self.trainer.logger.log_metrics({'threshold': self.hparams.model.threshold}, step=self.trainer.global_step)

    def training_step(self, composite_batch, batch_ind):
        l_targets, l_images, u_targets, u_ids, uw_images, us_images, v_targets, v_images, v_ids \
            = self._split_batch(composite_batch)

        inputs = torch.cat((l_images, us_images, uw_images)).type_as(l_images)
        logits = self(inputs)  # !!! IMPORTANT FOR SIMULTANEOUS BATCHNORM UPDATE !!!

        # Supervised loss
        batch_size = l_images.shape[0]
        l_logits = logits[:batch_size]
        l_loss = F.cross_entropy(l_logits, l_targets, reduction='mean')

        # Unsupervised loss
        us_logits, uw_logits = logits[batch_size:].chunk(2)
        uw_logits = uw_logits.detach()

        with torch.no_grad():
            self.model.disable_batch_norm_update()
            uw_logits_list = [self(uw_images).detach() for _ in range(self.hparams.model.T - 1)]
            uw_logits_ensemble = torch.stack(uw_logits_list + [uw_logits])

            v_logits = self(v_images).detach()
            v_logits_list = [self(v_images).detach() for _ in range(self.hparams.model.T)]
            v_logits_ensemble = torch.stack(v_logits_list)

            us_logits_ = self(us_images).detach()
            us_logits_list = [self(us_images).detach() for _ in range(self.hparams.model.T)]
            us_logits_ensemble = torch.stack(us_logits_list)

            self.model.enable_batch_norm_update()
            assert not (uw_logits_ensemble[0] == uw_logits_ensemble[1]).all()  # Checking if logits are actually different

        for strategy_name in self.strategy.strategies.keys():

            if self.strategy.strategies[strategy_name].is_ensemble:
                uw_pseudo_soft_targets = torch.softmax(uw_logits_ensemble, dim=-1)
                us_pseudo_soft_targets = torch.softmax(us_logits_ensemble, dim=-1)
                v_pseudo_soft_targets = torch.softmax(v_logits_ensemble, dim=-1)
            else:
                uw_pseudo_soft_targets = torch.softmax(uw_logits, dim=-1)
                us_pseudo_soft_targets = torch.softmax(us_logits_, dim=-1)
                v_pseudo_soft_targets = torch.softmax(v_logits, dim=-1)

            u_scores, u_pseudo_targets = self.strategy.get_certainty_and_label(uw_pseudo_soft_targets, strategy_name)
            us_scores, us_pseudo_targets = self.strategy.get_certainty_and_label(us_pseudo_soft_targets, strategy_name)
            v_scores, v_pseudo_targets = self.strategy.get_certainty_and_label(v_pseudo_soft_targets, strategy_name)
            # Unlabelled statistics
            self.ul_w_logger.log_statistics(u_scores, u_targets, u_pseudo_targets, u_ids,
                                            current_epoch=self.trainer.current_epoch,
                                            strategy_name=strategy_name)
            self.ul_s_logger.log_statistics(us_scores, u_targets, us_pseudo_targets, u_ids,
                                            current_epoch=self.trainer.current_epoch,
                                            strategy_name=strategy_name)
            self.val_logger.log_statistics(v_scores, v_targets, v_pseudo_targets, v_ids,
                                           current_epoch=self.trainer.current_epoch,
                                           strategy_name=strategy_name,
                                           current_globalstep=self.trainer.global_step)

            if strategy_name == self.decision_strategy:
                mask = u_scores.ge(self.hparams.model.threshold).float()
                u_loss = (F.cross_entropy(us_logits, u_pseudo_targets, reduction='none') * mask).mean()

        # Train loss / labelled accuracy
        loss = l_loss + self.hparams.model.lambda_u * u_loss

        result = TrainResult(minimize=loss)
        result.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        result.log('train_loss_l', l_loss, on_epoch=True, on_step=False, sync_dist=True)
        result.log('train_loss_ul', u_loss, on_epoch=True, on_step=False, sync_dist=True)
        result.log('train_mean_mask', mask.mean(), on_epoch=True, on_step=False, sync_dist=True)
        return result
