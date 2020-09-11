import torchvision.utils as vutils
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
import pandas as pd
from pytorch_lightning.core.step_result import TrainResult

from src.utils import simple_accuracy
from src.models.utils import WeightEMA
from src.models.backbones import WideResNet
from src.data.ssl_datasets import SLLDatasetsCollection, FixMatchCompositeTrainDataset


class FixMatch(LightningModule):
    def __init__(self, args: DictConfig, datasets_collection: SLLDatasetsCollection, artifacts_path: str = None,
                 run_id: str = None):
        super().__init__()
        self.lr = None  # Placeholder for auto_lr_find
        self.artifacts_path = artifacts_path
        self.run_id = run_id
        self.datasets_collection = datasets_collection
        self.model = WideResNet(depth=28, widen_factor=2, drop_rate=0.0, num_classes=len(datasets_collection.classes))
        self.ema_model = WideResNet(depth=28, widen_factor=2, drop_rate=0.0, num_classes=len(datasets_collection.classes))
        self.best_model = self.model  # Placeholder for checkpointing
        self.hparams = args  # Will be logged to mlflow
        self.ema_optimizer = WeightEMA(self.model, self.ema_model, alpha=self.hparams.model.ema_decay)
        self.logging_df = pd.DataFrame()

    def prepare_data(self):
        self.train_dataset = FixMatchCompositeTrainDataset(self.datasets_collection.train_l_dataset,
                                                           self.datasets_collection.train_ul_dataset,
                                                           self.hparams.model.mu)
        self.val_dataset = self.datasets_collection.val_dataset
        self.test_dataset = self.datasets_collection.test_dataset
        self.hparams.data_size = DictConfig({
            'train': {
                'lab': len(self.train_dataset.l_dataset),
                'unlab': len(self.train_dataset.ul_dataset)
            },
            'val': len(self.val_dataset),
            'test': len(self.test_dataset)
        })

    def configure_optimizers(self):
        if self.lr is not None:  # After auto_lr_find
            self.hparams.optimizer.lr = self.lr
        return SGD(self.model.parameters(), lr=self.hparams.optimizer.lr, momentum=self.hparams.optimizer.momentum,
                   nesterov=self.hparams.optimizer.nesterov, weight_decay=self.hparams.optimizer.weight_decay)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams.data.batch_size.train, num_workers=2,
                          drop_last=self.hparams.exp.drop_last_batch)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, shuffle=False, batch_size=self.hparams.data.batch_size.val)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, shuffle=False, batch_size=self.hparams.data.batch_size.test)

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.ema_optimizer.step()

    def forward(self, batch, model=None):
        if model is None:
            return self.model(batch)
        elif model == 'best':
            return self.best_model(batch)
        elif model == 'ema':
            return self.ema_model(batch)

    def training_step(self, composite_batch, batch_idx):
        l_targets = composite_batch[0][0][1]
        l_images = composite_batch[0][0][0]

        ul_targets = torch.cat([item[1] for item in composite_batch[1]])  # For later checks
        ul_ids = torch.cat([item[2] for item in composite_batch[1]])
        uw_images = torch.cat([item[0][0] for item in composite_batch[1]])
        us_images = torch.cat([item[0][1] for item in composite_batch[1]])

        # Supervised loss
        logits_l = self(l_images)
        loss_l = F.cross_entropy(logits_l, l_targets, reduction='mean')

        # Unsupervised loss
        logits_us = self(us_images)
        with torch.no_grad():
            logits_uw = self(uw_images)
            pseudo_label = torch.softmax(logits_uw.detach(), dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.hparams.model.threshold).float()
        loss_ul = (F.cross_entropy(logits_us, targets_u, reduction='none') * mask).mean()

        # Train loss / labelled accuracy
        loss = loss_l + self.hparams.model.lambda_u * loss_ul

        result = TrainResult(minimize=loss)
        result.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        result.log('train_loss_l', loss_l, on_epoch=True, on_step=False, sync_dist=True)
        result.log('train_loss_ul', loss_ul, on_epoch=True, on_step=False, sync_dist=True)

        # Unlabelled statistics
        if self.hparams.exp.log_ul_statistics == 'batch':
            certain_logits_uw = logits_uw[mask == 1.0].cpu().numpy()
            certain_ul_targets = ul_targets[mask == 1.0].cpu().numpy()
            all_logits_uw = logits_uw.cpu().numpy()
            all_ul_targets = ul_targets.cpu().numpy()

            certain_ul_acc = simple_accuracy(certain_logits_uw, certain_ul_targets)
            certain_ul_acc = torch.tensor(0.0) if np.isnan(certain_ul_acc) else torch.tensor(certain_ul_acc)

            all_ul_acc = simple_accuracy(all_logits_uw, all_ul_targets)
            all_ul_acc = torch.tensor(all_ul_acc)

            result.log('certain_ul_acc', certain_ul_acc, on_epoch=False, on_step=True, sync_dist=True)
            result.log('all_ul_acc', all_ul_acc, on_epoch=False, on_step=True, sync_dist=True)
            result.log('max_probs', max_probs.mean(), on_epoch=False, on_step=True, sync_dist=True)
            result.log('n_certain', mask.sum(), on_epoch=False, on_step=True, sync_dist=True)

        elif self.hparams.exp.log_ul_statistics == 'image':
            batch_df = pd.DataFrame(index=range(len(ul_ids)))
            batch_df['image_id'] = ul_ids.tolist()
            batch_df['score'] = max_probs.tolist()
            batch_df['correctness'] = (targets_u == ul_targets).tolist()
            # batch_df['experiment_id'] = self.run_id
            batch_df['epoch'] = self.trainer.current_epoch + 1
            # batch_df['score_type'] = 'softmax_output'
            self.logging_df = self.logging_df.append(batch_df, ignore_index=True)

        return result

    def validation_step(self, batch, batch_idx):
        results = {}
        images, targets, ids = batch
        logits = self(images, model='ema')
        loss = F.cross_entropy(logits, targets, reduction='mean')
        results['loss'] = loss.detach()
        results['logits'] = logits.detach()
        results['targets'] = targets.detach()
        return results

    def test_step(self, batch, batch_idx):
        results = {}
        images, targets, ids = batch
        logits = self(images, model='best')
        loss = F.cross_entropy(logits, targets, reduction='mean')
        results['loss'] = loss.detach()
        results['logits'] = logits.detach()
        results['targets'] = targets.detach()
        return results

    def validation_epoch_end(self, outputs):
        loss_val_mean = torch.tensor(np.array([x['loss'].cpu().numpy() for x in outputs]).mean())
        mlflow_metrics = {'val_loss': loss_val_mean, **self.calculate_metrics(outputs, prefix='val')}
        return {'loss': loss_val_mean, 'log': mlflow_metrics}

    def test_epoch_end(self, outputs):
        loss_test_mean = torch.tensor(np.array([x['loss'].cpu().numpy() for x in outputs]).mean())
        mlflow_metrics = {'test_loss': loss_test_mean, **self.calculate_metrics(outputs, prefix='test')}
        return {'loss': loss_test_mean, 'log': mlflow_metrics}

    @staticmethod
    def calculate_metrics(outputs, prefix):
        logits = np.array([prob for x in outputs for prob in x['logits'].cpu().numpy()])
        targets = np.array([label for x in outputs for label in x['targets'].cpu().numpy()])
        return {
            prefix + '_acc': simple_accuracy(logits, targets)
        }

    def on_epoch_end(self) -> None:
        if self.hparams.exp.log_ul_statistics == 'image' and (self.trainer.current_epoch + 1) % 100 == 0:
            epochs_range = self.logging_df.epoch.min(), self.logging_df.epoch.max()
            csv_path = f'{self.artifacts_path}/epochs_{epochs_range[0]}_{epochs_range[1]}.csv'
            self.logging_df.to_csv(csv_path, index=False)
            self.logging_df = pd.DataFrame()

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
