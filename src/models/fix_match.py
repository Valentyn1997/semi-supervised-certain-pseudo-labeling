import importlib

import torchvision.utils as vutils
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy

from src.models.certainty_strategy import AbstractStrategy
from src.utils import simple_accuracy
from src.models.utils import WeightEMA
from src.models.backbones import WideResNet
from src.data.ssl_datasets import SLLDatasetsCollection, FixMatchCompositeTrainDataset


class FixMatch(LightningModule):
    def __init__(self, args: DictConfig, datasets_collection: SLLDatasetsCollection, artifacts_path: str = None):
        super().__init__()
        self.lr = None  # Placeholder for auto_lr_find
        self.artifacts_path = artifacts_path
        self.datasets_collection = datasets_collection
        self.model = WideResNet(depth=28, widen_factor=2, drop_rate=0.0, num_classes=len(datasets_collection.classes))
        self.ema_model = WideResNet(depth=28, widen_factor=2, drop_rate=0.0, num_classes=len(datasets_collection.classes))
        self.best_model = self.model  # Placeholder for checkpointing
        self.hparams = args  # Will be logged to mlflow
        self.ema_optimizer = WeightEMA(self.model, self.ema_model, alpha=self.hparams.model.ema_decay)

        strategy_class_name = self.hparams.model.certainty_strategy
        module = importlib.import_module("src.models.certainty_strategy")
        strategy = getattr(module, strategy_class_name)()
        assert isinstance(strategy, AbstractStrategy)
        self.strategy = strategy

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
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams.data.batch_size.train, num_workers=2)

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

        # ul_targets = torch.cat([item[1] for item in composite_batch[1]])  # For later checks
        uw_images = torch.cat([item[0][0] for item in composite_batch[1]])
        us_images = torch.cat([item[0][1] for item in composite_batch[1]])

        # Supervised loss
        logits_l = self(l_images)
        loss_l = F.cross_entropy(logits_l, l_targets, reduction='mean')

        # Unsupervised loss
        logits_us = self(us_images)
        with torch.no_grad():
            if self.strategy.is_ensemble():
                output = torch.stack([self(uw_images) for _ in range(self.hparams.model.T)]).detach()
            else:
                output = self(uw_images).detach()

        # get the value of the max, and the index (between 1 and 10 for CIFAR10 for example)
        logits = torch.softmax(output, dim=-1)
        max_probs, targets_u = self.strategy.get_certainty_and_label(logits_t_n_c=logits)

        mask = max_probs.ge(self.hparams.model.threshold).float()
        loss_ul = (F.cross_entropy(logits_us, targets_u, reduction='none') * mask).mean()

        # Train loss / labelled accuracy
        loss = loss_l + self.hparams.model.lambda_u * loss_ul
        return {'loss': loss, 'loss_l': loss_l, 'loss_ul': loss_ul}

    def validation_step(self, batch, batch_idx):
        results = {}
        images, targets = batch
        logits = self(images, model='ema')
        loss = F.cross_entropy(logits, targets, reduction='mean')
        results['loss'] = loss.detach()
        results['preds'] = logits.detach()
        results['labels'] = targets.detach()
        return results

    def test_step(self, batch, batch_idx):
        results = {}
        images, targets = batch
        logits = self(images, model='best')
        loss = F.cross_entropy(logits, targets, reduction='mean')
        results['loss'] = loss.detach()
        results['preds'] = logits.detach()
        results['labels'] = targets.detach()
        return results

    def training_epoch_end(self, outputs):
        loss = np.array([x['loss'].mean().item() for x in outputs]).mean()
        loss_l = np.array([x['loss_l'].mean().item() for x in outputs]).mean()
        loss_ul = np.array([x['loss_ul'].mean().item() for x in outputs]).mean()
        mlflow_metrics = {'train_loss': loss, 'train_loss_l': loss_l, 'train_loss_ul': loss_ul}
        return {'loss': loss, 'log': mlflow_metrics}

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
        preds = np.array([prob for x in outputs for prob in x['preds'].cpu().numpy()])
        preds = np.argmax(preds, axis=1)
        labels = np.array([label for x in outputs for label in x['labels'].cpu().numpy()])
        return {
            prefix + '_acc': simple_accuracy(preds, labels)
        }

    def on_train_start(self) -> None:
        if self.artifacts_path is not None:  # Plotting one batch
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
