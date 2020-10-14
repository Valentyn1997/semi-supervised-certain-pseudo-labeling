import importlib

import torchvision.utils as vutils
from pytorch_lightning import LightningModule
from omegaconf import DictConfig
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
import numpy as np
from baal.bayesian import MCDropoutConnectModule
from baal.bayesian.weight_drop import patch_module
from pytorch_lightning.core.step_result import TrainResult

from src.models.certainty_strategy import AbstractStrategy
from src.utils import simple_accuracy
from src.models.utils import WeightEMA, UnlabelledStatisticsLogger
from src.models.backbones import WideResNet
from src.data.ssl_datasets import SLLDatasetsCollection, FixMatchCompositeTrainDataset

#
# def new_to(self, device):
#     self.parent_module.to(device)


class FixMatch(LightningModule):
    def __init__(self, args: DictConfig, datasets_collection: SLLDatasetsCollection, artifacts_path: str = None,
                 run_id: str = None):
        super().__init__()
        self.lr = None  # Placeholder for auto_lr_find
        self.artifacts_path = artifacts_path
        self.run_id = run_id
        self.datasets_collection = datasets_collection
        self.hparams = args  # Will be logged to mlflow

        if self.hparams.model.drop_type == 'Dropout':
            self.model = WideResNet(depth=28, widen_factor=2, drop_rate=self.hparams.model.drop_rate,
                                    num_classes=len(datasets_collection.classes))
        elif self.hparams.model.drop_type == 'DropConnect':
            self.model = WideResNet(depth=28, widen_factor=2, drop_rate=0.0, num_classes=len(datasets_collection.classes))
            self.model = patch_module(self.model, layers=['Conv2d'], weight_dropout=self.hparams.model.drop_rate, inplace=False)
            # self.model.to = new_to
        else:
            raise NotImplementedError

        self.ema_model = WideResNet(depth=28, widen_factor=2, drop_rate=0.0, num_classes=len(datasets_collection.classes))
        self.best_model = self.model  # Placeholder for checkpointing
        self.ema_optimizer = WeightEMA(self.model, self.ema_model,
                                       alpha=self.hparams.model.ema_decay,
                                       lr=self.hparams.optimizer.lr,
                                       weight_decay=self.hparams.optimizer.weight_decay
                                       if self.hparams.optimizer.weight_decay_time == 'after' else None)
        self.ul_logger = UnlabelledStatisticsLogger(level=self.hparams.exp.log_ul_statistics, save_frequency=500,
                                                    artifacts_path=self.artifacts_path)

        # Initialisation of certainty strategy
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
                   nesterov=self.hparams.optimizer.nesterov,
                   weight_decay=self.hparams.optimizer.weight_decay
                   if self.hparams.optimizer.weight_decay_time == 'before' else 0)

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

        u_targets = torch.cat([item[1] for item in composite_batch[1]])  # For later checks
        u_ids = torch.cat([item[2] for item in composite_batch[1]])
        uw_images = torch.cat([item[0][0] for item in composite_batch[1]])
        us_images = torch.cat([item[0][1] for item in composite_batch[1]])

        # Supervised loss
        l_logits = self(l_images)
        l_loss = F.cross_entropy(l_logits, l_targets, reduction='mean')

        # Unsupervised loss
        us_logits = self(us_images)
        with torch.no_grad():
            if self.strategy.is_ensemble:
                uw_logits = torch.stack([self(uw_images) for _ in range(self.hparams.model.T)]).detach()
                assert not (uw_logits[0] == uw_logits[1]).all()  # Checking if dropout actually works
            else:
                uw_logits = self(uw_images).detach()

        # get the value of the max, and the index (between 1 and 10 for CIFAR10 for example)
        uw_pseudo_soft_targets = torch.softmax(uw_logits, dim=-1)
        u_scores, u_pseudo_targets = self.strategy.get_certainty_and_label(softmax_outputs=uw_pseudo_soft_targets)
        mask = u_scores.ge(self.hparams.model.threshold).float()
        u_loss = (F.cross_entropy(us_logits, u_pseudo_targets, reduction='none') * mask).mean()

        # Train loss / labelled accuracy
        loss = l_loss + self.hparams.model.lambda_u * u_loss

        result = TrainResult(minimize=loss)
        result.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        result.log('train_loss_l', l_loss, on_epoch=True, on_step=False, sync_dist=True)
        result.log('train_loss_ul', u_loss, on_epoch=True, on_step=False, sync_dist=True)

        # Unlabelled statistics
        self.ul_logger.log_statistics(result, mask, u_scores, u_targets, u_pseudo_targets, u_ids,
                                      current_epoch=self.trainer.current_epoch + 1)

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
        self.ul_logger.on_epoch_end(current_epoch=self.trainer.current_epoch + 1)

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
