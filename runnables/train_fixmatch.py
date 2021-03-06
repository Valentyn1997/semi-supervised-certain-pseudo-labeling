import logging
import hydra
import torch
from omegaconf import DictConfig

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_logger import LearningRateLogger

from src import CONFIG_PATH, GLOBAL_ARTIFACTS_PATH
from src.utils import set_seed, CustomModelCheckpoint
from src.models.fix_match import FixMatch, MultiStrategyFixMatch
from src.data.ssl_datasets import SLLDatasetsCollection
from src.data.augmentations import BasicTransformation, WeakAugment, StrongAugment, WeakStrongAugment

logger = logging.getLogger(__name__)


@hydra.main(config_name=f'{CONFIG_PATH}/config.yaml', strict=False)
def main(args: DictConfig):

    # Secondary data args
    dataset_name = args.data.source

    # MlFlow Logging
    if args.exp.logging:
        experiment_name = f'FixMatch/{dataset_name}/{args.model.certainty_strategy}'
        mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri)

        run_id = mlf_logger.run_id
        experiment_id = mlf_logger.experiment.get_experiment_by_name(experiment_name).experiment_id
        artifacts_path = f'{GLOBAL_ARTIFACTS_PATH}/mlruns/{experiment_id}/{run_id}/artifacts'
    else:
        artifacts_path = None

    # Loading dataset, augmentations and model
    set_seed(args)
    basic_transform = BasicTransformation(source=args.data.source)
    train_l_transform = WeakAugment(basic_transform=basic_transform, flip=args.data.weak_aug.flip,
                                    random_pad_and_crop=args.data.weak_aug.random_pad_and_crop,
                                    crop_size=args.data.weak_aug.crop_size)
    train_ul_transform = WeakStrongAugment(weak_augment=train_l_transform,
                                           strong_augment=StrongAugment(basic_transform, crop_size=args.data.weak_aug.crop_size))
    datasets_collection = SLLDatasetsCollection(source=args.data.source,
                                                n_labelled=args.data.n_labelled,
                                                val_ratio=args.data.val_ratio,
                                                random_state=args.exp.seed,
                                                train_l_transform=train_l_transform,
                                                train_ul_transform=train_ul_transform,
                                                test_transform=basic_transform)

    # Model init
    model_class = MultiStrategyFixMatch if args.model.multi_strategy else FixMatch
    model = model_class(args, datasets_collection, artifacts_path=artifacts_path)

    # Early stopping & Checkpointing & LR-logging
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=args.exp.early_stopping_patience,
                                        verbose=False, mode='min')
    checkpoint_callback = CustomModelCheckpoint(model=model, verbose=True, monitor='val_loss', mode='min', save_top_k=1)
    lr_logging_callback = LearningRateLogger(logging_interval='epoch')

    logger.info(f'Run arguments: \n{args.pretty()}')

    # Training
    trainer = Trainer(gpus=eval(str(args.exp.gpus)),
                      logger=mlf_logger if args.exp.logging else None,
                      max_epochs=args.exp.max_epochs,
                      early_stop_callback=early_stop_callback if args.exp.early_stopping else None,
                      checkpoint_callback=checkpoint_callback if args.exp.checkpoint else None,
                      auto_lr_find=args.optimizer.auto_lr_find,
                      distributed_backend='dp',
                      row_log_interval=1,
                      profiler=True,
                      precision=args.exp.precision,
                      check_val_every_n_epoch=args.exp.check_val_every_n_epoch,
                      callbacks=[lr_logging_callback])
    trainer.fit(model)
    trainer.test(model)

    # Ending the run
    if args.exp.logging:
        mlf_logger.finalize()

    trainer.reset_train_dataloader(model)
    trainer.reset_val_dataloader(model)
    trainer.reset_test_dataloader(model)


if __name__ == "__main__":
    main()
