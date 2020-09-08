import logging
import hydra
import torch
from omegaconf import DictConfig

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src import MLFLOW_URI, CONFIG_PATH, ROOT_PATH
from src.utils import set_seed, CustomModelCheckpoint
from src.models.fix_match import FixMatch
from src.data.ssl_datasets import SLLDatasetsCollection
from src.data.augmentations import BasicTransformation, WeakAugment, StrongAugment, WeakStrongAugment

logger = logging.getLogger(__name__)


@hydra.main(config_name=f'{CONFIG_PATH}/config.yaml', strict=False)
def main(args: DictConfig):

    # Secondary data args
    dataset_name = args.data.source

    # MlFlow Logging
    if args.exp.logging:
        experiment_name = f'FixMatch/{dataset_name}'
        mlf_logger = MLFlowLogger(experiment_name=experiment_name, tracking_uri=MLFLOW_URI)
        run_id = mlf_logger.run_id
        experiment_id = mlf_logger.experiment.get_experiment_by_name(experiment_name).experiment_id
        cpnt_path = f'{ROOT_PATH}/mlruns/{experiment_id}/{run_id}/artifacts'
    else:
        cpnt_path = None

    # Loading dataset, augmentations and model
    set_seed(args)
    basic_transform = BasicTransformation(source=args.data.source)
    train_l_transform = WeakAugment(basic_transform=basic_transform, flip=args.data.weak_aug.flip,
                                    random_resize_crop=args.data.weak_aug.random_resize_crop)
    train_ul_transform = WeakStrongAugment(weak_augment=train_l_transform,
                                           strong_augment=StrongAugment(basic_transform))
    datasets_collection = SLLDatasetsCollection(source=args.data.source,
                                                n_labelled=args.data.n_labelled,
                                                val_ratio=args.data.val_ratio,
                                                random_state=args.exp.seed,
                                                train_l_transform=train_l_transform,
                                                train_ul_transform=train_ul_transform,
                                                test_transform=basic_transform)
    model = FixMatch(args, datasets_collection)

    # Early stopping & Checkpointing
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=args.exp.early_stopping_patience,
                                        verbose=False, mode='min')
    checkpoint_callback = CustomModelCheckpoint(model=model, verbose=True, monitor='val_loss', mode='min', save_top_k=1)

    logger.info(f'Run arguments: \n{args.pretty()}')

    # Training
    trainer = Trainer(gpus=eval(str(args.exp.gpus)),
                      logger=mlf_logger if args.exp.logging else None,
                      max_epochs=args.exp.max_epochs,
                      early_stop_callback=early_stop_callback,
                      checkpoint_callback=checkpoint_callback if args.exp.checkpoint else None,
                      auto_lr_find=args.optimizer.auto_lr_find)
    trainer.fit(model)

    # Testing doesn't work for dp mode
    model.model = model.best_model
    trainer.run_evaluation(test_mode=True)

    # Ending the run
    if args.exp.logging:
        mlf_logger.finalize()


if __name__ == "__main__":
    main()
