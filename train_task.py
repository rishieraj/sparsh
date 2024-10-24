# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os
import hydra
import numpy as np
import torch
import torch.utils.data as data
from omegaconf import OmegaConf, DictConfig
from hydra.core.hydra_config import HydraConfig

import wandb
from lightning.fabric import seed_everything

from tactile_ssl.utils import get_local_rank

from tactile_ssl.utils.logging import get_pylogger, print_config_tree
from tactile_ssl.trainer import Trainer

logger = get_pylogger(__name__)


def init_wandb(cfg: DictConfig):
    wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        dir=cfg.save_dir,
        id=f"{cfg.id}_{get_local_rank()}",
        group=cfg.group,
        tags=cfg.tags,
        notes=cfg.notes,
    )
    return wandb


def get_dataloader_reskin(cfg: DictConfig):
    data_cfg = cfg.data

    train_dataset_list = data_cfg.train_dataset_list
    val_dataset_list = data_cfg.val_dataset_list
    train_datasets, val_datasets = [], []
    for dataset_name in train_dataset_list:
        data_path = os.path.join(data_cfg.dataset.data_path, dataset_name)
        train_datasets.append(
            hydra.utils.instantiate(data_cfg.dataset, data_path=data_path)
        )
    for dataset_name in val_dataset_list:
        data_path = os.path.join(data_cfg.dataset.data_path, dataset_name)
        val_datasets.append(
            hydra.utils.instantiate(data_cfg.dataset, data_path=data_path)
        )
    train_dset = data.ConcatDataset(train_datasets)
    val_dset = data.ConcatDataset(val_datasets)
    return train_dset, val_dset


def get_dataloader_digit(cfg: DictConfig):
    data_cfg = cfg.data
    datasets = []
    list_datasets = data_cfg.dataset.config.list_datasets
    path_dataset = data_cfg.dataset.config.path_dataset

    
    for d in list_datasets:
        if data_cfg.dataset.config.look_in_folder:
            obj = d
            logger.info(f"Loading train dataset: {obj}")
            files_list = os.listdir(os.path.join(path_dataset, obj))
            
            for f in files_list:
                dataset_name = obj + "/" + f.split(".")[0]
                try:
                    datasets.append(
                        hydra.utils.instantiate(
                            data_cfg.dataset, dataset_name=dataset_name
                        )
                    )
                except Exception as e:
                    logger.error(f"Error loading {dataset_name}: {e}")

        else:
            datasets.append(hydra.utils.instantiate(data_cfg.dataset, dataset_name=d))
        
    train_dset_samples = sum([len(d) for d in datasets])

    # if dataset is larger than maximum budget of training data, sample uniformly
    data_cfg.max_train_data = train_dset_samples if data_cfg.max_train_data == None else data_cfg.max_train_data
    if train_dset_samples > data_cfg.max_train_data:
        try:
            idxs = [np.random.choice(len(d), data_cfg.max_train_data//len(datasets), replace=False) for d in datasets]
            datasets = [data.Subset(d, idxs[i]) for i, d in enumerate(datasets)]
            dataset = data.ConcatDataset(datasets)
        except:
            dataset = data.ConcatDataset(datasets)
            dataset, _ = data.random_split(
                dataset, [data_cfg.max_train_data, len(dataset) - data_cfg.max_train_data]
            )
    else:
        dataset = data.ConcatDataset(datasets)
    logger.info(f"Loaded dataset size: {len(dataset)}")

    dataset_test = None
    list_datasets_test = data_cfg.dataset.config.list_datasets_test
    if len(list_datasets_test) > 0:
        dataset_test = []
        for d in list_datasets_test:
            if data_cfg.dataset.config.look_in_folder:
                obj = d
                logger.info(f"Loading test dataset: {obj}")
                files_list = os.listdir(os.path.join(path_dataset, obj))
            
                for f in files_list:
                    dataset_name = obj + "/" + f.split(".")[0]
                    try:
                        dataset_test.append(
                            hydra.utils.instantiate(
                                data_cfg.dataset, dataset_name=dataset_name
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error loading {dataset_name}: {e}")
            else:
                dataset_test.append(
                    hydra.utils.instantiate(data_cfg.dataset, dataset_name=d)
                )
        dataset_test = data.ConcatDataset(dataset_test)

    if dataset_test is None:
        train_dset_size = int(len(dataset) * data_cfg.train_val_split)
        train_dset, val_dset = data.random_split(
            dataset, [train_dset_size, len(dataset) - train_dset_size]
        )
    else:
        train_dset = dataset
        val_dset = dataset_test


    # adjust training dataset size
    train_dset_size = int(len(train_dset) * data_cfg.train_data_budget)
    train_dset, _ = data.random_split(
        train_dset, [train_dset_size, len(train_dset) - train_dset_size]
    )

    try:
        val_dset_size = int(len(val_dset) * data_cfg.val_data_budget)
        val_dset, _ = data.random_split(
            val_dset, [val_dset_size, len(val_dset) - val_dset_size]
        )
    except:
        pass

    logger.info("Train/Val data split")
    logger.info(f"Train dataset size: {len(train_dset)}")
    logger.info(f"Val dataset size: {len(val_dset)}")

    return train_dset, val_dset


def get_dataloaders(cfg: DictConfig):
    data_cfg = cfg.data

    if data_cfg.sensor == "digit" or "gelsight" in data_cfg.sensor :
        train_dset, val_dset = get_dataloader_digit(cfg)
    elif data_cfg.sensor == "reskin":
        train_dset, val_dset = get_dataloader_reskin(cfg)
    else:
        raise NotImplementedError("Sensor type not implemented yet.")

    train_dataloader = data.DataLoader(train_dset, **cfg.data.train_dataloader)
    val_dataloader = data.DataLoader(val_dset, **cfg.data.val_dataloader)

    return train_dataloader, val_dataloader


def attempt_resume(cfg: DictConfig):
    ckpt_path = None
    if os.path.exists(f"{cfg.paths.output_dir}/config.yaml") and cfg.resume_id:
        job_id = HydraConfig.get().job.id
        logger.info(f"Attempting to resume experiment with {cfg.resume_id}")
        if not os.path.exists(f"{cfg.paths.output_dir}/checkpoints/"):
            logger.warning(
                f"Unable to resume: No checkpoints found for experiment with id {job_id}"
            )
            return False, cfg
        if not os.path.exists(f"{cfg.paths.output_dir}/wandb/"):
            logger.warning(
                f"Unable to resume: No wandb logs found for experiment with id {job_id}"
            )
            return False, cfg
        if not os.path.exists(f"{cfg.paths.output_dir}/config.yaml"):
            logger.warning(
                "Could not find a config.yaml file in the resume directory. Using the current config."
            )
            return False, cfg

        cfg = OmegaConf.load(f"{cfg.paths.output_dir}/config.yaml")

        ckpt_path = f"{cfg.paths.output_dir}/checkpoints/"
        OmegaConf.update(cfg, "ckpt_path", ckpt_path, force_add=True)
        experiment_name = cfg.experiment_name
        cfg.wandb.id = f"{job_id}_{experiment_name}"
        logger.info(
            f"Resuming experiment {job_id} with wandb_id: {cfg.wandb.id} from latest checkpoint at {cfg.ckpt_path}"
        )
        return True, cfg
    return False, cfg


def train(cfg: DictConfig):
    resume_state, cfg = attempt_resume(cfg)

    logger.info("Instantiating wandb ...")
    wandb = init_wandb(cfg.wandb)
    if not resume_state:
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.save(cfg, f"{cfg.paths.output_dir}/config.yaml")

    print_config_tree(cfg, resolve=True, save_to_file=True)
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
    _GLOBAL_SEED = cfg.seed
    np.random.seed(_GLOBAL_SEED)
    torch.manual_seed(_GLOBAL_SEED)
    torch.backends.cudnn.benchmark = True

    logger.info(
        f"Instantiating dataset & dataloaders for <{cfg.data.dataset._target_}>"
    )
    train_dataloader, val_dataloader = get_dataloaders(cfg)

    trainer = Trainer(wandb_logger=wandb, **cfg.trainer)

    logger.info(f"Instantiating model <{cfg.task._target_}>")
    model = hydra.utils.instantiate(cfg.task)

    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.ckpt_path)

    wandb.finish()


@hydra.main(version_base="1.3", config_path="config")
def main(cfg: DictConfig):
    """
    Main function to train the model
    """
    train(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
