# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os

import hydra
import numpy as np
import torch
import torch.utils.data as data
from hydra.core.hydra_config import HydraConfig
from lightning.fabric import seed_everything
from omegaconf import DictConfig, OmegaConf

import wandb

from tactile_ssl.trainer import Trainer  # noqa: E402
from tactile_ssl.utils import get_local_rank
from tactile_ssl.utils.logging import get_pylogger, print_config_tree  # noqa: E402

logger = get_pylogger(__name__)

OmegaConf.register_new_resolver("int_multiply", lambda a, b: int(a * b))


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


def get_dataloaders_magnetic_based(cfg: DictConfig):
    data_cfg = cfg.data

    if data_cfg.sensor == "tdex":
        dataset = hydra.utils.instantiate(data_cfg.dataset)
        train_dset_size = int(len(dataset) * cfg.data.train_val_split)

        train_dset, val_dset = data.random_split(
            dataset, [train_dset_size, len(dataset) - train_dset_size]
        )
    elif data_cfg.sensor == "reskin":
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


def get_dataloaders_vision_based(cfg: DictConfig):
    data_cfg = cfg.data
    n_sensors = len(cfg.data.sensor)
    train_dset = []
    val_dset = []

    for i in range(n_sensors):
        sensor_cfg = data_cfg.sensor

        if sensor_cfg.type == "digit" or sensor_cfg.type == "gelsight_mini":
            list_datasets = sensor_cfg.dataset.config.list_datasets
            train_dset_ids = sensor_cfg.dataset.config.dataset_ids_train
            val_dset_ids = sensor_cfg.dataset.config.dataset_ids_val

            for obj in list_datasets:
                for d_id in train_dset_ids:
                    dataset_name = obj + "/dataset_" + str(d_id)
                    dataset = hydra.utils.instantiate(
                        sensor_cfg.dataset,
                        sensor=sensor_cfg.type,
                        dataset_name=dataset_name,
                    )
                    train_dset.append(dataset)
                for d_id in val_dset_ids:
                    dataset_name = obj + "/dataset_" + str(d_id)
                    dataset = hydra.utils.instantiate(
                        sensor_cfg.dataset,
                        sensor=sensor_cfg.type,
                        dataset_name=dataset_name,
                    )
                    val_dset.append(dataset)

        elif sensor_cfg.type == "gelsight":
            list_datasets = sensor_cfg.dataset.config.list_datasets
            path_dataset = sensor_cfg.dataset.config.path_dataset
            all_datasets = []
            for obj in list_datasets:
                files_list = os.listdir(os.path.join(path_dataset, obj))
                for file in files_list:
                    dataset_name = obj + "/" + file.split(".")[0]
                    dataset = hydra.utils.instantiate(
                        sensor_cfg.dataset,
                        sensor=sensor_cfg.type,
                        dataset_name=dataset_name,
                    )
                    all_datasets.append(dataset)

            all_datasets = sorted(all_datasets, key=lambda x: len(x), reverse=True)
            train_dset_size = int(
                len(all_datasets) * sensor_cfg.dataset.config.train_val_split
            )
            train_dset = train_dset + all_datasets[:train_dset_size]
            val_dset = val_dset + all_datasets[train_dset_size:]

    if isinstance(train_dset, list):
        train_dset = data.ConcatDataset(train_dset)
        val_dset = data.ConcatDataset(val_dset)

    return train_dset, val_dset


def get_dataloaders(cfg: DictConfig):
    train_dset, val_dset = get_dataloaders_vision_based(cfg)

    train_dataloader = data.DataLoader(train_dset, **cfg.data.train_dataloader)
    val_dataloader = data.DataLoader(val_dset, **cfg.data.val_dataloader)

    return train_dataloader, val_dataloader


def attempt_resume(cfg: DictConfig):
    ckpt_path = None
    if os.environ.get('SLURM_RESTART_COUNT') is not None: 
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        requeue_count = os.environ.get('SLURM_RESTART_COUNT') 
        logger.info(f"requeue count for job {slurm_job_id}: {requeue_count}")
        cfg.resume_id = slurm_job_id
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

def save_embeddings(embeddings, filenames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, "embeddings.npz"), 
             embeddings=embeddings,
             filenames=filenames)

def train(cfg: DictConfig):
    resume_state, cfg = attempt_resume(cfg)
    logger.info(f"Resume state: {resume_state}, {cfg.ckpt_path}")
    logger.info("Instantiating wandb ...")
    wandb = init_wandb(cfg.wandb)
    if ~resume_state:
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        OmegaConf.save(cfg, f"{cfg.paths.output_dir}/config.yaml")

    print_config_tree(cfg, resolve=True, save_to_file=True)
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    n_sensors = len(cfg.data.sensor) if OmegaConf.is_list(cfg.data.sensor) else 1
    sensors_type = (
        [cfg.data.sensor[i].type for i in range(n_sensors)]
        if n_sensors > 1
        else cfg.data.sensor
    )
    logger.info(f"Instantiating dataset & dataloaders for <{sensors_type}>")
    train_dataloader, val_dataloader = get_dataloaders(cfg)

    trainer = Trainer(wandb_logger=wandb, **cfg.trainer)

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    
    # Load checkpoint ============== ADD THIS ==============
    if cfg.ckpt_path:
        logger.info(f"Loading weights from {cfg.ckpt_path}")
        checkpoint = torch.load(cfg.ckpt_path)
        model.load_state_dict(checkpoint["model"])
    
    # Run inference ============== MODIFIED SECTION ==============
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_embeddings = []
    all_filenames = []
    
    with torch.no_grad():
        for batch in val_dataloader:
            # Assuming batch contains images and filenames
            images = batch["image"].to(device)
            filenames = batch["filename"]  # Modify based on your dataset
            
            # Get embeddings - modify based on your model architecture
            embeddings = model.encoder(images)  # Or model.get_embeddings(images)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_filenames.extend(filenames)
    
    # Concatenate and save
    all_embeddings = np.concatenate(all_embeddings)
    save_embeddings(all_embeddings, all_filenames, cfg.paths.output_dir)

    # trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.ckpt_path)

    wandb.finish()


@hydra.main(version_base="1.3", config_path="config", config_name="default.yaml")
def main(cfg: DictConfig):
    """
    Main function to train the model
    """
    train(cfg)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
