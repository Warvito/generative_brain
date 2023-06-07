from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from custom_transforms import ApplyTokenizerd
from mlflow import start_run
from monai import transforms
from monai.data import PersistentDataset
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm


# ----------------------------------------------------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------------------------------------------------
def get_datalist(
    ids_path: str,
):
    """Get data dicts for data loaders."""
    df = pd.read_csv(ids_path, sep="\t")

    data_dicts = []
    for index, row in df.iterrows():
        data_dicts.append(
            {
                "image": str(row["image"]),
                "report": "T1-weighted image of a brain.",
            }
        )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


def get_dataloader(
    cache_dir: Union[str, Path],
    batch_size: int,
    training_ids: str,
    validation_ids: str,
    num_workers: int = 8,
    model_type: str = "autoencoder",
):
    # Define transformations
    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.SpatialCropd(keys=["image"], roi_start=[16, 16, 96], roi_end=[176, 240, 256]),
            transforms.SpatialPadd(
                keys=["image"],
                spatial_size=[160, 224, 160],
            ),
            ApplyTokenizerd(keys=["report"]),
            transforms.ToTensord(keys=["image", "report"]),
        ]
    )
    if model_type == "autoencoder":
        train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                transforms.SpatialCropd(keys=["image"], roi_start=[16, 16, 96], roi_end=[176, 240, 256]),
                transforms.SpatialPadd(
                    keys=["image"],
                    spatial_size=[160, 224, 160],
                ),
                transforms.RandFlipd(
                    keys=["image"],
                    spatial_axis=0,
                    prob=0.5,
                ),
                transforms.RandAffined(
                    keys=["image"],
                    translate_range=(1, 1, 1),
                    scale_range=(-0.02, 0.02),
                    spatial_size=[160, 224, 160],
                    prob=0.1,
                ),
                transforms.RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.1),
                transforms.RandAdjustContrastd(keys=["image"], gamma=(0.97, 1.03), prob=0.1),
                transforms.ThresholdIntensityd(keys=["image"], threshold=1, above=False, cval=1.0),
                transforms.ThresholdIntensityd(keys=["image"], threshold=0, above=True, cval=0),
            ]
        )
    if model_type == "diffusion":
        train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                transforms.SpatialCropd(keys=["image"], roi_start=[16, 16, 96], roi_end=[176, 240, 256]),
                transforms.SpatialPadd(
                    keys=["image"],
                    spatial_size=[160, 224, 160],
                ),
                transforms.RandFlipd(
                    keys=["image"],
                    spatial_axis=0,
                    prob=0.5,
                ),
                transforms.RandAffined(
                    keys=["image"],
                    translate_range=(1, 1, 1),
                    scale_range=(-0.02, 0.02),
                    spatial_size=[160, 224, 160],
                    prob=0.1,
                ),
                transforms.RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.1),
                transforms.RandAdjustContrastd(keys=["image"], gamma=(0.97, 1.03), prob=0.1),
                transforms.ThresholdIntensityd(keys=["image"], threshold=1, above=False, cval=1.0),
                transforms.ThresholdIntensityd(keys=["image"], threshold=0, above=True, cval=0),
                ApplyTokenizerd(keys=["report"]),
                transforms.RandLambdad(
                    keys=["report"],
                    prob=0.10,
                    func=lambda x: torch.cat(
                        (49406 * torch.ones(1, 1), 49407 * torch.ones(1, x.shape[1] - 1)), 1
                    ).long(),
                ),  # 49406: BOS token 49407: PAD token
                transforms.ToTensord(keys=["image", "report"]),
            ]
        )

    train_dicts = get_datalist(ids_path=training_ids)
    train_ds = PersistentDataset(data=train_dicts, transform=train_transforms, cache_dir=str(cache_dir))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    val_dicts = get_datalist(ids_path=validation_ids)
    val_ds = PersistentDataset(data=val_dicts, transform=val_transforms, cache_dir=str(cache_dir))
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    return train_loader, val_loader


def get_downsampled_dataloader(
    cache_dir: Union[str, Path],
    batch_size: int,
    training_ids: str,
    validation_ids: str,
    num_workers: int = 8,
    model_type: str = "autoencoder",
):
    # Define transformations
    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.SpatialCropd(keys=["image"], roi_start=[16, 16, 96], roi_end=[176, 240, 256]),
            transforms.SpatialPadd(
                keys=["image"],
                spatial_size=[160, 224, 160],
            ),
            transforms.Resized(
                keys=["image"],
                spatial_size=[80, 112, 80],
            ),
            ApplyTokenizerd(keys=["report"]),
            transforms.ToTensord(keys=["image", "report"]),
        ]
    )
    if model_type == "autoencoder":
        train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                transforms.SpatialCropd(keys=["image"], roi_start=[16, 16, 96], roi_end=[176, 240, 256]),
                transforms.SpatialPadd(
                    keys=["image"],
                    spatial_size=[160, 224, 160],
                ),
                transforms.RandFlipd(
                    keys=["image"],
                    spatial_axis=0,
                    prob=0.5,
                ),
                transforms.RandAffined(
                    keys=["image"],
                    translate_range=(1, 1, 1),
                    scale_range=(-0.02, 0.02),
                    spatial_size=[160, 224, 160],
                    prob=0.1,
                ),
                transforms.RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.1),
                transforms.RandAdjustContrastd(keys=["image"], gamma=(0.97, 1.03), prob=0.1),
                transforms.ThresholdIntensityd(keys=["image"], threshold=1, above=False, cval=1.0),
                transforms.ThresholdIntensityd(keys=["image"], threshold=0, above=True, cval=0),
                transforms.Resized(
                    keys=["image"],
                    spatial_size=[80, 112, 80],
                ),
            ]
        )
    if model_type == "diffusion":
        train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                transforms.SpatialCropd(keys=["image"], roi_start=[16, 16, 96], roi_end=[176, 240, 256]),
                transforms.SpatialPadd(
                    keys=["image"],
                    spatial_size=[160, 224, 160],
                ),
                transforms.RandFlipd(
                    keys=["image"],
                    spatial_axis=0,
                    prob=0.5,
                ),
                transforms.RandAffined(
                    keys=["image"],
                    translate_range=(1, 1, 1),
                    scale_range=(-0.02, 0.02),
                    spatial_size=[160, 224, 160],
                    prob=0.1,
                ),
                transforms.RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.1),
                transforms.RandAdjustContrastd(keys=["image"], gamma=(0.97, 1.03), prob=0.1),
                transforms.ThresholdIntensityd(keys=["image"], threshold=1, above=False, cval=1.0),
                transforms.ThresholdIntensityd(keys=["image"], threshold=0, above=True, cval=0),
                transforms.Resized(
                    keys=["image"],
                    spatial_size=[80, 112, 80],
                ),
                ApplyTokenizerd(keys=["report"]),
                transforms.RandLambdad(
                    keys=["report"],
                    prob=0.10,
                    func=lambda x: torch.cat(
                        (49406 * torch.ones(1, 1), 49407 * torch.ones(1, x.shape[1] - 1)), 1
                    ).long(),
                ),  # 49406: BOS token 49407: PAD token
                transforms.ToTensord(keys=["image", "report"]),
            ]
        )

    train_dicts = get_datalist(ids_path=training_ids)
    train_ds = PersistentDataset(data=train_dicts, transform=train_transforms, cache_dir=str(cache_dir))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    val_dicts = get_datalist(ids_path=validation_ids)
    val_ds = PersistentDataset(data=val_dicts, transform=val_transforms, cache_dir=str(cache_dir))
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    return train_loader, val_loader


def get_upsampler_dataloader(
    cache_dir: Union[str, Path],
    batch_size: int,
    training_ids: str,
    validation_ids: str,
    num_workers: int = 8,
):
    # Define transformations
    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.SpatialCropd(keys=["image"], roi_start=[16, 16, 96], roi_end=[176, 240, 256]),
            transforms.SpatialPadd(
                keys=["image"],
                spatial_size=[160, 224, 160],
            ),
            transforms.RandSpatialCropd(keys=["image"], roi_size=[80, 112, 80], random_size=False),
            transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
            transforms.Resized(
                keys=["low_res_image"],
                spatial_size=[40, 56, 40],
            ),
            ApplyTokenizerd(keys=["report"]),
            transforms.ToTensord(keys=["image", "low_res_image", "report"]),
        ]
    )
    train_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.SpatialCropd(keys=["image"], roi_start=[16, 16, 96], roi_end=[176, 240, 256]),
            transforms.SpatialPadd(
                keys=["image"],
                spatial_size=[160, 224, 160],
            ),
            transforms.RandSpatialCropd(keys=["image"], roi_size=[80, 112, 80], random_size=False),
            transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
            transforms.Resized(
                keys=["low_res_image"],
                spatial_size=[40, 56, 40],
            ),
            ApplyTokenizerd(keys=["report"]),
            transforms.ToTensord(keys=["image", "low_res_image", "report"]),
        ]
    )

    train_dicts = get_datalist(ids_path=training_ids)
    train_ds = PersistentDataset(data=train_dicts, transform=train_transforms, cache_dir=str(cache_dir))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    val_dicts = get_datalist(ids_path=validation_ids)
    val_ds = PersistentDataset(data=val_dicts, transform=val_transforms, cache_dir=str(cache_dir))
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    return train_loader, val_loader


def get_datalist_anycontrast(
    ids_path: str,
):
    """Get data dicts for data loaders."""
    df = pd.read_csv(ids_path, sep="\t")

    data_dicts = []
    for index, row in df.iterrows():
        data_dicts.append(
            {
                "image": str(row["image"]),
                "report": "T1-weighted image of a brain.",
            }
        )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts


def get_dataloader_anycontrast(
    cache_dir: Union[str, Path],
    batch_size: int,
    training_ids: str,
    validation_ids: str,
    num_workers: int = 8,
    model_type: str = "autoencoder",
):
    # Define transformations
    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.SpatialCropd(keys=["image"], roi_start=[16, 16, 96], roi_end=[176, 240, 256]),
            transforms.SpatialPadd(
                keys=["image"],
                spatial_size=[160, 224, 160],
            ),
            ApplyTokenizerd(keys=["report"]),
            transforms.ToTensord(keys=["image", "report"]),
        ]
    )
    if model_type == "autoencoder":
        train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                transforms.SpatialCropd(keys=["image"], roi_start=[16, 16, 96], roi_end=[176, 240, 256]),
                transforms.SpatialPadd(
                    keys=["image"],
                    spatial_size=[160, 224, 160],
                ),
                transforms.RandFlipd(
                    keys=["image"],
                    spatial_axis=0,
                    prob=0.5,
                ),
                transforms.RandAffined(
                    keys=["image"],
                    translate_range=(5, 5, 5),
                    rotate_range=np.pi / 18,
                    shear_range=0.05,
                    scale_range=(-0.10, 0.10),
                    spatial_size=[160, 224, 160],
                    prob=0.25,
                ),
                transforms.RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.25),
                transforms.RandAdjustContrastd(keys=["image"], gamma=(0.5, 2), prob=0.25),
                transforms.ThresholdIntensityd(keys=["image"], threshold=1, above=False, cval=1.0),
                transforms.ThresholdIntensityd(keys=["image"], threshold=0, above=True, cval=0),
            ]
        )
    if model_type == "diffusion":
        train_transforms = transforms.Compose(
            [
                transforms.LoadImaged(keys=["image"]),
                transforms.EnsureChannelFirstd(keys=["image"]),
                transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
                transforms.SpatialCropd(keys=["image"], roi_start=[16, 16, 96], roi_end=[176, 240, 256]),
                transforms.SpatialPadd(
                    keys=["image"],
                    spatial_size=[160, 224, 160],
                ),
                transforms.RandFlipd(
                    keys=["image"],
                    spatial_axis=0,
                    prob=0.5,
                ),
                transforms.RandAffined(
                    keys=["image"],
                    translate_range=(1, 1, 1),
                    scale_range=(-0.02, 0.02),
                    spatial_size=[160, 224, 160],
                    prob=0.1,
                ),
                transforms.RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.1),
                transforms.RandAdjustContrastd(keys=["image"], gamma=(0.97, 1.03), prob=0.1),
                transforms.ThresholdIntensityd(keys=["image"], threshold=1, above=False, cval=1.0),
                transforms.ThresholdIntensityd(keys=["image"], threshold=0, above=True, cval=0),
                ApplyTokenizerd(keys=["report"]),
                transforms.RandLambdad(
                    keys=["report"],
                    prob=0.10,
                    func=lambda x: torch.cat(
                        (49406 * torch.ones(1, 1), 49407 * torch.ones(1, x.shape[1] - 1)), 1
                    ).long(),
                ),  # 49406: BOS token 49407: PAD token
                transforms.ToTensord(keys=["image", "report"]),
            ]
        )

    train_dicts = get_datalist_anycontrast(ids_path=training_ids)
    train_ds = PersistentDataset(data=train_dicts, transform=train_transforms, cache_dir=str(cache_dir))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    val_dicts = get_datalist_anycontrast(ids_path=validation_ids)
    val_ds = PersistentDataset(data=val_dicts, transform=val_transforms, cache_dir=str(cache_dir))
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    return train_loader, val_loader


# ----------------------------------------------------------------------------------------------------------------------
# LOGS
# ----------------------------------------------------------------------------------------------------------------------
def recursive_items(dictionary, prefix=""):
    for key, value in dictionary.items():
        if type(value) in [dict, DictConfig]:
            yield from recursive_items(value, prefix=str(key) if prefix == "" else f"{prefix}.{str(key)}")
        else:
            yield (str(key) if prefix == "" else f"{prefix}.{str(key)}", value)


def log_mlflow(
    model,
    config,
    args,
    experiment: str,
    run_dir: Path,
    val_loss: float,
):
    """Log model and performance on Mlflow system"""
    config = {**OmegaConf.to_container(config), **vars(args)}
    print(f"Setting mlflow experiment: {experiment}")
    mlflow.set_experiment(experiment)

    with start_run():
        print(f"MLFLOW URI: {mlflow.tracking.get_tracking_uri()}")
        print(f"MLFLOW ARTIFACT URI: {mlflow.get_artifact_uri()}")

        for key, value in recursive_items(config):
            mlflow.log_param(key, str(value))

        mlflow.log_artifacts(str(run_dir / "train"), artifact_path="events_train")
        mlflow.log_artifacts(str(run_dir / "val"), artifact_path="events_val")
        mlflow.log_metric(f"loss", val_loss, 0)

        raw_model = model.module if hasattr(model, "module") else model
        mlflow.pytorch.log_model(raw_model, "final_model")


def get_figure(
    img: torch.Tensor,
    recons: torch.Tensor,
):
    img_npy_0 = np.clip(a=img[0, 0, :, :, 60].cpu().numpy(), a_min=0, a_max=1)
    recons_npy_0 = np.clip(a=recons[0, 0, :, :, 60].cpu().numpy(), a_min=0, a_max=1)
    img_npy_1 = np.clip(a=img[0, 0, :, :, 30].cpu().numpy(), a_min=0, a_max=1)
    recons_npy_1 = np.clip(a=recons[0, 0, :, :, 30].cpu().numpy(), a_min=0, a_max=1)
    img_npy_2 = np.clip(a=img[0, 0, :, :, 50].cpu().numpy(), a_min=0, a_max=1)
    recons_npy_2 = np.clip(a=recons[0, 0, :, :, 50].cpu().numpy(), a_min=0, a_max=1)
    img_npy_3 = np.clip(a=img[0, 0, :, :, 40].cpu().numpy(), a_min=0, a_max=1)
    recons_npy_3 = np.clip(a=recons[0, 0, :, :, 40].cpu().numpy(), a_min=0, a_max=1)

    img_row_0 = np.concatenate(
        (
            img_npy_0,
            recons_npy_0,
            img_npy_1,
            recons_npy_1,
        ),
        axis=1,
    )

    img_row_1 = np.concatenate(
        (
            img_npy_2,
            recons_npy_2,
            img_npy_3,
            recons_npy_3,
        ),
        axis=1,
    )

    img = np.concatenate(
        (
            img_row_0,
            img_row_1,
        ),
        axis=0,
    )

    fig = plt.figure(dpi=300)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    return fig


def log_reconstructions(
    image: torch.Tensor,
    reconstruction: torch.Tensor,
    writer: SummaryWriter,
    step: int,
    title: str = "RECONSTRUCTION",
) -> None:
    fig = get_figure(
        image,
        reconstruction,
    )
    writer.add_figure(title, fig, step)


@torch.no_grad()
def log_ldm_sample_unconditioned(
    model: nn.Module,
    stage1: nn.Module,
    text_encoder,
    scheduler: nn.Module,
    spatial_shape: Tuple,
    writer: SummaryWriter,
    step: int,
    device: torch.device,
    scale_factor: float = 1.0,
) -> None:
    latent = torch.randn((1,) + spatial_shape)
    latent = latent.to(device)

    prompt_embeds = torch.cat((49406 * torch.ones(1, 1), 49407 * torch.ones(1, 76)), 1).long()
    prompt_embeds = text_encoder(prompt_embeds.squeeze(1).to(device))
    prompt_embeds = prompt_embeds[0]

    for t in tqdm(scheduler.timesteps, ncols=70):
        noise_pred = model(x=latent, timesteps=torch.asarray((t,)).to(device), context=prompt_embeds)
        latent, _ = scheduler.step(noise_pred, t, latent)

    x_hat = stage1.decode(latent / scale_factor)
    img_0 = np.clip(a=x_hat[0, 0, :, :, 60].cpu().numpy(), a_min=0, a_max=1)
    fig = plt.figure(dpi=300)
    plt.imshow(img_0, cmap="gray")
    plt.axis("off")
    writer.add_figure("SAMPLE", fig, step)
