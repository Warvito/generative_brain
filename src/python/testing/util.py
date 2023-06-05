"""Utility functions for testing."""
from __future__ import annotations

import pandas as pd
from monai import transforms
from monai.data import Dataset
from torch.utils.data import DataLoader


def get_test_dataloader(
    batch_size: int,
    test_ids: str,
    num_workers: int = 8,
    upper_limit: int | None = None,
):
    test_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
            transforms.SpatialCropd(keys=["image"], roi_start=[16, 16, 96], roi_end=[176, 240, 256]),
            transforms.SpatialPadd(
                keys=["image"],
                spatial_size=[160, 224, 160],
            ),
            transforms.ToTensord(keys=["image"]),
        ]
    )

    test_dicts = get_datalist(ids_path=test_ids, upper_limit=upper_limit)
    test_ds = Dataset(data=test_dicts, transform=test_transforms)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
        persistent_workers=True,
    )

    return test_loader


def get_datalist(
    ids_path: str,
    upper_limit: int | None = None,
):
    """Get data dicts for data loaders."""
    df = pd.read_csv(ids_path, sep="\t")

    if upper_limit is not None:
        df = df[:upper_limit]

    data_dicts = []
    for index, row in df.iterrows():
        data_dicts.append(
            {
                "image": str(row["image"]),
            }
        )

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts
