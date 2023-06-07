""" Script to compute the Frechet Inception Distance (FID) of the samples of the LDM. """
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from generative.metrics import FIDMetric
from monai import transforms
from monai.config import print_config
from monai.data import Dataset
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import get_test_dataloader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--sample_dir", help="Location of the samples to evaluate.")
    parser.add_argument("--test_ids", help="Location of file with test ids.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")

    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    samples_dir = Path(args.sample_dir)

    # Load pretrained model
    device = torch.device("cuda")
    model = torch.hub.load("Warvito/MedicalNet-models", "medicalnet_resnet50_23datasets")
    model = model.to(device)
    model.eval()

    # Samples
    samples_datalist = []
    for sample_path in sorted(list(samples_dir.glob("*.nii.gz"))):
        samples_datalist.append(
            {
                "image": str(sample_path),
            }
        )
    print(f"{len(samples_datalist)} images found in {str(samples_dir)}")

    sample_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            transforms.ToTensord(keys=["image"]),
        ]
    )

    samples_ds = Dataset(
        data=samples_datalist,
        transform=sample_transforms,
    )
    samples_loader = DataLoader(
        samples_ds,
        batch_size=16,
        shuffle=False,
        num_workers=8,
    )

    samples_features = []
    for batch in tqdm(samples_loader):
        img = batch["image"]
        with torch.no_grad():
            outputs = model(img.to(device))
            outputs = F.adaptive_avg_pool3d(outputs, (1, 1, 1)).view(outputs.size(0), -1)  # Global average pooling

        samples_features.append(outputs.cpu())
    samples_features = torch.cat(samples_features, dim=0)

    # Test set
    test_loader = get_test_dataloader(
        batch_size=16,
        test_ids=args.test_ids,
        num_workers=args.num_workers,
        upper_limit=1000,
    )

    test_features = []
    for batch in tqdm(test_loader):
        img = batch["image"]
        with torch.no_grad():
            outputs = model(img.to(device))
            outputs = F.adaptive_avg_pool3d(outputs, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # Global average pooling

        test_features.append(outputs.cpu())
    test_features = torch.cat(test_features, dim=0)

    # Compute FID
    metric = FIDMetric()
    fid = metric(samples_features, test_features)

    print(fid)


if __name__ == "__main__":
    args = parse_args()
    main(args)
