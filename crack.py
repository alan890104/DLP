import torch
from typing import Literal
from model import ResNet18, ResNet50, ResNet152, ResNetFactory, _ResNet
from data import EvalDataset
from torch.utils.data import DataLoader
from cli import inf_cli
import pandas as pd
from torchvision import transforms


@torch.no_grad()
def main():
    test_tfs = transforms.Compose(
        [
            transforms.Resize(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    testset = EvalDataset("152", test_tfs)
    test_loader = DataLoader(testset, batch_size=64, shuffle=False)

    paths = []
    preds = []

    for _, path in test_loader:
        paths.extend(path)
        preds.extend([1] * len(path))

    pd.DataFrame(
        {
            "ID": paths,
            "label": preds,
        }
    ).to_csv(f"crack.csv", index=False)


if __name__ == "__main__":
    main()
