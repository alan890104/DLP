from torch.utils.data import Dataset, DataLoader
from typing import Literal
from PIL import Image
import pandas as pd
from glob import glob
import os


class ResNetDataset(Dataset):
    def __init__(
        self,
        kind: Literal["18", "50", "152"],
        mode: Literal["train", "valid"],
        transform=None,
    ):
        sub = "valid.csv" if mode else "train.csv"
        self.root = os.path.join("dataset", f"resnet{kind}")
        self.index = pd.read_csv(os.path.join(self.root, sub))
        self.transform = transform

    def __getitem__(self, index):
        row = self.index.iloc[index]
        path = os.path.join(self.root, "new_dataset", row["Path"])
        label = row["label"]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, path

    def __len__(self):
        return len(self.index)


class EvalDataset:
    def __init__(
        self,
        kind: Literal["18", "50", "152"],
        transform=None,
    ):
        self.paths = glob(
            os.path.join(
                "dataset",
                f"resnet{kind}",
                "new_dataset",
                "new_dataset",
                "test",
                "*.bmp",
            )
        )
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        path = f"./new_dataset/test/{os.path.basename(path)}"
        return image, path

    def __len__(self):
        return len(self.paths)


if __name__ == "__main__":
    from torchvision import transforms

    tfs = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = ResNetDataset("18", "train", tfs)
    print(dataset[0][0].shape)

    print(dataset.index["label"].value_counts())
