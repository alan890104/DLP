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
    args = inf_cli().parse_args()
    print(args)

    test_tfs = transforms.Compose(
        [
            transforms.Resize(args.dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    nn: _ResNet = ResNetFactory(args.resnet)
    nn.load_state_dict(torch.load(args.checkpoint))
    nn.cuda()
    testset = EvalDataset(args.resnet, test_tfs)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    paths = []
    preds = []

    for data, path in test_loader:
        data = data.cuda()
        output = nn(data)
        pred = output.argmax(dim=1)

        paths.extend(path)
        preds.extend(pred.tolist())

    pd.DataFrame(
        {
            "ID": paths,
            "label": preds,
        }
    ).to_csv(f"{args.checkpoint}.csv", index=False)


if __name__ == "__main__":
    main()
