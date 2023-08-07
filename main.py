import torch
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torchvision import transforms
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter
from model import _ResNet, ResNetFactory
from data import ResNetDataset
from cli import cli
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns
import random
import os


def set_seed(seed: int = 890104):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


os.makedirs("checkpoint", exist_ok=True)

assert torch.cuda.is_available(), "CUDA is not available"

summary = SummaryWriter()


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    criterion: _Loss,
    optimizer: Optimizer,
    epoch: int,
    weight: torch.Tensor,
):
    model.train()
    train_loss = 0
    correct = 0
    for _, (data, target, _) in enumerate(train_loader):
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        loss = criterion(output, target, weight=weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1)
        train_loss += loss.item()
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)
    summary.add_scalar("Loss/train", train_loss, global_step=epoch)
    summary.add_scalar("Acc/train", train_acc, global_step=epoch)
    return train_loss, train_acc


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    criterion: _Loss,
    epoch: int = None,
    weight: torch.Tensor = None,
):
    model.eval()
    val_loss = 0
    correct = 0
    preds = []
    actuals = []

    for data, target, _ in test_loader:
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        pred = output.argmax(dim=1)
        loss = criterion(output, target, weight=weight)
        val_loss += loss.item()
        correct += pred.eq(target.view_as(pred)).sum().item()
        preds.extend(pred.tolist())
        actuals.extend(target.tolist())

    val_loss /= len(test_loader.dataset)
    val_acc = correct / len(test_loader.dataset)
    if epoch is not None:
        summary.add_scalar("Loss/train", val_loss, global_step=epoch)
        summary.add_scalar("Acc/train", val_acc, global_step=epoch)
    return val_loss, val_acc, preds, actuals


def main():
    args = cli().parse_args()
    print(args)

    set_seed(890104)

    g = torch.Generator()
    g.manual_seed(890104)

    train_tfs = transforms.Compose(
        [
            transforms.Resize(args.dim),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(60),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_tfs = transforms.Compose(
        [
            transforms.Resize(args.dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    epoch = args.epoch
    nn: _ResNet = ResNetFactory(args.resnet)
    criterion = F.cross_entropy

    if args.checkpoint:
        print(f"Loading checkpoint {args.checkpoint}")
        nn.load_state_dict(torch.load(args.checkpoint))

    optimizer = torch.optim.Adam(nn.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-5)

    trainset: ResNetDataset = ResNetDataset(args.resnet, "train", train_tfs)
    validset: ResNetDataset = ResNetDataset(args.resnet, "valid", val_tfs)
    train_loader: DataLoader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
    )

    valid_loader: DataLoader = DataLoader(
        validset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
    )

    # This is for class imbalance, however, it does not work well
    #
    # class_weight = compute_class_weight(
    #     class_weight="balanced",
    #     classes=np.unique(trainset.index["label"]),
    #     y=trainset.index["label"],
    # )
    # weight = torch.tensor(class_weight, dtype=torch.float).cuda()

    weight = None
    nn = nn.cuda()

    # train
    if not args.val:
        baseline_acc = 0.8
        with tqdm(total=epoch) as pbar:
            for e in range(1, epoch + 1):
                train_loss, train_acc = train(
                    nn,
                    train_loader,
                    criterion,
                    optimizer,
                    e,
                    weight,
                )
                val_loss, val_acc, _, _ = validate(
                    nn,
                    valid_loader,
                    criterion,
                    e,
                    weight,
                )
                scheduler.step()

                pbar.set_description(
                    f"Epoch {e} | train_loss: {round(train_loss, 4)} | train_acc: {round(train_acc*100, 2)}% | val_loss: {round(val_loss, 4)} | val_acc: {round(val_acc*100, 2)}%"
                )
                pbar.update(1)

                if val_acc > baseline_acc:
                    baseline_acc = val_acc
                    torch.save(
                        nn.state_dict(),
                        f"checkpoint/resnet{args.resnet}-epoch{e}-acc{round(val_acc*100,2)}.pt",
                    )

    val_loss, val_acc, preds, actuals = validate(nn, valid_loader, criterion)
    matrix = confusion_matrix(actuals, preds)
    print(f"Validation loss: {val_loss}, Validation acc: {val_acc}")
    print(matrix)
    sns.heatmap(matrix, annot=True).figure.savefig(f"confmatrix-{args.resnet}.png")


if __name__ == "__main__":
    main()
