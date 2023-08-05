from argparse import ArgumentParser


def cli() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--resnet",
        type=str,
        choices=["18", "50", "152"],
        required=True,
        help="model name",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=450,
        help="image dimension",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        default=False,
        help="Only validate model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="learning rate",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=400,
        help="epoch",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint path",
    )
    return parser


def inf_cli() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--resnet",
        type=str,
        choices=["18", "50", "152"],
        required=True,
        help="model name",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="image dimension",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="checkpoint path",
    )
    return parser
