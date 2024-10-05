import argparse

import torch

from torchrbm.bernoulli_bernoulli.train import train as train_bernoulli
from torchrbm.dataset import load_dataset
from torchrbm.dataset.parser import add_args_dataset
from torchrbm.parser import (
    add_args_pytorch,
    add_args_rbm,
    add_args_saves,
    remove_argument,
)
from torchrbm.utils import get_checkpoints


def create_parser():
    parser = argparse.ArgumentParser(description="Train a Restricted Boltzmann Machine")
    parser = add_args_dataset(parser)
    parser = add_args_rbm(parser)
    parser = add_args_saves(parser)
    parser = add_args_pytorch(parser)
    remove_argument(parser, "use_torch")
    return parser


def main(args: dict):
    checkpoints = get_checkpoints(
        num_updates=args["num_updates"], n_save=args["n_save"], spacing=args["spacing"]
    )
    train_dataset, test_dataset = load_dataset(
        dataset_name=args["data"],
        subset_labels=args["subset_labels"],
        variable_type=args["variable_type"],
        use_weights=args["use_weights"],
        alphabet=args["alphabet"],
        train_size=args["train_size"],
        test_size=args["test_size"],
        use_torch=not (args["use_numpy"]),
        device=args["device"],
        dtype=args["dtype"],
    )
    print(train_dataset)
    train_bernoulli(
        dataset=train_dataset,
        test_dataset=test_dataset,
        args=args,
        dtype=args["dtype"],
        checkpoints=checkpoints,
    )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)
    match args["dtype"]:
        case "int":
            args["dtype"] = torch.int64
        case "float":
            args["dtype"] = torch.float32
        case "double":
            args["dtype"] = torch.float64
    main(args=args)
