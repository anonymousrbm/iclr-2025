import argparse

from torchrbm.dataset import load_dataset
from torchrbm.dataset.parser import add_args_dataset
from torchrbm.parser import add_args_pytorch
from torchrbm.parser import remove_argument
from torchrbm.rcm.parser import add_args_rcm
from torchrbm.rcm.training import train


def create_parser():
    parser = argparse.ArgumentParser("Train a Restricted Coulomb Machine")
    parser = add_args_dataset(parser)
    parser = add_args_rcm(parser)
    parser = add_args_pytorch(parser)
    remove_argument(parser, "variable_type")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    args = vars(args)
    args["variable_type"] = "Ising"

    train_dataset, test_dataset = load_dataset(
        dataset_name=args["data"],
        subset_labels=args["subset_labels"],
        variable_type=args["variable_type"],
        use_weights=args["use_weights"],
        alphabet=args["alphabet"],
        train_size=args["train_size"],
        test_size=args["test_size"],
    )
    print(train_dataset)
    train(
        train_dataset=train_dataset.data,
        test_dataset=test_dataset.data,
        weights_train=train_dataset.weights,
        weights_test=test_dataset.weights,
        args=args,
        mesh_file=args["mesh_file"],
    )
