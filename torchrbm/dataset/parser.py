import argparse


def add_args_dataset(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    dataset_args = parser.add_argument_group("Dataset")
    dataset_args.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="Name of the dataset ('HGD', 'MNIST', 'BKACE', 'PF00072', 'PF13354'), or path to a data file (type should be .h5 or .fasta)",
    )
    dataset_args.add_argument(
        "--subset_labels",
        nargs="*",
        default=None,
        type=int,
        help="(Defaults to None). The subset of labels to use during training. None means all dataset.",
    )
    dataset_args.add_argument(
        "--train_size",
        type=float,
        default=0.6,
        help="(Defaults to 0.6). The proportion of the dataset to use as training set.",
    )
    dataset_args.add_argument(
        "--test_size",
        type=float,
        default=None,
        help="(Defaults to None). The proportion of the dataset ot use as testing set.",
    )
    dataset_args.add_argument(
        "--variable_type",
        type=str,
        default="Bernoulli",
        help="(Defaults to 'Bernoulli'). The type of the variables of the dataset.",
        choices=["Bernoulli", "Ising", "Continuous", "Potts"],
    )
    dataset_args.add_argument(
        "--use_weights",
        default=False,
        action="store_true",
        help="Compute the weights associated to each sequence.",
    )
    dataset_args.add_argument(
        "--alphabet",
        type=str,
        default="protein",
        choices=["protein", "rna", "dna"],
        help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.",
    )
    dataset_args.add_argument(
        "--use_numpy",
        default=False,
        action="store_true",
        help="(Defaults to False). Load the dataset with numpy arrays.",
    )
    return parser
