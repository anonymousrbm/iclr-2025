from typing import Optional, Tuple

import numpy as np
import torch

from torchrbm.dataset.dataset_class import RBMDataset
from torchrbm.dataset.definitions import ROOT_DIR_DATASET_PACKAGE
from torchrbm.dataset.load_fasta import load_FASTA
from torchrbm.dataset.load_h5 import load_HDF5


def load_dataset(
    dataset_name: str,
    subset_labels=None,
    variable_type: str = "Bernoulli",
    use_weights=False,
    train_size: float = 0.6,
    test_size: Optional[float] = None,
    alphabet="protein",
    seed: int = 19023741073419046239412739401234901,
    use_torch: bool = False,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Tuple[RBMDataset, RBMDataset]:
    rng = np.random.default_rng(seed)
    data = None
    weights = None
    names = None
    labels = None
    is_binary = True

    match dataset_name:
        # Binary datasets
        case "MNIST":
            dataset_name = str(
                (ROOT_DIR_DATASET_PACKAGE / "data/MNIST_train.h5").resolve()
            )
        case "HGD":
            dataset_name = str((ROOT_DIR_DATASET_PACKAGE / "data/HGD.h5").resolve())
        case "MICKEY":
            dataset_name = str((ROOT_DIR_DATASET_PACKAGE / "data/mickey.h5").resolve())
        case "CelebA_64_bw":
            dataset_name = str(
                (ROOT_DIR_DATASET_PACKAGE / "data/CelebA_64_bw.h5").resolve()
            )
        case "CelebA_64_bw_cont":
            dataset_name = str(
                (ROOT_DIR_DATASET_PACKAGE / "data/CelebA_64_bw_cont.h5").resolve()
            )
        case "CelebA_32_bw":
            dataset_name = str(
                (ROOT_DIR_DATASET_PACKAGE / "data/CelebA_32_bw.h5").resolve()
            )
        case "CIFAR10":
            dataset_name = str((ROOT_DIR_DATASET_PACKAGE / "data/CIFAR10.h5").resolve())
        case "2d3c":
            dataset_name = str((ROOT_DIR_DATASET_PACKAGE / "data/2d3c.h5").resolve())

        case _:
            pass

    if dataset_name[-3:] == ".h5":
        data, labels = load_HDF5(filename=dataset_name, variable_type=variable_type)
    elif dataset_name[-6:] == ".fasta":
        data, weights, names = load_FASTA(
            filename=dataset_name,
            variable_type=variable_type,
            use_weights=use_weights,
            alphabet=alphabet,
            device=device,
        )
        is_binary = False
    else:
        raise ValueError(
            """
            Dataset could not be loaded as the type is not recognized.
            It should be either:
                - '.h5',
                - '.fasta'
            """
        )
    # Select subset of dataset w.r.t. labels
    dataset_select = []
    labels_select = []
    if subset_labels is not None and labels is not None:
        for label in subset_labels:
            mask = labels == label
            dataset_select.append(np.array(data[mask], dtype=float))
            labels_select.append(np.array(labels[mask]))
        data = np.concatenate(dataset_select)
        labels = np.concatenate(labels_select)

    if weights is None:
        weights = np.ones(data.shape[0])
    if names is None:
        names = np.arange(data.shape[0])
    if labels is None:
        labels = -np.ones(data.shape[0])

    # Shuffle dataset
    permutation_index = rng.permutation(data.shape[0])

    # Split train/test
    train_size = int(train_size * data.shape[0])
    if test_size is not None:
        test_size = int(test_size * data.shape[0])
    else:
        test_size = data.shape[0] - train_size

    train_dataset = RBMDataset(
        data=data[permutation_index[:train_size]],
        variable_type=variable_type,
        labels=labels[permutation_index[:train_size]],
        weights=weights[permutation_index[:train_size]],
        names=names[permutation_index[:train_size]],
        dataset_name=dataset_name,
        is_binary=is_binary,
        use_torch=use_torch,
        device=device,
        dtype=dtype,
    )
    test_dataset = None
    if test_size > 0:
        test_dataset = RBMDataset(
            data=data[permutation_index[train_size : train_size + test_size]],
            variable_type=variable_type,
            labels=labels[permutation_index[train_size : train_size + test_size]],
            weights=weights[permutation_index[train_size : train_size + test_size]],
            names=names[permutation_index[train_size : train_size + test_size]],
            dataset_name=dataset_name,
            is_binary=is_binary,
            use_torch=use_torch,
            device=device,
            dtype=dtype,
        )
    return train_dataset, test_dataset
