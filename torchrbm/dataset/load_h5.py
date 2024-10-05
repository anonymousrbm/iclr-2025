import h5py
import numpy as np


def load_HDF5(filename, variable_type="Bernoulli"):
    labels = None
    with h5py.File(filename, "r") as f:
        if "samples" not in f.keys():
            raise ValueError(
                f"Could not find 'samples' key if hdf5 file keys: {f.keys()}"
            )
        dataset = f["samples"][()]
        if "labels" in f.keys():
            labels = f["labels"][()]
            if labels.shape[0] != dataset.shape[0]:
                print(
                    f"Ignoring labels since its dimension ({labels.shape[0]}) does not match the number of samples ({dataset.shape[0]})."
                )
                labels = None
    if "cont" not in filename:
        unique_values = np.unique(dataset)
        is_ising = np.all(unique_values == np.array([-1, 1]))
        is_bernoulli = np.all(unique_values == np.array([0, 1]))
    else:
        unique_values = [0, 1]
        is_bernoulli = True
        is_ising = False
    if len(unique_values) != 2:
        raise ValueError(
            f"The dataset should be binary valued but got {len(unique_values)} different unique values: {unique_values}"
        )
    if not (is_ising or is_bernoulli):
        raise ValueError(
            f"The dataset should have either [0, 1] or [-1, 1] values, got {unique_values}"
        )

    match variable_type:
        case "Bernoulli":
            if is_ising:
                dataset = (dataset + 1) / 2
        case "Ising":
            if is_bernoulli:
                dataset = dataset * 2 - 1
        case _:
            raise ValueError(
                f"Variable type {variable_type} is not valid for binary dataset. Use 'Bernoulli' or 'Ising'."
            )
    return dataset, labels
