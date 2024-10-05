import itertools
import pathlib
import sys

import h5py
import numpy as np
import scipy
import torch

Tensor = torch.Tensor


def get_checkpoints(num_updates: int, n_save: int, spacing: str = "exp") -> np.ndarray:
    """Select the list of training times (ages) at which saving the model

    Parameters
    ----------
    num_updates : int
        The number of gradient updates to perform during training
    n_save : int
        The number of models to save
    spacing : str, optional
        linear ("lin") or exponential spacing ("exp"), by default "exp"

    Returns
    -------
    np.ndarray
        The array with the index of the models to save
    """
    if spacing == "exp":
        checkpoints = []
        xi = num_updates
        for _ in range(n_save):
            checkpoints.append(xi)
            xi = xi / num_updates ** (1 / n_save)
        checkpoints = np.unique(np.array(checkpoints, dtype=np.int32))
    elif spacing == "linear":
        checkpoints = np.linspace(1, num_updates, n_save).astype(np.int32)
    checkpoints = np.unique(np.append(checkpoints, num_updates))
    return checkpoints


def get_eigenvalues_history(filename):
    with h5py.File(filename, "r") as f:
        gradient_updates = []
        eigenvalues = []
        for key in f.keys():
            if "update" in key:
                weight_matrix = f[key]["weight_matrix"][()]
                weight_matrix = weight_matrix.reshape(-1, weight_matrix.shape[-1])
                eig = scipy.linalg.svd(weight_matrix, compute_uv=False)
                eigenvalues.append(eig.reshape(*eig.shape, 1))
                gradient_updates.append(int(key.split("_")[1]))

        # Sort the results
        sorting = np.argsort(gradient_updates)
        gradient_updates = np.array(gradient_updates)[sorting]
        eigenvalues = np.array(np.hstack(eigenvalues).T)[sorting]

    return gradient_updates, eigenvalues


def get_saved_updates(filename: str) -> np.ndarray:
    updates = []
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            if "update" in key:
                update = int(key.replace("update_", ""))
                updates.append(update)
    return np.sort(np.array(updates))


def get_ptt_updates(filename: str) -> np.ndarray:
    ptt_updates = []
    updates = []
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            if "update" in key:
                update = int(key.replace("update_", ""))
                updates.append(update)
                if f[key]["save_ptt"][()]:
                    ptt_updates.append(update)
    updates = np.sort(np.array(updates))
    ptt_updates.append(updates[-1])
    return np.sort(np.array(ptt_updates))


def get_ll_updates(filename: str) -> np.ndarray:
    ll_updates = []
    updates = []
    with h5py.File(filename, "r") as f:
        for key in f.keys():
            if "update" in key:
                update = int(key.replace("update_", ""))
                updates.append(update)
                if f[key]["save_ll"]:
                    ll_updates.append(update)
    updates = np.sort(np.array(updates))
    ll_updates.append(updates[-1])
    return np.sort(np.array(ll_updates))


def get_binary_configurations(
    n_dim: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    max_dim = 20
    if n_dim > max_dim:
        raise ValueError(
            f"The number of dimension for the binary configurations exceeds the maximum number of dimension: {max_dim}"
        )
    return (
        torch.from_numpy(np.array(list(itertools.product(range(2), repeat=n_dim))))
        .to(dtype)
        .to(device)
    )


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Credits to "https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input"
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def check_file_existence(filename: str):
    if pathlib.Path(filename).exists():
        question = f"File: {filename} exists. Do you want to override it ?"
        if query_yes_no(question=question, default="yes"):
            print(f"Deleting {filename}.")
            pathlib.Path(filename).unlink()
        else:
            print("No overriding.")
            sys.exit(0)
