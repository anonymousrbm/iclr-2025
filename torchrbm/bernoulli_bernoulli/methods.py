from pathlib import Path

import h5py
import torch

from torchrbm.bernoulli_bernoulli.classes import BBParams
from torchrbm.bernoulli_bernoulli.energy import compute_energy_visibles
from torchrbm.bernoulli_bernoulli.init import init_chains, init_parameters
from torchrbm.bernoulli_bernoulli.io import save_model
from torchrbm.bernoulli_bernoulli.partition_function import (
    compute_partition_function_ais,
)
from torchrbm.dataset.dataset_class import RBMDataset

Tensor = torch.Tensor


@torch.jit.script
def compute_log_likelihood(v_data: Tensor, params: BBParams, log_z: float) -> float:
    """Compute the log likelihood of the RBM on the data, given its log partition function.

    Parameters
    ----------
    v_data : Tensor
        Data to estimate LL
    params : Tensor
        Parameters of the RBM
    log_z : float
        Log partition function

    Returns
    ----------
    float:
        Log-likelihood
    """
    return -compute_energy_visibles(v_data, params).mean().item() - log_z


def create_machine(
    filename: str,
    num_visibles: int,
    num_hiddens: int,
    num_chains: int,
    batch_size: int,
    gibbs_steps: int,
    learning_rate: float,
    log: bool,
    dataset: RBMDataset,
    test_dataset: RBMDataset,
    device: torch.device,
    dtype: torch.dtype,
):
    """Create a RBM and save it to a new file.

    Parameters
    ----------
    filename : str
        Path to the h5 archive
    num_visibles : int
        Number of visible units
    num_hiddens : int
        Number of hidden units
    num_chains : int
        Number of parallel chains for gradient estimation
    batch_size : int
        Number of dataset samples used for gradient estimation
    gibbs_steps : int
        Number of Gibbs steps performed between each gradient update
    learning_rate : float
        Learning rate
    log : bool
        Deprecated
    dataset : RBMDataset
        Train dataset
    test_dataset : RBMDataset
        Test dataset
    device : torch.device
        PyTorch device for the model's parameters
    dtype : torch.dtype
        Dtype for the model's parameters
    """
    params = init_parameters(
        num_visibles=num_visibles,
        num_hiddens=num_hiddens,
        dataset=dataset,
        device=device,
        dtype=dtype,
    )
    parallel_chains = init_chains(num_samples=num_chains, params=params)

    with h5py.File(filename, "w") as file_model:
        hyperparameters = file_model.create_group("hyperparameters")
        hyperparameters["num_hiddens"] = num_hiddens
        hyperparameters["num_visibles"] = num_visibles
        hyperparameters["num_chains"] = num_chains
        hyperparameters["batch_size"] = batch_size
        hyperparameters["gibbs_steps"] = gibbs_steps
        hyperparameters["filename"] = str(filename)
        hyperparameters["learning_rate"] = learning_rate
        file_model["parallel_chains"] = parallel_chains.visible.cpu().numpy()

    # For now we estimate the initial log Z using AIS temperature
    log_z = compute_partition_function_ais(
        num_chains=1000, num_beta=5000, params=params
    )
    log_weights = torch.zeros(batch_size, device=device, dtype=dtype)

    energy_train_dataset = compute_energy_visibles(dataset.data, params)
    energy_test_dataset = compute_energy_visibles(test_dataset.data, params)

    train_ll = (-energy_train_dataset - log_z).mean().item()
    test_ll = (-energy_test_dataset - log_z).mean().item()

    save_model(
        filename=filename,
        params=params,
        chains=parallel_chains,
        num_updates=0,
        time=0,
        save_ll=True,
        save_ptt=True,
        train_ll=train_ll,
        test_ll=test_ll,
        log_weights=log_weights,
        log_z=log_z,
    )
    if log:
        filename = Path(filename)
        log_filename = filename.parent / Path(f"log-{filename.stem}.csv")
        with open(log_filename, "w", encoding="utf-8") as log_file:
            log_file.write("ess,eps\n")
