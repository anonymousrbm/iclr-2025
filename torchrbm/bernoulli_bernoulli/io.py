from typing import Tuple

import h5py
import numpy as np
import torch

from torchrbm.bernoulli_bernoulli.classes import BBParams
from torchrbm.classes import Chain
from torchrbm.utils import get_saved_updates

Tensor = torch.Tensor


def save_model(
    filename: str,
    params: BBParams,
    chains: Chain,
    num_updates: int,
    time: float,
    log_z: float,
    log_weights: Tensor,
    train_ll: float,
    test_ll: float,
    save_ll: bool = False,
    save_ptt: bool = True,
):
    """Save the current state of the model

    Parameters
    ----------
    filename : str
        Path to the h5 archive
    params : BBParams
        Parameters of the RBM
    chains : Chain
        Parallel chains used during training
    num_updates : int
        Number of elapsed updates
    time : float
        Elapsed time since the beginning of the training
    log_z : float
        Estimation of the log partition function (AIS traj)
    log_weight : Tensor
        Weights used during the estimation of the log partition function (AIS traj)
    train_ll : float
        Estimation of the train LL (AIS traj)
    test_ll : float
        Estimation of the test LL (AIS traj)
    save_ll : bool
        Flag the update to be used to estimate LL, by default False
    save_ptt: bool
        Flag the update to be used when performing PTT, by default False
    """

    with h5py.File(filename, "a") as f:
        checkpoint = f.create_group(f"update_{num_updates}")

        # Save the parameters of the model
        checkpoint["vbias"] = params.vbias.detach().cpu().numpy()
        checkpoint["hbias"] = params.hbias.detach().cpu().numpy()
        checkpoint["weight_matrix"] = params.weight_matrix.detach().cpu().numpy()

        # Save current random state
        checkpoint["torch_rng_state"] = torch.get_rng_state()
        checkpoint["numpy_rng_arg0"] = np.random.get_state()[0]
        checkpoint["numpy_rng_arg1"] = np.random.get_state()[1]
        checkpoint["numpy_rng_arg2"] = np.random.get_state()[2]
        checkpoint["numpy_rng_arg3"] = np.random.get_state()[3]
        checkpoint["numpy_rng_arg4"] = np.random.get_state()[4]
        checkpoint["time"] = time

        # LL AIS trajectory variables
        checkpoint["log_z"] = log_z
        checkpoint["log_weights"] = log_weights.cpu().numpy()

        # LL
        checkpoint["train_ll"] = train_ll
        checkpoint["test_ll"] = test_ll

        # save flags for PTT and LL
        checkpoint["save_ptt"] = save_ptt
        checkpoint["save_ll"] = save_ll

        # Update the parallel chains to resume training
        if "parallel_chains" in f.keys():
            f["parallel_chains"][...] = chains.visible.cpu().numpy()
        else:
            f["parallel_chains"] = chains.visible.cpu().numpy()

        # Update the total number of epochs
        if "epochs" in f["hyperparameters"].keys():
            del f["hyperparameters"]["epochs"]


def load_params(
    filename: str, index: int, device: torch.device, dtype: torch.dtype
) -> BBParams:
    """Load the parameters of the RBM from the specified archive at the given update index

    Parameters
    ----------
    filename : str
        Path to the h5 archive
    index : int
        Index of the machine to load
    device : torch.device
        PyTorch device on which to load the parameters
    dtype : torch.dtype
        Dtype for the parameters

    Returns
    ----------
    BBParams
        The parameters of the RBM
    """
    last_file_key = f"update_{index}"
    with h5py.File(filename, "r") as f:
        weight_matrix = torch.tensor(
            f[last_file_key]["weight_matrix"][()],
            device=device,
            dtype=dtype,
        )
        vbias = torch.tensor(f[last_file_key]["vbias"][()], device=device, dtype=dtype)
        hbias = torch.tensor(f[last_file_key]["hbias"][()], device=device, dtype=dtype)
    return BBParams(weight_matrix=weight_matrix, vbias=vbias, hbias=hbias)


def load_model(
    filename: str,
    index: int,
    device: torch.device,
    dtype: torch.dtype,
    set_rng_state: bool = False,
) -> Tuple[BBParams, Chain, float, dict]:
    """Load a RBM from a h5 archive

    Parameters.nu
    ----------
    filename : str
        Path to the h5 archive
    index : int
        Index of the machine to load
    device : torch.device
        PyTorch device on which to load the parameters and the chains
    dtype : torch.dtype
        Dtype for the parameters and the chains
    set_rng_state: bool
        Restore the random state at the given epoch (Useful to restore training). By default False

    Returns
    -------
    Tuple[BBParams, Chain, float, dict]
        RBM parameters, parallel chains, time elapsed since the beginning of the training, hyperparameters
    """
    last_file_key = f"update_{index}"
    hyperparameters = dict()
    with h5py.File(filename, "r") as f:
        visible = torch.from_numpy(f["parallel_chains"][()]).to(device).to(dtype)
        start = np.array(f[last_file_key]["time"]).item()
        # Elapsed time
        start = np.array(f[last_file_key]["time"]).item()

        # Hyperparameters
        hyperparameters["batch_size"] = int(f["hyperparameters"]["batch_size"][()])
        hyperparameters["gibbs_steps"] = int(f["hyperparameters"]["gibbs_steps"][()])
        hyperparameters["learning_rate"] = float(
            f["hyperparameters"]["learning_rate"][()]
        )
        if "list_idx_sample" in f.keys():
            hyperparameters["list_idx_sample"] = f["list_idx_sample"][()]
        else:
            hyperparameters["list_idx_sample"] = None
    params = load_params(filename, index, device, dtype)

    mean_visible = torch.zeros_like(visible, device=device, dtype=dtype)
    mean_hidden = torch.sigmoid((params.hbias + (visible @ params.weight_matrix)))
    hidden = torch.bernoulli(mean_hidden)

    perm_chains = Chain(
        visible=visible,
        mean_visible=mean_visible,
        hidden=hidden,
        mean_hidden=mean_hidden,
    ).to(device)
    if set_rng_state:
        with h5py.File(filename, "r") as f:
            torch.set_rng_state(
                torch.tensor(np.array(f[last_file_key]["torch_rng_state"]))
            )
            np_rng_state = tuple(
                [
                    f[last_file_key]["numpy_rng_arg0"][()].decode("utf-8"),
                    f[last_file_key]["numpy_rng_arg1"][()],
                    f[last_file_key]["numpy_rng_arg2"][()],
                    f[last_file_key]["numpy_rng_arg3"][()],
                    f[last_file_key]["numpy_rng_arg4"][()],
                ]
            )
            np.random.set_state(np_rng_state)
    return (params, perm_chains, start, hyperparameters)


def load_rcm(
    filename: str,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> dict[str, Tensor] | None:
    """Load the RCM saved in the given RBM h5 archive.

    Parameters
    ----------
    filename : str
        Path to the h5 archive
    device : torch.device
        PyTorch device on which to load the RCM
    dtype: torch.dtype
        Dtype for the RCM

    Returns
    ----------
    dict[str, Tensor] | None
        The RCM if it exists, otherwise None
    """
    rcm = None
    with h5py.File(filename, "r") as f:
        if "rcm" in f.keys():
            U = torch.from_numpy(np.array(f["rcm"]["U"])).to(device=device, dtype=dtype)
            m = torch.from_numpy(np.array(f["rcm"]["m"])).to(device=device, dtype=dtype)
            mu = torch.from_numpy(np.array(f["rcm"]["mu"])).to(
                device=device, dtype=dtype
            )
            p_m = torch.from_numpy(np.array(f["rcm"]["pdm"])).to(
                device=device, dtype=dtype
            )
            rcm = {"U": U, "m": m, "mu": mu, "p_m": p_m}
    return rcm


def load_ais_traj_params(filename: str, index: int) -> Tuple[float, np.ndarray]:
    saved_updates = get_saved_updates(filename)
    with h5py.File(filename, "r") as f:
        log_z_init = f[f"update_{saved_updates[0]}"]["log_z"][()]  # float
        log_weights = f[f"update_{index}"]["log_weights"][()]
    return log_z_init, log_weights
