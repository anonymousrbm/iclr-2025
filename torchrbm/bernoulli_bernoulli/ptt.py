from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from torchrbm.bernoulli_bernoulli.classes import BBParams
from torchrbm.bernoulli_bernoulli.grad import compute_gradient
from torchrbm.bernoulli_bernoulli.init import init_chains, init_data_state
from torchrbm.bernoulli_bernoulli.io import load_params
from torchrbm.bernoulli_bernoulli.methods import compute_energy_visibles
from torchrbm.bernoulli_bernoulli.sampling import sample_state
from torchrbm.classes import Chain
from torchrbm.rcm.rbm import sample_rbm

Tensor = torch.Tensor


def init_sampling(
    n_gen: int,
    list_params: List[BBParams],
    rcm: Optional[dict] = None,
    start_v: Optional[Tensor] = None,
    it_mcmc: int = 1000,
    device: torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    show_pbar: bool = True,
) -> List[Chain]:
    use_dataset = start_v is not None
    use_rcm = (rcm is not None) and not use_dataset

    all_chains = []
    if show_pbar:
        pbar = tqdm(total=len(list_params))
        pbar.set_description("Initializing PTT chains")

    for i, params in enumerate(list_params):
        init_v = torch.bernoulli(
            torch.ones(n_gen, params.vbias.shape[0], device=device, dtype=dtype)
        )

        # Start every model with independant configurations from the RCM
        if use_rcm:
            init_v = (
                sample_rbm(
                    p_m=rcm["p_m"],
                    mu=rcm["mu"],
                    U=rcm["U"],
                    num_samples=n_gen,
                    device=device,
                    dtype=dtype,
                )
                + 1
            ) / 2

        # Start every model from random permutations of the input dataset
        if use_dataset:
            perm_index = torch.randperm(start_v.shape[0])
            init_v = start_v[perm_index][:n_gen]

        chains = init_chains(num_samples=n_gen, params=params, start_v=init_v)

        # Iterate over the chains for some time
        chains = sample_state(gibbs_steps=it_mcmc, chains=chains, params=params)

        all_chains.append(
            Chain(
                visible=chains.visible.clone(),
                mean_visible=None,
                hidden=chains.hidden.clone(),
                mean_hidden=None,
            )
        )
        if show_pbar:
            pbar.update(1)
    return all_chains


def sampling_step(
    rcm: dict,
    list_params: List[BBParams],
    chains: List[Chain],
    it_mcmc: int,
) -> List[Chain]:
    """Performs it_mcmc sampling steps with all the models and samples a new configurartion for the RCM.

    Args:
        rcm (dict): Dict with rcm parameters
        fname_rbm (str): Path to rbm.
        chains (Tensor): Previous configuration of the chains.
        it_mcmc (int): Number of steps to perform.
        epochs (list): List of epochs to be used for the sampling.

    Returns:
        Tensor: Updated chains.
    """
    n_chains = len(chains[0].visible)

    use_rcm = rcm is not None
    # Sample from the rcm
    if use_rcm:
        # The first provided params should be the ones from the RCM
        gen0 = sample_rbm(
            p_m=rcm["p_m"],
            mu=rcm["mu"],
            U=rcm["U"],
            num_samples=n_chains,
            device=chains[0].visible.device,
            dtype=torch.float32,
        )
        gen0 = (gen0 + 1) / 2
        chains[0] = init_chains(n_chains, list_params[0], start_v=gen0)

    # Sample from rbm
    for idx, params in enumerate(list_params[int(use_rcm) :], int(use_rcm)):
        chains[idx] = sample_state(
            chains=chains[idx], params=params, gibbs_steps=it_mcmc
        )
    return chains


@torch.jit.script
def swap_config_multi(
    params: List[BBParams], chains: List[Chain], index: Optional[List[Tensor]] = None
) -> Tuple[List[Chain], Tensor, Optional[List[Tensor]]]:
    n_chains, L = chains[0].visible.shape
    n_rbms = len(params)
    acc_rate = torch.zeros(n_rbms - 1)
    for idx in range(n_rbms - 1):
        delta_energy = (
            -compute_energy_visibles(chains[idx + 1].visible, params[idx])
            + compute_energy_visibles(chains[idx].visible, params[idx])
            + compute_energy_visibles(chains[idx + 1].visible, params[idx + 1])
            - compute_energy_visibles(chains[idx].visible, params[idx + 1])
        )
        # print(torch.exp(delta_energy))
        ## Since PyTorch 2.3.0 there is a swap_tensors function which might be more efficient here
        swap = torch.exp(delta_energy) > torch.rand(
            size=(n_chains,), device=delta_energy.device
        )

        if index is not None:
            swapped_index_0 = torch.where(swap, index[idx + 1], index[idx])
            swapped_index_1 = torch.where(swap, index[idx], index[idx + 1])
            index[idx] = swapped_index_0
            index[idx + 1] = swapped_index_1

        acc_rate[idx] = (swap.sum() / n_chains).cpu()
        # Put the swap mask at the same shape as the chains
        swap = swap.unsqueeze(1).repeat(1, L)
        swapped_chains_0 = torch.where(
            swap,
            chains[idx + 1].visible,
            chains[idx].visible,
        )
        swapped_chains_1 = torch.where(
            swap,
            chains[idx].visible,
            chains[idx + 1].visible,
        )
        chains[idx].visible = swapped_chains_0
        chains[idx + 1].visible = swapped_chains_1

    return chains, acc_rate, index


def ptt_sampling(
    rcm: dict,
    list_params: List[BBParams],
    chains: List[Chain],
    index: Optional[List[Tensor]],
    it_mcmc: int = None,
    increment: int = 10,
    show_pbar: bool = True,
    show_acc_rate: bool = True,
) -> Tuple[List[Chain], Tensor, Optional[List[Tensor]]]:
    assert (
        len(list_params) == len(chains)
    ), f"list_params and chains must have the same length, but got {len(list_params)} and {len(chains)}"
    if show_pbar:
        pbar = tqdm(total=it_mcmc, leave=False)
    acc_rates = torch.zeros(len(list_params) - 1)
    for steps in range(0, it_mcmc, increment):
        if show_pbar:
            pbar.update(increment)
        chains, acc_rate, index = swap_config_multi(
            chains=chains, params=list_params, index=index
        )
        acc_rates += acc_rate
        chains = sampling_step(
            rcm=rcm,
            list_params=list_params,
            chains=chains,
            it_mcmc=increment,
        )
    acc_rates /= it_mcmc // increment
    if show_pbar:
        pbar.close()
    if show_acc_rate:
        print("acc_rate: ", acc_rates)
    return chains, acc_rates, index


def get_epochs_pt_sampling(
    filename: str,
    rcm: Optional[dict],
    updates: np.ndarray,
    target_acc_rate: float,
    device: torch.device,
    dtype: torch.dtype,
):
    use_rcm = rcm is not None
    selected_idx = [updates[0]]
    pbar = tqdm(enumerate(updates[1:], 1), total=len(updates[1:]))
    num_chains = 100
    it_mcmc = 10

    # Init first chain
    params = load_params(filename, updates[0], device=device, dtype=dtype)
    if use_rcm:
        new_chains = sample_rbm(
            p_m=rcm["p_m"],
            mu=rcm["mu"],
            U=rcm["U"],
            num_samples=num_chains,
            device=device,
            dtype=dtype,
        )
        new_chains = init_chains(num_chains, params, start_v=(new_chains + 1) / 2)
    else:
        new_chains = init_chains(num_samples=num_chains, params=params)
        new_chains = sample_state(chains=new_chains, params=params, gibbs_steps=it_mcmc)
        rcm = None
    # Copy of the first chain
    chains = [
        Chain(
            visible=new_chains.visible.clone(),
            mean_visible=None,
            hidden=None,
            mean_hidden=None,
        )
    ]
    list_params = [params.clone()]
    for i, idx in pbar:
        # Copy of the previous configuration to revert to it after
        saved_chains = new_chains.visible.cpu().clone()

        selected_idx.append(idx)
        params = load_params(filename=filename, index=idx, device=device, dtype=dtype)
        new_chains = sample_state(chains=new_chains, params=params, gibbs_steps=it_mcmc)

        chains.append(new_chains)
        list_params.append(params)
        chains, acc_rate, _ = ptt_sampling(
            rcm=rcm,
            list_params=list_params,
            chains=chains,
            index=None,
            it_mcmc=1,
            increment=1,
            show_pbar=False,
        )
        selected_idx.pop(-1)
        pbar.write(str(acc_rate[-1]))
        if acc_rate[-1] < target_acc_rate:
            selected_idx.append(updates[i - 1])
            pbar.write(str(selected_idx))
        else:
            new_chains.visible = saved_chains.cuda()
            chains.pop(-1)
            list_params.pop(-1)
    if updates[-1] not in selected_idx:
        selected_idx.append(updates[-1])
    return np.array(selected_idx)
