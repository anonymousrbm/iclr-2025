from typing import List
from typing import Optional
from typing import Tuple

import h5py
import torch

from torchrbm.bernoulli_bernoulli.classes import BBParams
from torchrbm.bernoulli_bernoulli.energy import compute_energy
from torchrbm.bernoulli_bernoulli.init import init_chains
from torchrbm.bernoulli_bernoulli.sampling import sample_state
from torchrbm.classes import Chain

Tensor = torch.Tensor


def find_inverse_temperatures(target_acc_rate: float, params: BBParams) -> Tensor:
    inverse_temperatures = torch.linspace(0, 1, 1000)
    selected_temperatures = [0]
    num_visibles, num_hidden = params.weight_matrix.shape
    n_chains = 100

    prev_chains = init_chains(num_samples=n_chains, params=params)
    new_chains = init_chains(num_samples=n_chains, params=params)
    for i in range(len(inverse_temperatures) - 1):
        prev_chains = sample_state(
            gibbs_steps=10,
            chains=prev_chains,
            params=params,
            beta=selected_temperatures[-1],
        )
        new_chains = sample_state(
            gibbs_steps=10,
            chains=new_chains,
            params=params,
            beta=inverse_temperatures[i],
        )

        _, acc_rate, _ = swap_configurations(
            chains=[prev_chains, new_chains],
            params=params,
            inverse_temperatures=torch.tensor(
                [selected_temperatures[-1], inverse_temperatures[i]]
            ),
        )
        if acc_rate[-1] < target_acc_rate + 0.1:
            selected_temperatures.append(inverse_temperatures[i])
            prev_chains = new_chains.clone()
    if selected_temperatures[-1] != 1.0:
        selected_temperatures.append(1)
    return torch.tensor(selected_temperatures)


def swap_configurations(
    chains: List[Chain],
    params: BBParams,
    inverse_temperatures: Tensor,
    index: Optional[List[Tensor]] = None,
):
    n_chains, L = chains[0].visible.shape
    acc_rate = []
    for idx in range(inverse_temperatures.shape[0] - 1):
        energy_0 = compute_energy(
            v=chains[idx].visible, h=chains[idx].hidden, params=params
        )
        energy_1 = compute_energy(
            v=chains[idx + 1].visible, h=chains[idx + 1].hidden, params=params
        )
        delta_energy = (
            -energy_1 * inverse_temperatures[idx]
            + energy_0 * inverse_temperatures[idx]
            + energy_1 * inverse_temperatures[idx + 1]
            - energy_0 * inverse_temperatures[idx + 1]
        )
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
        swapped_chains_0_v = torch.where(
            swap,
            chains[idx + 1].visible,
            chains[idx].visible,
        )
        swapped_chains_1_v = torch.where(
            swap,
            chains[idx].visible,
            chains[idx + 1].visible,
        )
        swapped_chains_0_h = torch.where(
            swap,
            chains[idx + 1].hidden,
            chains[idx].hidden,
        )
        swapped_chains_1_h = torch.where(
            swap,
            chains[idx].hidden,
            chains[idx + 1].hidden,
        )
        chains[idx].visible = swapped_chains_0_v
        chains[idx + 1].visible = swapped_chains_1_v
        chains[idx].hidden = swapped_chains_0_h
        chains[idx + 1].hidden = swapped_chains_1_h

    return chains, acc_rate, index


def pt_sampling(
    it_mcmc: int,
    increment: int,
    target_acc_rate: float,
    num_chains: int,
    params: Tuple[Tensor, Tensor, Tensor],
    out_file,
):
    inverse_temperatures = find_inverse_temperatures(target_acc_rate, params)
    list_chains = []
    for i in range(inverse_temperatures.shape[0]):
        list_chains.append(init_chains(num_chains, params))

    # Annealing to initialize the chains
    index = []
    for i in range(inverse_temperatures.shape[0]):
        for j in range(i, inverse_temperatures.shape[0]):
            list_chains[j] = sample_state(
                gibbs_steps=increment,
                chains=list_chains[j],
                params=params,
                beta=inverse_temperatures[i],
            )
        index.append(
            torch.ones(list_chains[i].visible.shape[0], device=list_chains[i].device)
            * i
        )

    counts = 0
    while counts < it_mcmc:
        counts += increment
        # Iterate chains
        for i in range(len(list_chains)):
            list_chains[i] = sample_state(
                gibbs_steps=increment,
                chains=list_chains[i],
                params=params,
                beta=inverse_temperatures[i],
            )

        # Swap chains
        list_chains, acc_rate, index = swap_configurations(
            chains=list_chains,
            params=params,
            inverse_temperatures=inverse_temperatures,
            index=index,
        )
        with h5py.File(out_file, "a") as f:
            f[f"index_{counts}"] = torch.vstack(index).cpu().numpy()

    return list_chains, inverse_temperatures, index
