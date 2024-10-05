from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch

from torchrbm.bernoulli_bernoulli.classes import BBParams
from torchrbm.bernoulli_bernoulli.energy import compute_energy_hiddens
from torchrbm.bernoulli_bernoulli.energy import compute_energy_visibles
from torchrbm.bernoulli_bernoulli.init import init_chains
from torchrbm.bernoulli_bernoulli.sampling import sample_state
from torchrbm.classes import Chain
from torchrbm.utils import get_binary_configurations

Tensor = torch.Tensor


def compute_partition_function(params: BBParams, all_config: Tensor) -> float:
    """Compute the exact log partition function.

    Parameters
    ----------
    params : BBParams
        Parameters of the RBM
    all_config : Tensor
        Tensor containing the enumeration of all possible states of one of the layers

    Returns
    ----------
    float
        Exact computation of the log partition function
    """
    n_dim_config = all_config.shape[1]
    n_visible, n_hidden = params.vbias.shape[0], params.hbias.shape[0]
    if n_dim_config == n_hidden:
        energy = compute_energy_hiddens(all_config, params)
    elif n_dim_config == n_visible:
        energy = compute_energy_visibles(all_config, params)
    else:
        raise ValueError(
            f"The number of dimension for the configurations '{n_dim_config}' does not match the number of visibles '{n_visible}' or the number of hidden '{n_hidden}'"
        )
    return torch.logsumexp(-energy, 0).item()


def compute_partition_function_ais(
    num_chains: int, num_beta: int, params: BBParams, vbias_ref: Optional[Tensor] = None
) -> float:
    """Compute the log partition function using Annealed Importance Sampling with temperature

    Parameters
    ----------
    num_chains : int
        Number of parallel chains for the estimation
    num_beta : int
        Number of inverse temperatures used during annealing
    params : BBParams
        Parameters of the RBM
    vbias_ref : Tensor, None
        Optional vbias specification to anneal from an independant model instead of the zero weight model

    Returns
    ----------
    float
        Log partition function
    """
    num_visibles, num_hiddens = params.weight_matrix.shape
    device = params.weight_matrix.device

    all_betas = np.linspace(start=0, stop=1, num=num_beta)

    # Compute the reference log partition function
    ## Here the case where all the weights are 0

    if vbias_ref is not None:
        vbias_init = vbias_ref.clone()
        log_z_init = num_hiddens * np.log(2) + torch.log1p(torch.exp(vbias_init)).sum()
    else:
        vbias_init = torch.zeros_like(params.vbias)
        log_z_init = (num_visibles + num_hiddens) * np.log(2)
    hbias_init = torch.zeros_like(params.hbias)
    params_ref = BBParams(
        weight_matrix=torch.zeros_like(params.weight_matrix),
        vbias=vbias_init,
        hbias=hbias_init,
    )
    # initialize the chains
    chains = init_chains(num_chains, params_ref)

    log_weights = torch.zeros(num_chains, device=device)
    # interpolate between true distribution and ref distribution
    for i, beta in enumerate(all_betas[:-1]):
        prev_params = params.clone()
        curr_params = params.clone()

        prev_params.weight_matrix = (
            beta * params.weight_matrix + (1 - beta) * params_ref.weight_matrix
        )
        prev_params.vbias = beta * params.vbias + (1 - beta) * params_ref.vbias
        prev_params.hbias = beta * params.hbias + (1 - beta) * params_ref.hbias

        curr_params.weight_matrix = (
            all_betas[i + 1] * params.weight_matrix
            + (1 - all_betas[i + 1]) * params_ref.weight_matrix
        )
        curr_params.vbias = (
            all_betas[i + 1] * params.vbias + (1 - all_betas[i + 1]) * params_ref.vbias
        )
        curr_params.hbias = (
            all_betas[i + 1] * params.hbias + (1 - all_betas[i + 1]) * params_ref.hbias
        )
        log_weights, chains = update_weights_ais(
            prev_params=prev_params,
            curr_params=curr_params,
            chains=chains,
            log_weights=log_weights,
        )
    log_z = torch.logsumexp(log_weights, 0) - np.log(num_chains) + log_z_init
    return log_z.item()


@torch.jit.script
def update_weights_ais(
    prev_params: BBParams, curr_params: BBParams, chains: Chain, log_weights: Tensor
) -> Tuple[Tensor, Chain]:
    """Update the weights used during Annealed Importance Sampling

    Parameters
    ----------
    prev_params : BBParams
        Parameters at time t-1
    curr_params : BBParams
        Parameters at time t
    chains : Chain
        Parallel chains at time t-1
    log_weights :
        Log weights at time t-1

    Returns
    ----------
    Tuple[Tensor, Chain]
        Log weights at time t, parallel chains at time t
    """
    chains = sample_state(gibbs_steps=1, chains=chains, params=curr_params)
    energy_prev = compute_energy_visibles(chains.visible, prev_params)
    energy_curr = compute_energy_visibles(chains.visible, curr_params)
    log_weights += -energy_curr + energy_prev
    return log_weights, chains


def compute_partition_function_ais_traj(
    num_chains: int, list_params: List[BBParams], start_v: Optional[Tensor] = None
) -> List[float]:
    """Compute the log partition function using Annealed Importance Sampling along the training trajectory

    Parameters
    ----------
    num_chains : int
        Number of parallel chains
    list_params : List[BBParams]
        Ordered list of the parameters of the RBM during training
    start_v : Tensor, None
        Optional starting point of the chains for the annealing

    Returns
    ----------
    List[float]
        Ordered list of all the log partition function of list_params
    """
    n_models = len(list_params)
    device = list_params[0].weight_matrix.device
    dtype = list_params[0].weight_matrix.dtype
    init_params = list_params[0]

    # Compute the first logZ0 exactly if it is feasible otherwise approximate it using AIS
    if min(init_params.weight_matrix.shape) <= 20:
        all_config = get_binary_configurations(
            n_dim=min(init_params.weight_matrix.shape), device=device, dtype=dtype
        )
        logZ0 = compute_partition_function(init_params, all_config)
    else:
        logZ0 = compute_partition_function_ais(
            num_chains=5000,
            num_beta=1000,
            params=init_params,
            device=device,
        )

    chains = init_chains(num_chains, list_params[0], start_v=start_v)

    logZ = torch.zeros(n_models - 1, device=device)
    all_logZ = [logZ0]
    log_weights = torch.zeros(num_chains, device=device)
    for i, params in enumerate(list_params[:-1]):
        log_weights, chains = update_weights_ais(
            prev_params=params,
            curr_params=list_params[i + 1],
            chains=chains,
            log_weights=log_weights,
        )
        logZ = torch.logsumexp(log_weights, 0) - np.log(num_chains) + logZ0
        all_logZ.append(logZ.item())
    return all_logZ
