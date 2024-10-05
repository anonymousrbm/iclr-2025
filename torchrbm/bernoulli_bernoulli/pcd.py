from typing import Tuple

import torch

from torchrbm.bernoulli_bernoulli.classes import BBParams
from torchrbm.bernoulli_bernoulli.grad import compute_gradient
from torchrbm.bernoulli_bernoulli.init import init_data_state
from torchrbm.bernoulli_bernoulli.sampling import sample_state
from torchrbm.classes import Chain

Tensor = torch.Tensor


# @torch.jit.script
def fit_batch_pcd(
    batch: Tuple[Tensor, Tensor],
    parallel_chains: Chain,
    params: BBParams,
    gibbs_steps: int,
    beta: float,
) -> Tuple[Chain, dict]:
    """Sample the RBM and compute the gradient

    Parameters
    ----------
    batch : Tuple[Tensor, Tensor]
        Dataset samples, weights associated
    parallel_chains : Chain
        Parallel chains used for gradient computation
    params : BBParams
        Parameters of the RBM
    gibbs_steps : int
        Number of Gibbs steps to perform
    beta : float
        Inverse temperature

    Returns
    ----------
    Tuple[Chain, dict]
        Updated parallel chains, log dictionnary
    """
    v_data, w_data = batch
    curr_batch = init_data_state(
        data=v_data,
        weights=w_data,
        params=params,
    )
    # sample permanent chains
    parallel_chains = sample_state(
        gibbs_steps=gibbs_steps,
        chains=parallel_chains,
        params=params,
        beta=beta,
    )
    compute_gradient(
        data=curr_batch, chains=parallel_chains, params=params, centered=True
    )
    logs = {}
    return parallel_chains, logs
