from typing import Tuple

import torch
from torch.nn.functional import softmax

from torchrbm.bernoulli_bernoulli.classes import BBParams
from torchrbm.classes import Chain
from torchrbm.bernoulli_bernoulli.energy import compute_energy
from torchrbm.bernoulli_bernoulli.grad import compute_gradient
from torchrbm.bernoulli_bernoulli.init import init_data_state
from torchrbm.bernoulli_bernoulli.sampling import sample_state


def transition_kernel_log(
    chains : Chain,
    chains_prime : Chain,
    params : BBParams,
    ) -> torch.Tensor:
    """Calculate the log of the transition kernel probability P((v, h) -> (v', h')).
    
    Args:
        chains (Chain): Previous state of the model.
        chains_prime (Chain): New state of the model.
        params (BBParams): Parameters of the model.
        
    Returns:
        torch.Tensor: The log of the transition kernel probability P((v, h) -> (v', h')) for each sample in the batch.
    """
    
    # Compute probabilities of h_prime given v
    p_h_prime_given_v = torch.sigmoid(params.hbias + (chains.visible @ params.weight_matrix))
    
    # Compute probabilities of v_prime given h_prime
    p_v_prime_given_h_prime = torch.sigmoid(params.vbias + (chains_prime.hidden @ params.weight_matrix.T))
    
    # Log probability of h_prime given v
    log_prob_h_prime_given_v = torch.sum(
        chains_prime.hidden * torch.log(p_h_prime_given_v + 1e-10) +
            (1 - chains_prime.hidden) * torch.log(1 - p_h_prime_given_v + 1e-10),
        dim=1
    )
    
    # Log probability of v_prime given h_prime
    log_prob_v_prime_given_h_prime = torch.sum(
        chains_prime.visible * torch.log(p_v_prime_given_h_prime + 1e-10) +
            (1 - chains_prime.visible) * torch.log(1 - p_v_prime_given_h_prime + 1e-10),
        dim=1
    )
    
    # Combine the logs of the probabilities
    log_transition_prob = log_prob_h_prime_given_v + log_prob_v_prime_given_h_prime
    
    return log_transition_prob


def log_update(
    chains_old : Chain,
    chains_new : Chain,
    params_old : BBParams,
    params_new : BBParams
) -> torch.Tensor:
    """Computes the log-update for the weights of the chains.
    
    Args:
        chains_old (Chain): Previous state of the model.
        chains_new (Chain): New state of the model.
        params_old (BBParams): Previous parameters of the model.
        params_new (BBParams): New parameters of the model.
        
    Returns:
        torch.Tensor: log-update of the chain weights.
    
    """
    log_update = (
        compute_energy(chains_new.visible, chains_new.hidden, params_new) -
        compute_energy(chains_new.visible, chains_new.hidden, params_old)
    )

    return log_update
    

def compute_ess(logit_weights: torch.Tensor) -> torch.Tensor:
    """Computes the Effective Sample Size of the chains.
    
    Args:
        logit_weights: minus log-weights of the chains.
    """
    lwc = logit_weights - logit_weights.min()
    numerator = torch.square(torch.mean(torch.exp(-lwc)))
    denominator = torch.mean(torch.exp(-2.0 * lwc))
    
    return numerator / denominator


@torch.jit.script
def systematic_resampling(
    chains: Chain
) -> Chain:
    """Performs the systematic resampling of the chains according to their relative weight and
    sets the logit_weights back to zero.

    Args:
        chains (Chain): Chains.

    Returns:
        Chain: Resampled chains.
    """
    num_chains = chains.visible.shape[0]
    device = chains.visible.device
    weights = softmax(-chains.logit_weights, -1)
    weights_span = torch.cumsum(weights.double(), dim=0).float()
    rand_unif = torch.rand(size=(1,), device=device)
    arrow_span = (torch.arange(num_chains, device=device) + rand_unif) / num_chains
    mask = (weights_span.reshape(num_chains, 1) >= arrow_span).sum(1)
    counts = torch.diff(mask, prepend=torch.tensor([0], device=device))
    chains.visible = torch.repeat_interleave(chains.visible, counts, dim=0)
    chains.hidden = torch.repeat_interleave(chains.hidden, counts, dim=0)
    chains.logit_weights = torch.zeros_like(chains.logit_weights)

    return chains

def multinomial_resampling(
        chains: Chain
) -> Chain:
    """Performs the systematic resampling of the parallel chains according to their relative weight.

    Args:
        parallel_chains (Tuple[Tensor, Tensor]): (v, h) parallel chains.
        logit_weights (Tensor): Unnormalized log probability of the parallel chains.
        device (torch.device): Device.

    Returns:
        Tuple[Tensor, Tensor]: Resampled parallel chains.
    """
    num_chains = chains.visible.shape[0]
    device = chains.visible.device
    weights = softmax(-chains.logit_weights, -1)
    

    bootstrap_idxs = weights.multinomial(num_samples=num_chains, replacement=True)
    chains.visible=chains.visible[bootstrap_idxs]
    chains.hidden=chains.hidden[bootstrap_idxs]
    chains.logit_weights = torch.zeros_like(chains.logit_weights)

    return chains