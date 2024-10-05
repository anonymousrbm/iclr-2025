import torch

from torchrbm.bernoulli_bernoulli.classes import BBParams
from torchrbm.classes import Chain


@torch.jit.script
def sample_hiddens(chains: Chain, params: BBParams, beta: float = 1.0) -> Chain:
    """Sample the hidden layer conditionally to the visible one

    Parameters
    ----------
    chains : Chain
        Current state of the Markov chain
    params : BBParams
        Parameters of the model
    beta : float, optional
        Inverse temperature, by default 1.0

    Returns
    -------
    Chain
        New state of the Markov chain
    """
    chains.mean_hidden = torch.sigmoid(
        beta * (params.hbias + (chains.visible @ params.weight_matrix))
    )
    chains.hidden = torch.bernoulli(chains.mean_hidden)
    return chains


@torch.jit.script
def sample_visibles(chains: Chain, params: BBParams, beta: float = 1.0) -> Chain:
    """Sample the visible layer conditionally to the hidden one

    Parameters
    ----------
    chains : Chain
        Current state of the Markov chain
    params : BBParams
        Parameters of the model
    beta : float, optional
        Inverse temperature, by default 1.0

    Returns
    -------
    Chain
        New state of the Markov chain
    """
    chains.mean_visible = torch.sigmoid(
        beta * (params.vbias + (chains.hidden @ params.weight_matrix.T))
    )
    chains.visible = torch.bernoulli(chains.mean_visible)
    return chains


@torch.jit.script
def sample_state(
    gibbs_steps: int, chains: Chain, params: BBParams, beta: float = 1.0
) -> Chain:
    """Update the state of the Markov chain accordingly to the parameters of the RBM

    Parameters
    ----------
    gibbs_steps : int
        Number of Gibbs steps
    chains : Chain
        Current state of the Markov chain
    params : BBParams
        Parameters of the model
    beta : float, optional
        Inverse temperature, by default 1.0

    Returns
    -------
    Chain
        New state of the Markov chain
    """
    for _ in range(gibbs_steps):
        chains = sample_hiddens(chains=chains, params=params, beta=beta)
        chains = sample_visibles(chains=chains, params=params, beta=beta)
    return chains
