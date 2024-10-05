import torch

from torchrbm.bernoulli_bernoulli.classes import BBParams

Tensor = torch.Tensor


@torch.jit.script
def compute_energy(
    v: Tensor,
    h: Tensor,
    params: BBParams,
) -> Tensor:
    """Compute the Hamiltonian on the visible and hidden variables

    Parameters
    ----------
    v : Tensor
        Visible units
    h : Tensor
        Hidden units
    params : BBParams
        Parameters of the RBM

    Returns
    -------
    Tensor
        Energy of the data points
    """
    fields = torch.tensordot(params.vbias, v, dims=[[0], [1]]) + torch.tensordot(
        params.hbias, h, dims=[[0], [1]]
    )
    interaction = torch.multiply(
        v, torch.tensordot(h, params.weight_matrix, dims=[[1], [1]])
    ).sum(1)

    return -fields - interaction


@torch.jit.script
def compute_energy_visibles(v: Tensor, params: BBParams) -> Tensor:
    """Returns the energy of the model computed on the input data

    Parameters
    ----------
    v : Tensor
        Visible data
    params : BBParams
        Parameters of the RBM

    Returns
    -------
    Tensor
        Energy of the data points
    """
    field = v @ params.vbias
    exponent = params.hbias + (v @ params.weight_matrix)
    log_term = torch.where(
        exponent < 10, torch.log(1.0 + torch.exp(exponent)), exponent
    )
    return -field - log_term.sum(1)


@torch.jit.script
def compute_energy_hiddens(h: Tensor, params: BBParams):
    """Returns the energy of the model computed on hidden configurations

    Parameters
    ----------
    h : Tensor
        Hidden configuration
    params : BBParams
        Parameters of the RBM

    Returns
    -------
    Tensor
        Energy of the hidden configurations
    """
    field = h @ params.hbias
    exponent = params.vbias + (h @ params.weight_matrix.T)
    log_term = torch.where(
        exponent < 10, torch.log(1.0 + torch.exp(exponent)), exponent
    )
    return -field - log_term.sum(1)
