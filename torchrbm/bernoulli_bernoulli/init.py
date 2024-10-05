from typing import Optional

import numpy as np
import torch

from torchrbm.bernoulli_bernoulli.classes import BBParams
from torchrbm.classes import Chain
from torchrbm.classes import DataState
from torchrbm.dataset.dataset_class import RBMDataset

Tensor = torch.Tensor


def init_parameters(
    num_visibles: int,
    num_hiddens: int,
    dataset: RBMDataset,
    device: torch.device,
    dtype: torch.dtype,
    var_init: float = 1e-4,
) -> BBParams:
    """Initialize the parameters of the RBM.
    Hidden biases are set to 0, visible biases are set to the frequencies of the dataset
    and the weight matrix with a gaussian distribution of variance var_init.

    Parameters
    ----------
    num_visibles: int
        Number of visibles units
    num_hiddens: int
        Number of hidden units
    dataset: RBMDataset
        Training dataset
    device: torch.device
        PyTorch device for the parameters
    dtype: torch.dtype
        PyTorch dtype for the parameters
    var_init: float, optional
        Variance of the weight matrix, by default 1e-4

    Returns
    ----------
    BBParams
        Initialized parameters
    """
    eps = 1e-4

    weight_matrix = (
        torch.randn(size=(num_visibles, num_hiddens), device=device, dtype=dtype)
        * var_init
    )
    frequencies = dataset.data.mean(0)
    if isinstance(frequencies, np.ndarray):
        frequencies = torch.from_numpy(frequencies).to(device=device, dtype=dtype)
    frequencies = torch.clamp(frequencies, min=eps, max=(1.0 - eps))
    vbias = (torch.log(frequencies) - torch.log(1.0 - frequencies)).to(device)
    hbias = torch.zeros(num_hiddens, device=device, dtype=dtype)
    return BBParams(weight_matrix=weight_matrix, vbias=vbias, hbias=hbias)


def init_chains(
    num_samples: int,
    params: BBParams,
    start_v: Optional[Tensor] = None,
) -> Chain:
    """Initialize a Markov chain for the RBM by sampling a uniform distribution on the visible layer
    and sampling the hidden layer according to the visible one

    Parameters
    ----------
    num_samples : int
        Number of parallel chains
    params : BBParams
        Parameters of the model
    start_v : torch.Tensor, optional
        Initial value for the chains, by default None

    Returns
    -------
    Chain
        Initialized Markov chain
    """
    num_visibles, _ = params.weight_matrix.shape
    mean_visible = (
        torch.ones(size=(num_samples, num_visibles), device=params.weight_matrix.device)
        / 2
    )
    if start_v is None:
        visible = torch.bernoulli(mean_visible)
    else:
        visible = start_v
    visible = visible.to(params.weight_matrix.dtype)
    mean_hidden = torch.sigmoid((params.hbias + (visible @ params.weight_matrix)))
    hidden = torch.bernoulli(mean_hidden)
    return Chain(
        visible=visible,
        hidden=hidden,
        mean_visible=mean_visible,
        mean_hidden=mean_hidden,
    )


@torch.jit.script
def init_data_state(data: Tensor, weights: Tensor, params: BBParams) -> DataState:
    """Sample the hidden layer of the RBM given the input data

    Parameters
    ----------
    data : torch.Tensor
        Input batch
    weights : torch.Tensor
        weights of the data samples
    params : BBParams
        Parameters of the RBM

    Returns
    -------
    DataState
        Initialized DataState
    """
    mean_hidden = torch.sigmoid((params.hbias + (data @ params.weight_matrix)))
    hidden = torch.bernoulli(mean_hidden)
    return DataState(
        visible=data,
        hidden=hidden,
        mean_hidden=mean_hidden,
        weights=weights,
    )
