import torch
from torch.nn.functional import softmax

from torchrbm.bernoulli_bernoulli.classes import BBParams
from torchrbm.classes import Chain
from torchrbm.classes import DataState


@torch.jit.script
def compute_gradient(
    data: DataState, chains: Chain, params: BBParams, centered: bool = True
) -> None:
    """Compute the gradient for each of the parameters and attach it.

    Parameters
    ----------
    data : DataState
        Current batch
    chains : Chain
        Current state of the parallel chains
    params: BBParams
        Current parameters of the model
    centered : bool
        Compute the centered gradient, by default True
    """
    # Turn the logit_weights of the chains into normalized weights
    chain_weights = softmax(-chains.logit_weights, -1).unsqueeze(-1)

    # Averages over data and generated samples
    v_data_mean = (data.visible * data.weights).sum(0) / data.weights.sum()
    torch.clamp_(v_data_mean, min=1e-4, max=(1.0 - 1e-4))
    h_data_mean = (data.mean_hidden * data.weights).sum(0) / data.weights.sum()
    v_gen_mean = (chains.visible * chain_weights).sum(0)
    torch.clamp_(v_gen_mean, min=1e-4, max=(1.0 - 1e-4))
    h_gen_mean = (chains.hidden * chain_weights).sum(0)

    if centered:
        # Centered variables
        v_data_centered = data.visible - v_data_mean
        h_data_centered = data.mean_hidden - h_data_mean
        v_gen_centered = chains.visible - v_data_mean
        h_gen_centered = chains.hidden - h_data_mean

        # Gradient
        grad_weight_matrix = (
            (v_data_centered * data.weights).T @ h_data_centered
        ) / data.weights.sum() - ((v_gen_centered * chain_weights).T @ h_gen_centered)
        grad_vbias = v_data_mean - v_gen_mean - (grad_weight_matrix @ h_data_mean)
        grad_hbias = h_data_mean - h_gen_mean - (v_data_mean @ grad_weight_matrix)
    else:
        v_data_centered = data.visible
        h_data_centered = data.mean_hidden
        v_gen_centered = chains.visible
        h_gen_centered = chains.hidden

        # Gradient
        grad_weight_matrix = (
            (data.visible * data.weights).T @ data.mean_hidden
        ) / data.weights.sum() - ((chains.visible * chain_weights).T @ chains.hidden)
        grad_vbias = v_data_mean - v_gen_mean
        grad_hbias = h_data_mean - h_gen_mean
    # Attach to the parameters
    params.weight_matrix.grad.set_(grad_weight_matrix)
    params.vbias.grad.set_(grad_vbias)
    params.hbias.grad.set_(grad_hbias)
