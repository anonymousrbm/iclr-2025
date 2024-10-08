import numpy as np
import torch

from torchrbm.potts_bernoulli.tools import get_covariance_matrix


def get_ortho(mat: torch.Tensor):
    """Orthonormalize the column vectors of a matrix.

    Parameters
    ----------
    mat : torch.Tensor
        Matrix to orthonormalized. (a, b)

    Returns
    -------
    torch.Tensor
        Orthonormalized matrix. (a, b)
    """
    res = mat.clone()
    n, d = mat.shape

    u0 = mat[:, 0] / mat[:, 0].norm()
    res[:, 0] = u0
    for i in range(1, d):
        ui = mat[:, i]
        for j in range(i):
            ui -= (ui @ res[:, j]) * res[:, j]
        res[:, i] = ui / ui.norm()
    return res


def compute_U_old(
    M: torch.Tensor,
    weights: torch.Tensor,
    intrinsic_dimension: int,
    device: torch.device,
    dtype: torch.dtype,
    with_bias: bool = False,
) -> torch.Tensor:
    """Compute the first right eigenvector of the dataset.

    Parameters
    ----------
    M : torch.Tensor
        Dataset. (n_sample, n_visible)
    weights : torch.Tensor
        Weights of each sample (n_sample,)
    intrinsic_dimension : int
        Number of principal axis to compute.
    device : torch.device
        Device.
    dtype : torch.dtype
        Dtype

    Returns
    -------
    torch.Tensor
        Right eigenvectors. (n_dim, n_visible)
    """
    M = M * torch.sqrt(weights)
    mean_value = M.mean(0)
    if with_bias:
        M -= mean_value
    num_samples, num_visibles = M.shape
    max_iter = 100
    err_threshold = 1e-15
    print("Computing principal components...")
    curr_v = (
        torch.rand(num_samples, intrinsic_dimension, device=device, dtype=dtype) * 2 - 1
    )
    u = torch.rand(num_visibles, intrinsic_dimension, device=device, dtype=dtype)
    curr_id_mat = (
        torch.rand(intrinsic_dimension, intrinsic_dimension, device=device, dtype=dtype)
        * 2
        - 1
    )
    for n in range(max_iter):
        v = curr_v.clone()
        curr_v = M @ u
        if num_samples < num_visibles:
            id_mat = (v.T @ curr_v) / num_samples
            curr_v = get_ortho(curr_v)
        curr_u = M.T @ curr_v
        if num_visibles <= num_samples:
            id_mat = (u.T @ curr_u) / num_samples
            curr_u = get_ortho(curr_u)
        u = curr_u.clone()
        if (id_mat - curr_id_mat).norm() < err_threshold:
            break
        curr_id_mat = id_mat.clone()
    u = get_ortho(u)
    print("Done.")
    print(f"n_iter: {n}/{max_iter}; err: {(id_mat - curr_id_mat).norm()}")
    z = 0
    if with_bias:
        bias_vector = mean_value - u.T @ mean_value @ u.T
        z = bias_vector.norm()
        bias_vector /= z
        u = torch.hstack([bias_vector.unsqueeze(1), u])
    return u, z / np.sqrt(num_visibles)


def compute_U(
    M: torch.Tensor,
    weights: torch.Tensor,
    intrinsic_dimension: int,
    device: torch.device,
    dtype: torch.dtype,
    with_bias: bool = False,
):
    _, num_visibles = M.shape
    weights /= weights.sum()
    cov_data = get_covariance_matrix(M, weights, device=device, center=with_bias).to(
        device=device, dtype=dtype
    )

    _, V_dataT = torch.lobpcg(cov_data, k=intrinsic_dimension)
    u = V_dataT
    u = u[:, :intrinsic_dimension]

    mean_value = (M.T @ weights).squeeze()
    z = 0
    if with_bias:
        bias_vector = mean_value - u.T @ mean_value @ u.T
        z = bias_vector.norm()
        bias_vector /= z
        u = torch.hstack([bias_vector.unsqueeze(1), u])
    return u, z / np.sqrt(num_visibles)
