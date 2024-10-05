import torch


Tensor = torch.Tensor


@torch.jit.script
def one_hot(x: Tensor, num_classes: int = -1, dtype: torch.dtype = torch.float32):
    """A one-hot encoding function faster than the PyTorch one working with torch.int32 and returning a float Tensor"""
    if num_classes < 0:
        num_classes = x.max() + 1
    res = torch.zeros(x.shape[0], x.shape[1], num_classes, device=x.device, dtype=dtype)
    tmp = torch.meshgrid(
        torch.arange(x.shape[0], device=x.device),
        torch.arange(x.shape[1], device=x.device),
        indexing="ij",
    )
    index = (tmp[0], tmp[1], x)
    values = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=dtype)
    res.index_put_(index, values)
    return res


def unravel_index(indices: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Converts a tensor of flat indices into a tensor of coordinate vectors.
    This is a `torch` implementation of `numpy.unravel_index`.

    Parameters
    ----------
    indices : torch.Tensor
        Tensor of flat indices. (*,)
    shape : torch.Size
        Target shape.

    Returns
    -------
    torch.Tensor
        The unraveled coordinates, (*, D).

    Notes
    -------
    See: https://github.com/pytorch/pytorch/issues/35674
    """

    shape = indices.new_tensor((*shape, 1))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return torch.div(indices[..., None], coefs, rounding_mode="trunc") % shape[:-1]


def log2cosh(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable version of log(2*cosh(x)).

    Parameters
    ----------
    x : torch.Tensor

    Returns
    -------
    torch.Tensor
        log(2*cosh(x))
    """
    return torch.abs(x) + torch.log1p(torch.exp(-2 * torch.abs(x)))
