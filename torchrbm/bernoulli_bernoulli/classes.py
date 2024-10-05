from typing import List
from typing import Optional

import torch

Tensor = torch.Tensor


@torch.jit.script
class BBParams:
    """Parameters of the Bernoulli-Bernoulli RBM"""

    def __init__(
        self,
        weight_matrix: Tensor,
        vbias: Tensor,
        hbias: Tensor,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if device is None:
            device = weight_matrix.device
        if dtype is None:
            dtype = weight_matrix.dtype
        self.weight_matrix = weight_matrix.to(device).to(dtype)
        self.vbias = vbias.to(device).to(dtype)
        self.hbias = hbias.to(device).to(dtype)

        self.device = device
        self.dtype = dtype

    @torch.jit.export
    def parameters(self) -> List[Tensor]:
        """Returns a list containing the parameters of the RBM

        Returns
        -------
        List[Tensor]
            weight_matrix, vbias, hbias
        """
        return [self.weight_matrix, self.vbias, self.hbias]

    @torch.jit.export
    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        if device is not None:
            self.weight_matrix = self.weight_matrix.to(device)
            self.vbias = self.vbias.to(device)
            self.hbias = self.hbias.to(device)
            self.device = device

        if dtype is not None:
            self.weight_matrix = self.weight_matrix.to(dtype)
            self.vbias = self.vbias.to(dtype)
            self.hbias = self.hbias.to(dtype)
            self.dtype = dtype
        return self

    @torch.jit.export
    def clone(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype
        return BBParams(
            weight_matrix=self.weight_matrix.clone(),
            vbias=self.vbias.clone(),
            hbias=self.hbias.clone(),
            device=device,
            dtype=dtype,
        )
