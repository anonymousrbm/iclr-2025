from typing import Optional

import torch

Tensor = torch.Tensor


@torch.jit.script
class Chain:
    def __init__(
        self,
        visible: Tensor,
        hidden: Tensor,
        mean_visible: Tensor,
        mean_hidden: Tensor,
        logit_weights : Tensor=None,
    ) -> None:
        self.visible = visible
        self.hidden = hidden
        self.mean_visible = mean_visible
        self.mean_hidden = mean_hidden
        if logit_weights is not None:
            self.logit_weights = logit_weights
        else:
            self.logit_weights = torch.zeros(len(visible))

    @torch.jit.export
    def to(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        if device is not None:
            if self.visible is not None:
                self.visible = self.visible.to(device)
            if self.mean_visible is not None:
                self.mean_visible = self.mean_visible.to(device)
            if self.hidden is not None:
                self.hidden = self.hidden.to(device)
            if self.mean_hidden is not None:
                self.mean_hidden = self.mean_hidden.to(device)
            if self.logit_weights is not None:
                self.logit_weights = self.logit_weights.to(device)

        if dtype is not None:
            self.visible = self.visible.to(dtype)
            self.mean_visible = self.mean_visible.to(dtype)
            self.hidden = self.hidden.to(dtype)
            self.mean_hidden = self.mean_hidden.to(dtype)
            self.logit_weights = self.logit_weights.to(dtype)
        return self

    @torch.jit.export
    def clone(
        self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        if device is None:
            device = self.visible.device
        if dtype is None:
            dtype = self.visible.dtype
        new_vis = None
        new_hid = None
        new_mean_hid = None
        new_mean_vis = None
        new_logit_weights = None
        if self.visible is not None:
            new_vis = self.visible.clone().to(device=device, dtype=dtype)
        if self.hidden is not None:
            new_hid = self.hidden.clone().to(device=device, dtype=dtype)
        if self.mean_hidden is not None:
            new_mean_hid = self.mean_hidden.clone().to(device=device, dtype=dtype)
        if self.mean_visible is not None:
            new_mean_vis = self.mean_visible.clone().to(device=device, dtype=dtype)
        if self.logit_weights is not None:
            new_logit_weights = self.logit_weights.clone().to(device=device, dtype=dtype)
        return Chain(
            visible=new_vis,
            hidden=new_hid,
            mean_hidden=new_mean_hid,
            mean_visible=new_mean_vis,
            logit_weights=new_logit_weights,
        )

    @torch.jit.export
    def permute(self):
        new_index = torch.randperm(self.visible.shape[0])
        self.visible = self.visible[new_index]
        self.hidden = self.hidden[new_index]
        self.mean_hidden = self.mean_hidden[new_index]
        self.mean_visible = self.mean_visible[new_index]
        self.logit_weights = self.logit_weights[new_index]


@torch.jit.script
class DataState:
    def __init__(
        self,
        visible: Tensor,
        hidden: Tensor,
        mean_hidden: Tensor,
        weights: Tensor,
    ) -> None:
        self.visible = visible
        self.hidden = hidden
        self.mean_hidden = mean_hidden
        self.weights = weights
