import gzip
import textwrap
from typing import Dict
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RBMDataset(Dataset):
    def __init__(
        self,
        data: np.ndarray,
        variable_type: str,
        labels: np.ndarray,
        weights: np.ndarray,
        names: np.ndarray,
        dataset_name: str,
        is_binary: bool,
        use_torch: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.data = data
        self.variable_type = variable_type
        self.labels = labels
        self.weights = weights[:, None]
        self.names = names
        self.dataset_name = dataset_name
        self.use_torch = use_torch
        self.device = device
        self.dtype = dtype
        self.is_binary = is_binary
        if use_torch:
            if variable_type == "Potts":
                self.dtype = torch.int32
            self.data = torch.from_numpy(self.data).to(self.device).to(self.dtype)
            self.weights = torch.from_numpy(self.weights).to(self.device).to(self.dtype)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        return {
            "data": self.data[index],
            "labels": self.labels[index],
            "weights": self.weights[index],
            "names": self.names[index],
        }

    def __str__(self) -> str:
        return textwrap.dedent(
            f"""
        Dataset: {self.dataset_name}
        Variable type: {self.variable_type}
        Number of samples: {self.data.shape[0]}
        Number of features: {self.data.shape[1]}
        """
        )

    def get_num_visibles(self) -> int:
        return self.data.shape[1]

    def get_num_states(self) -> int:
        return int(self.data.max() + 1)

    def get_effective_size(self) -> int:
        return int(self.weights.sum())

    def get_gzip_entropy(self, mean_size: int = 50, num_samples: int = 100):
        pbar = tqdm(range(mean_size))
        pbar.set_description("Compute entropy gzip")
        en = np.zeros(mean_size)
        for i in pbar:
            en[i] = len(
                gzip.compress(
                    (
                        self.data[torch.randperm(self.data.shape[0])[:num_samples]]
                    ).astype(int)
                )
            )
        return np.mean(en)
