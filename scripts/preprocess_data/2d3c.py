import h5py
import numpy as np

from torchrbm.dataset.definitions import ROOT_DIR_DATASET_PACKAGE


def preprocess_2d3c(
    filename=ROOT_DIR_DATASET_PACKAGE / "data/data_2d3c_balanced_seed18_N1000.npy",
    out_dir=ROOT_DIR_DATASET_PACKAGE / "data",
):
    dataset = np.load(filename).T
    with h5py.File(out_dir / "2d3c.h5", "w") as f:
        f["samples"] = dataset


if __name__ == "__main__":
    preprocess_2d3c()
