import h5py
import numpy as np

from torchrbm.dataset.definitions import ROOT_DIR_DATASET_PACKAGE


def preprocess_HGD(
    filename=ROOT_DIR_DATASET_PACKAGE / "data/1kg_xtrain.d",
    out_dir=ROOT_DIR_DATASET_PACKAGE / "data",
):
    dataset = np.genfromtxt(filename).T
    with h5py.File(out_dir / "HGD.h5", "w") as f:
        f["samples"] = dataset


if __name__ == "__main__":
    preprocess_HGD()
