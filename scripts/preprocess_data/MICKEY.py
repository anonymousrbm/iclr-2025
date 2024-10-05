import h5py
import numpy as np

from torchrbm.dataset.definitions import ROOT_DIR_DATASET_PACKAGE


def preprocess_MICKEY(
    filename=ROOT_DIR_DATASET_PACKAGE / "data/mickey.npy",
    out_dir=ROOT_DIR_DATASET_PACKAGE / "data",
):
    dataset = np.load(filename).T
    with h5py.File(out_dir / "mickey.h5", "w") as f:
        f["samples"] = dataset


if __name__ == "__main__":
    preprocess_MICKEY()
