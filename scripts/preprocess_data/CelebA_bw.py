import h5py
import numpy as np

from torchrbm.dataset.definitions import ROOT_DIR_DATASET_PACKAGE


def preprocess_CelebA_64_bw(
    filename=ROOT_DIR_DATASET_PACKAGE / "data/CelebA_HQ_64_bw.npy",
    binary_threshold=0.5,
    out_dir=ROOT_DIR_DATASET_PACKAGE / "data",
):
    dataset = np.load(filename)  # 30000, 64, 64
    dataset = dataset.reshape(30000, -1)
    dataset = (dataset > binary_threshold).astype("float")

    with h5py.File(out_dir / "CelebA_64_bw.h5", "w") as f:
        f["samples"] = dataset


def preprocess_CelebA_64_bw_cont(
    filename=ROOT_DIR_DATASET_PACKAGE / "data/CelebA_HQ_64_bw.npy",
    out_dir=ROOT_DIR_DATASET_PACKAGE / "data",
):
    dataset = np.load(filename)  # 30000, 64, 64
    dataset = dataset.reshape(30000, -1)

    with h5py.File(out_dir / "CelebA_64_bw_cont.h5", "w") as f:
        f["samples"] = dataset


def preprocess_CelebA_32_bw(
    filename=ROOT_DIR_DATASET_PACKAGE / "data/CelebA_HQ_32_bw.npy",
    binary_threshold=0.5,
    out_dir=ROOT_DIR_DATASET_PACKAGE / "data",
):
    dataset = np.load(filename)  # 30000, 32, 32
    dataset = dataset.reshape(30000, -1)
    dataset = (dataset > binary_threshold).astype("float")

    with h5py.File(out_dir / "CelebA_32_bw.h5", "w") as f:
        f["samples"] = dataset


if __name__ == "__main__":
    preprocess_CelebA_32_bw()
    preprocess_CelebA_64_bw()
    preprocess_CelebA_64_bw_cont()
