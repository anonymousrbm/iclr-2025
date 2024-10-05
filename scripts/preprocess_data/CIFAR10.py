import h5py
import numpy as np

from torchrbm.dataset.definitions import ROOT_DIR_DATASET_PACKAGE


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def unpackbits(x, num_bits):
    if np.issubdtype(x.dtype, np.floating):
        raise ValueError("numpy data type needs to be int-like")
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2 ** np.arange(num_bits, dtype=x.dtype).reshape([1, num_bits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [num_bits])


def load_CIFAR10(
    filename=ROOT_DIR_DATASET_PACKAGE / "data/cifar-10-batches-py/data_batch_1",
    binary_threshold=0.3,
    out_dir=ROOT_DIR_DATASET_PACKAGE / "data",
):
    dataset_dict = unpickle(filename)
    dataset = np.array(dataset_dict[b"data"])
    labels = np.array(dataset_dict[b"labels"])

    # dataset is with variable between 0 and 255
    # cast it to [0,1]
    dataset /= 255.0

    dataset = (dataset > binary_threshold).astype("float")
    # dataset = (
    #     np.unpackbits(dataset)
    #     .reshape(dataset.shape[0], dataset.shape[1] * 8)
    #     .astype("float")
    # )

    with h5py.File(out_dir / "CIFAR10.h5", "w") as f:
        f["samples"] = dataset
        f["labels"] = labels
