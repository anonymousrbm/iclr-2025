import argparse
import h5py
import torch

from torchrbm.bernoulli_bernoulli.io import load_params
from torchrbm.bernoulli_bernoulli.pt import pt_sampling
from torchrbm.utils import get_saved_updates


def main(filename, out_file, num_samples, it_mcmc, target_acc_rate):
    device = "cuda"
    dtype = torch.float32

    age = get_saved_updates(filename)[-1]
    params = load_params(filename, age, device=device, dtype=dtype)

    with h5py.File(out_file, "w") as f:
        f["x"] = 1

    list_chains, inverse_temperatures, index = pt_sampling(
        it_mcmc=it_mcmc,
        increment=1,
        target_acc_rate=target_acc_rate,
        num_chains=num_samples,
        params=params,
        out_file=out_file,
    )

    for i in range(len(list_chains)):
        with h5py.File(out_file, "a") as f:
            f[f"gen_{i}"] = list_chains[i].visible.cpu().numpy()
    with h5py.File(out_file, "a") as f:
        f["sel_beta"] = inverse_temperatures


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PT sampling on the provided model")
    parser.add_argument("-i", "--filename", type=str, help="Model to use for sampling")
    parser.add_argument(
        "-o", "--out_file", type=str, help="Path to save the samples after generation"
    )
    parser.add_argument(
        "--num_samples",
        default=1000,
        type=int,
        help="(Defaults to 1000). Number of generated samples.",
    )
    parser.add_argument(
        "--target_acc_rate",
        default=0.3,
        type=float,
        help="(Defaults to 0.3). Target acceptance rate between two consecutive models.",
    )
    parser.add_argument(
        "--it_mcmc",
        default=1000,
        type=int,
        help="(Defaults to 1000). Number of MCMC steps to perform.",
    )
    args = parser.parse_args()
    main(
        filename=args.filename,
        out_file=args.out_file,
        num_samples=args.num_samples,
        it_mcmc=args.it_mcmc,
        target_acc_rate=args.target_acc_rate,
    )
