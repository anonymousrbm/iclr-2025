import argparse

import h5py
import numpy as np
import torch
from tqdm import tqdm

from torchrbm.bernoulli_bernoulli.init import init_chains
from torchrbm.bernoulli_bernoulli.io import load_params
from torchrbm.bernoulli_bernoulli.io import load_rcm
from torchrbm.bernoulli_bernoulli.ptt import init_sampling
from torchrbm.bernoulli_bernoulli.ptt import ptt_sampling
from torchrbm.bernoulli_bernoulli.sampling import sample_state
from torchrbm.classes import Chain
from torchrbm.dataset import load_dataset
from torchrbm.dataset.parser import add_args_dataset
from torchrbm.rcm.rbm import sample_rbm
from torchrbm.utils import check_file_existence
from torchrbm.utils import get_ptt_updates


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("PTT sampling on the provided model")
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
        "--it_mcmc",
        default=1000,
        type=int,
        help="(Defaults to 1000). Number of MCMC steps to perform.",
    )
    parser.add_argument(
        "--index",
        default=False,
        action="store_true",
        help="(Defaults to False). Save the index of the chains during sampling to measure mixing time.",
    )
    parser.add_argument(
        "--index_freq",
        default=1,
        type=int,
        help="(Defaults to 1). Frequency at which to save the index.",
    )
    parser.add_argument(
        "--trajectory",
        default=False,
        action="store_true",
        help="(Defaults to False). Save the trajectories during sampling.",
    )
    parser.add_argument(
        "--traj_freq",
        default=10,
        type=int,
        help="(Defaults to 10). Frequency at which to save the trajectories.",
    )
    parser = add_args_dataset(parser)
    return parser


def get_epochs_pt_sampling(
    filename: str,
    filename_rcm: str,
    updates: np.ndarray,
    target_acc_rate: float,
    device: torch.device,
    dtype: torch.dtype,
):
    use_rcm = filename_rcm is not None
    # updates = np.concatenate([np.ones(1, dtype=int), updates[900:]])
    # updates = [1, 10, 543, 568]
    selected_idx = [updates[0]]
    pbar = tqdm(enumerate(updates[1:], 1), total=len(updates[1:]))
    num_chains = 10000
    it_mcmc = 1000
    # Init first chain
    params = load_params(filename, updates[0], device=device, dtype=dtype)
    if use_rcm:
        with h5py.File(filename_rcm, "r") as f:
            tmp_name = "best_trial"
            U = torch.from_numpy(f["const"]["U"][()]).to(device)
            m = torch.from_numpy(f["const"]["m"][()]).to(device)
            mu = torch.from_numpy(f["const"]["mu"][()]).to(device)
            p_m = torch.from_numpy(f[tmp_name]["pdm"][()]).to(device)
        rcm = {"U": U, "m": m, "mu": mu, "p_m": p_m}
        new_chains = sample_rbm(
            p_m=p_m, mu=mu, U=U, num_samples=num_chains, device=device, dtype=dtype
        )
        new_chains = init_chains(num_chains, params, start_v=(new_chains + 1) / 2)
    else:
        new_chains = init_chains(num_samples=num_chains, params=params)
        new_chains = sample_state(chains=new_chains, params=params, gibbs_steps=it_mcmc)
        rcm = None
    # Copy of the first chain
    chains = [
        Chain(
            visible=new_chains.visible.clone(),
            mean_visible=None,
            hidden=None,
            mean_hidden=None,
        )
    ]
    list_params = [params.clone()]
    for i, idx in pbar:
        # Copy of the previous configuration to revert to it after
        # saved_chains = new_chains.visible.cpu().clone()
        # saved_prev = chains[-1].clone()
        # saved_old_chains = [chain.clone() for chain in chains]
        # torch.cuda.empty_cache()
        selected_idx.append(idx)
        with h5py.File(filename, "r") as f:
            params = load_params(
                filename=filename, index=idx, device=device, dtype=dtype
            )
        new_chains = sample_state(chains=new_chains, params=params, gibbs_steps=it_mcmc)

        # chains = [prev_chains, new_chains]
        chains.append(new_chains)
        list_params.append(params.clone())
        chains, acc_rate, _ = ptt_sampling(
            rcm=rcm,
            list_params=list_params,
            chains=chains,
            index=None,
            it_mcmc=1,
            increment=1,
            show_pbar=False,
        )
        # chains = saved_old_chains
        # chains.append(new_chains)
        # chains[-2] = saved_prev
        selected_idx.pop(-1)
        pbar.write(str(acc_rate[-1]))
        if acc_rate[-1] < target_acc_rate:
            selected_idx.append(updates[i - 1])
            pbar.write(str(selected_idx))
        else:
            # new_chains.visible = saved_chains.cuda()
            chains.pop(-1)
            list_params.pop(-1)
    if updates[-1] not in selected_idx:
        selected_idx.append(updates[-1])
    return np.array(selected_idx)


def main(filename, out_file, num_samples, it_mcmc, args):
    check_file_existence(out_file)

    device = "cuda"
    dtype = torch.float32

    # Load the RCM associated to the RBM
    rcm = load_rcm(filename, device, dtype)

    # Load the RBMs for PTT
    selected_idx = get_ptt_updates(filename)
    list_params = []
    for _, upd in enumerate(selected_idx):
        list_params.append(load_params(filename, upd, device, dtype))

    # Initialize PTT chains
    chains = init_sampling(
        rcm=rcm,
        list_params=list_params,
        n_gen=num_samples,
        it_mcmc=10000,
        device=device,
        dtype=dtype,
    )

    # Load the dataset to project the trajectories
    if args["trajectory"]:
        with h5py.File(out_file, "a") as f:
            f.create_group("trajectory")
        dataset, _ = load_dataset(
            dataset_name=args["data"],
            subset_labels=args["subset_labels"],
            variable_type=args["variable_type"],
            use_weights=args["use_weights"],
            alphabet=args["alphabet"],
            train_size=args["train_size"],
            test_size=args["test_size"],
            use_torch=True,
            device=device,
            dtype=dtype,
        )
        _, _, proj_mat = torch.linalg.svd(dataset.data - dataset.data.mean(0))
        proj_mat = proj_mat.T[:, :2]
        num_visibles = dataset.get_num_visibles()

    # Initialize index of the chains
    index = None
    if args["index"]:
        with h5py.File(out_file, "a") as f:
            f.create_group("index")
        index = []
        for idx, chain in enumerate(chains):
            index.append(torch.ones(chain.visible.shape[0], device=device) * idx)

    for it in tqdm(range(it_mcmc)):
        chains, _, index = ptt_sampling(
            rcm=rcm,
            list_params=list_params,
            chains=chains,
            index=index,
            it_mcmc=1,
            increment=1,
            show_pbar=False,
            show_acc_rate=False,
        )
        # Save index
        if args["index"]:
            if it % args["index_freq"] == 0:
                with h5py.File(out_file, "a") as f:
                    f["index"][f"it_{it}"] = torch.vstack(index).cpu().numpy()

        # Save trajectory
        if args["trajectory"]:
            if it % args["traj_freq"] == 0:
                curr_pos = chains[-1].visible @ proj_mat / num_visibles**0.5
                with h5py.File(out_file, "a") as f:
                    f["trajectory"][f"it_{it}"] = curr_pos.cpu().numpy()

    '''with h5py.File(out_file, "a") as f:
        f["selected_idx"] = selected_idx
        for i, idx in enumerate(selected_idx):
            f[f"gen_chains_{idx}"] = chains[i].visible.cpu().numpy()'''


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)
    main(
        filename=args["filename"],
        out_file=args["out_file"],
        num_samples=args["num_samples"],
        it_mcmc=args["it_mcmc"],
        args=args,
    )
