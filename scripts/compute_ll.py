import argparse

import h5py
import numpy as np
import torch
from tqdm import tqdm

from torchrbm.bernoulli_bernoulli.io import load_params
from torchrbm.bernoulli_bernoulli.io import load_rcm
from torchrbm.bernoulli_bernoulli.methods import compute_log_likelihood
from torchrbm.bernoulli_bernoulli.partition_function import compute_partition_function
from torchrbm.bernoulli_bernoulli.partition_function import (
    compute_partition_function_ais,
)
from torchrbm.bernoulli_bernoulli.partition_function import (
    compute_partition_function_ais_traj,
)
from torchrbm.dataset import load_dataset
from torchrbm.dataset.parser import add_args_dataset
from torchrbm.parser import add_args_pytorch
from torchrbm.rcm.rbm import sample_rbm
from torchrbm.utils import get_binary_configurations
from torchrbm.utils import get_ll_updates
from torchrbm.utils import get_saved_updates

Tensor = torch.Tensor


def create_parser():
    parser = argparse.ArgumentParser("Compute the LL of the RBM at every saved update.")
    parser.add_argument("-i", "--filename", type=str, help="RBM HDF5 archive.")
    parser.add_argument(
        "--exact",
        action="store_true",
        default=False,
        help="Compute the temperature AIS LL estimation (Nh <= 20 or Nv <= 20 required).",
    )
    parser.add_argument(
        "--ais",
        action="store_true",
        default=False,
        help="Compute the temperature AIS LL estimation.",
    )
    parser.add_argument(
        "--ais_ref",
        action="store_true",
        default=False,
        help="(Defaults to False). Change the reference distribution for Annealed Importance Sampling.",
    )
    parser.add_argument(
        "--repeat",
        default=1,
        type=int,
        help="(Defaults to 1). number of repeat AIS and PTT",
    )
    parser.add_argument(
        "--ais_traj",
        action="store_true",
        default=False,
        help="Compute the trajectory AIS LL estimation.",
    )
    parser = add_args_dataset(parser)
    parser = add_args_pytorch(parser)
    return parser


def main(filename, train_dataset, test_dataset, args, device, dtype):
    train_data = train_dataset.data
    test_data = test_dataset.data
    updates = get_saved_updates(filename)
    ll_updates = get_ll_updates(filename)
    params = load_params(filename, updates[0], device, dtype)
    n_visibles, n_hidden = params.vbias.shape[0], params.hbias.shape[0]
    if args["exact"]:
        if min(n_visibles, n_hidden) > 20:
            print(
                f"Does not compute the exact LL since Nv ({n_visibles}) > 20 and Nh ({n_hidden}) > 20"
            )
            args["exact"] = False
        else:
            all_config = get_binary_configurations(
                n_dim=min(n_visibles, n_hidden), device=device, dtype=dtype
            )
    if args["ais_traj"]:
        for idx_rep in range(args["repeat"]):
            num_chains_ais_traj = 2000
            ll_updates = get_ll_updates(filename)
            list_params = []
            for i, upd in enumerate(ll_updates):
                list_params.append(load_params(filename, upd, device, dtype))
            rcm = load_rcm(filename, device, dtype)
            start_v = None
            if rcm is not None:
                start_v = sample_rbm(
                    rcm["p_m"], rcm["mu"], rcm["U"], num_chains_ais_traj, device, dtype
                )
                start_v = (start_v + 1) / 2
            log_Z_ais_traj = compute_partition_function_ais_traj(
                num_chains=num_chains_ais_traj, list_params=list_params, start_v=start_v
            )
            ll_ais_traj_train = np.zeros(len(log_Z_ais_traj))
            ll_ais_traj_test = np.zeros(len(log_Z_ais_traj))
            for i, curr_log_z in enumerate(log_Z_ais_traj):
                ll_ais_traj_train[i] = compute_log_likelihood(
                    v_data=train_data, params=list_params[i], log_z=curr_log_z
                )
                ll_ais_traj_test[i] = compute_log_likelihood(
                    v_data=test_data, params=list_params[i], log_z=curr_log_z
                )
            for i, upd in enumerate(ll_updates):
                with h5py.File(filename, "a") as f:
                    for k in [
                        f"ll_ais_traj_train_{idx_rep}",
                        f"ll_ais_traj_test_{idx_rep}",
                        f"log_Z_ais_traj_{idx_rep}",
                    ]:
                        if k in f[f"update_{upd}"].keys():
                            del f[f"update_{upd}"][k]
                    f[f"update_{upd}"][
                        f"ll_ais_traj_train_{idx_rep}"
                    ] = ll_ais_traj_train[i]
                    f[f"update_{upd}"][
                        f"ll_ais_traj_test_{idx_rep}"
                    ] = ll_ais_traj_test[i]
                    f[f"update_{upd}"][f"log_Z_ais_traj_{idx_rep}"] = log_Z_ais_traj[i]
    for upd in tqdm(ll_updates, total=ll_updates.shape[0]):
        compute_ais = args["ais"]
        compute_exact = args["exact"]
        params = load_params(filename, index=upd, device=device, dtype=dtype)

        # Exact computation
        if compute_exact:
            curr_log_z = compute_partition_function(
                params=params, all_config=all_config
            )
            train_ll = compute_log_likelihood(
                v_data=train_data, params=params, log_z=curr_log_z
            )
            test_ll = compute_log_likelihood(
                v_data=test_data, params=params, log_z=curr_log_z
            )
            # all_logZ.append(curr_log_z)
            with h5py.File(filename, "a") as f:
                for k in ["ll_exact_train", "ll_exact_test", "log_Z_exact"]:
                    if k in f[f"update_{upd}"].keys():
                        del f[f"update_{upd}"][k]
                f[f"update_{upd}"]["ll_exact_train"] = train_ll
                f[f"update_{upd}"]["ll_exact_test"] = test_ll
                f[f"update_{upd}"]["log_Z_exact"] = curr_log_z

        # AIS trajectory
        if compute_ais:
            for idx_rep in range(args["repeat"]):
                num_sample_ais = 2_000
                num_beta_ais = 5_00
                if args["ais_ref"]:
                    eps = 1e-4
                    frequencies = train_dataset.data.mean(0)
                    if isinstance(frequencies, np.ndarray):
                        frequencies = torch.from_numpy(frequencies).to(device).to(dtype)
                    frequencies = torch.clamp(frequencies, min=eps, max=(1.0 - eps))
                    vbias_init = (
                        torch.log(frequencies) - torch.log(1.0 - frequencies)
                    ).to(device)
                else:
                    vbias_init = None
                curr_log_z = compute_partition_function_ais(
                    num_beta=num_beta_ais,
                    num_chains=num_sample_ais,
                    params=params,
                    vbias_ref=vbias_init,
                )
                train_ll = compute_log_likelihood(
                    v_data=train_data, params=params, log_z=curr_log_z
                )
                test_ll = compute_log_likelihood(
                    v_data=test_data, params=params, log_z=curr_log_z
                )
                with h5py.File(filename, "a") as f:
                    for k in [
                        f"ll_ais_train_{idx_rep}",
                        f"ll_ais_test_{idx_rep}",
                        f"log_Z_ais_{idx_rep}",
                    ]:
                        if k in f[f"update_{upd}"].keys():
                            del f[f"update_{upd}"][k]
                    f[f"update_{upd}"][f"ll_ais_train_{idx_rep}"] = train_ll
                    f[f"update_{upd}"][f"ll_ais_test_{idx_rep}"] = test_ll
                    f[f"update_{upd}"][f"log_Z_ais_{idx_rep}"] = curr_log_z


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)
    match args["dtype"]:
        case "float":
            dtype = torch.float32
        case "float32":
            dtype = torch.float32
        case "double":
            dtype = torch.float64
        case "float64":
            dtype = torch.float64
        case _:
            raise ValueError(f"dtype unrecognized: {args['dtype']}")
    device = torch.device(args["device"])
    train_dataset, test_dataset = load_dataset(
        dataset_name=args["data"],
        subset_labels=args["subset_labels"],
        variable_type=args["variable_type"],
        use_weights=args["use_weights"],
        alphabet=args["alphabet"],
        train_size=args["train_size"],
        test_size=args["test_size"],
        use_torch=True,
        dtype=dtype,
        device=args["device"],
    )
    main(
        filename=args["filename"],
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        args=args,
        device=args["device"],
        dtype=dtype,
    )
