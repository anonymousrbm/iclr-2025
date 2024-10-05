import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from torchrbm.bernoulli_bernoulli.classes import BBParams
from torchrbm.bernoulli_bernoulli.init import init_chains as init_chains_bernoulli
from torchrbm.bernoulli_bernoulli.methods import compute_log_likelihood
from torchrbm.bernoulli_bernoulli.partition_function import (
    compute_partition_function,
    compute_partition_function_ais,
)
from torchrbm.bernoulli_bernoulli.sampling import sample_state as sample_state_bernoulli
from torchrbm.dataset import load_dataset
from torchrbm.dataset.parser import add_args_dataset
from torchrbm.parser import add_args_pytorch
from torchrbm.rcm.rbm import sample_potts_rcm, sample_rbm
from torchrbm.utils import get_binary_configurations


def add_args_convert(parser: argparse.ArgumentParser):
    convert_args = parser.add_argument_group("Convert")
    convert_args.add_argument(
        "--path",
        "-i",
        type=Path,
        required=True,
        help="Path to the folder h5 archive of the RCM.",
    )
    convert_args.add_argument(
        "--output",
        "-o",
        type=Path,
        default="RBM.h5",
        help="(Defaults to RBM.h5). Path to the file where to save the model in RBM format.",
    )
    convert_args.add_argument(
        "--num_hiddens",
        type=int,
        default=50,
        help="(Defaults to 50). Target number of hidden nodes for the RBM.",
    )
    convert_args.add_argument(
        "--therm_steps",
        type=int,
        default=10000,
        help="(Defaults to 1e4). Number of steps to be performed to thermalize the chains.",
    )
    convert_args.add_argument(
        "--trial",
        type=int,
        default=None,
        help="(Defaults to the best trial). RCM trial to use",
    )

    rbm_args = parser.add_argument_group("RBM")
    rbm_args.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="(Defaults to 0.01). Learning rate.",
    )
    rbm_args.add_argument(
        "--gibbs_steps",
        type=int,
        default=20,
        help="(Defaults to 10). Number of Gibbs steps for each gradient estimation.",
    )
    rbm_args.add_argument(
        "--batch_size",
        type=int,
        default=2000,
        help="(Defaults to 1000). Minibatch size.",
    )
    rbm_args.add_argument(
        "--seed",
        type=int,
        default=945723295,
        help="(Defaults to 9457232957489). Seed for the experiments.",
    )
    rbm_args.add_argument(
        "--num_chains",
        default=2000,
        type=int,
        help="(Defaults to 2000). The number of permanent chains.",
    )
    jarjar_args = parser.add_argument_group("Jar-RBM")
    jarjar_args.add_argument(
        "--min_eps",
        type=float,
        default=0.7,
        help="(Defaults to 0.7). Minimum effective population size allowed.",
    )
    return parser


def create_parser():
    parser = argparse.ArgumentParser(
        description="Convert RCM into an RBM readable format."
    )
    parser = add_args_dataset(parser)
    parser = add_args_convert(parser)
    parser = add_args_pytorch(parser)
    return parser


def ising_to_bernoulli(params: BBParams) -> BBParams:
    params.vbias = 2.0 * (params.vbias - params.weight_matrix.sum(1))
    params.hbias = 2.0 * (-params.hbias - params.weight_matrix.sum(0))
    params.weight_matrix = 4.0 * params.weight_matrix
    return params


def convert(args: dict, device: torch.device, dtype: torch.dtype):
    train_dataset, test_dataset = load_dataset(
        dataset_name=args["data"],
        subset_labels=args["subset_labels"],
        variable_type=args["variable_type"],
        use_weights=args["use_weights"],
        alphabet=args["alphabet"],
        train_size=args["train_size"],
        test_size=args["test_size"],
        use_torch=not (args["use_numpy"]),
        device=args["device"],
        dtype=args["dtype"],
    )
    # Set the random seed
    if args["seed"] is not None:
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])

    if args["trial"] is None:
        trial_name = "best_trial"
    else:
        trial_name = f"trial_{args['trial']}"
    # Import parameters
    print(f"Trial selected: {trial_name}")
    with h5py.File(args["path"], "r") as f:
        vbias_rcm = (
            torch.from_numpy(np.array(f[trial_name]["vbias_rbm"])).to(device).to(dtype)
        )
        hbias_rcm = (
            torch.from_numpy(np.array(f[trial_name]["hbias_rbm"])).to(device).to(dtype)
        )
        weight_matrix_rcm = (
            torch.from_numpy(np.array(f[trial_name]["W_rbm"])).to(device).to(dtype)
        )
        parallel_chains_v = (
            torch.from_numpy(np.array(f[trial_name]["samples_gen"]))
            .to(device)
            .to(dtype)
        )
        p_m = torch.from_numpy(np.array(f[trial_name]["pdm"])).to(device).to(dtype)
        m = torch.from_numpy(np.array(f["const"]["m"])).to(device).to(dtype)
        mu = torch.from_numpy(np.array(f["const"]["mu"])).to(device).to(dtype)
        U = torch.from_numpy(np.array(f["const"]["U"])).to(device).to(dtype)
        if "time" in f.keys():
            total_time = np.array(f["time"]).item()
        else:
            total_time = 0
        potts = f["hyperparameters"]["potts"][()]
        if potts:
            num_colors = f["hyperparameters"]["num_colors"][()]
    start = time.time()
    params = BBParams(weight_matrix=weight_matrix_rcm, vbias=vbias_rcm, hbias=hbias_rcm)
    params = ising_to_bernoulli(params=params)
    num_visibles, num_hiddens_rcm = params.weight_matrix.shape

    num_hiddens_add = args["num_hiddens"] - num_hiddens_rcm
    if num_hiddens_add < 0:
        print("The target number of hidden nodes is lower than the RCMs one.")
        num_hiddens_add = 0
    print(f"Adding {num_hiddens_add} hidden nodes.")

    hbias_add = torch.zeros(size=(num_hiddens_add,), device=device)
    weight_matrix_add = (
        torch.randn(size=(num_visibles, num_hiddens_add), device=device) * 1e-4
    )
    params.hbias = torch.cat([params.hbias, hbias_add])
    params.weight_matrix = torch.cat([params.weight_matrix, weight_matrix_add], dim=1)
    num_hiddens = num_hiddens_rcm + num_hiddens_add

    parallel_chains_v = sample_rbm(p_m, mu, U, args["num_chains"], device, dtype)
    # Convert parallel chains into (0, 1) format
    parallel_chains_v = (parallel_chains_v + 1) / 2

    # Thermalize chains
    print("Thermalizing the parallel chains...")
    num_chains = len(parallel_chains_v)

    parallel_chains = init_chains_bernoulli(
        num_chains, params=params, start_v=parallel_chains_v
    )
    parallel_chains = sample_state_bernoulli(
        chains=parallel_chains,
        params=params,
        gibbs_steps=args["therm_steps"],
    )

    # Compute initial log partition function
    if min(params.weight_matrix.shape) <= 20:
        all_config = get_binary_configurations(
            min(params.weight_matrix.shape), device=device, dtype=dtype
        )
        log_z = compute_partition_function(params=params, all_config=all_config)
    else:
        log_z = compute_partition_function_ais(
            num_chains=1000, num_beta=5000, params=params
        )

    log_weights = torch.zeros(
        parallel_chains.visible.shape[0], device=device, dtype=dtype
    )
    train_ll = compute_log_likelihood(train_dataset.data, params, log_z)
    test_ll = compute_log_likelihood(test_dataset.data, params, log_z)

    # Generate output file
    with h5py.File(args["output"], "w") as f:
        hyperparameters = f.create_group("hyperparameters")
        hyperparameters["num_hiddens"] = num_hiddens
        hyperparameters["num_visibles"] = num_visibles
        hyperparameters["training_mode"] = "PCD"
        hyperparameters["batch_size"] = args["batch_size"]
        hyperparameters["gibbs_steps"] = args["gibbs_steps"]
        hyperparameters["min_eps"] = args["min_eps"]
        hyperparameters["epochs"] = 0
        hyperparameters["filename"] = str(args["output"])
        hyperparameters["learning_rate"] = args["learning_rate"]
        hyperparameters["beta"] = 1.0  # TODO: parser
        hyperparameters["variable_type"] = args["variable_type"]
        hyperparameters["dataset_name"] = args["data"]

        checkpoint = f.create_group("update_1")
        checkpoint["torch_rng_state"] = torch.get_rng_state()
        checkpoint["numpy_rng_arg0"] = np.random.get_state()[0]
        checkpoint["numpy_rng_arg1"] = np.random.get_state()[1]
        checkpoint["numpy_rng_arg2"] = np.random.get_state()[2]
        checkpoint["numpy_rng_arg3"] = np.random.get_state()[3]
        checkpoint["numpy_rng_arg4"] = np.random.get_state()[4]
        checkpoint["weight_matrix"] = params.weight_matrix.cpu().float().numpy()
        checkpoint["vbias"] = params.vbias.cpu().float().numpy()
        checkpoint["hbias"] = params.hbias.cpu().float().numpy()
        checkpoint["gradient_updates"] = 0
        checkpoint["free_energy"] = 0.0
        checkpoint["time"] = time.time() - start + total_time
        checkpoint["learning_rate"] = args["learning_rate"]
        checkpoint["save_ll"] = True
        checkpoint["save_ptt"] = True
        # LL AIS trajectory
        checkpoint["log_z"] = log_z
        checkpoint["log_weights"] = log_weights.cpu().numpy()
        checkpoint["train_ll"] = train_ll
        checkpoint["test_ll"] = test_ll

        rcm = f.create_group("rcm")
        rcm["U"] = U.cpu().numpy()
        rcm["mu"] = mu.cpu().numpy()
        rcm["m"] = m.cpu().numpy()
        rcm["pdm"] = p_m.cpu().numpy()

        # checkpoint["deviation"] = 0.01
        f["parallel_chains"] = parallel_chains.visible.cpu().float().numpy()
        f["list_idx_sample"] = [1]

    # Generate log output file
    # log_filename = args["output"].parent / Path(f"log-{args['output'].stem}.csv")
    # with open(log_filename, "w") as log_file:
    #     log_file.write("eps,steps,deviation,lr_vbias,lr_hbias,lr_weight_matrix\n")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args = vars(args)
    match args["dtype"]:
        case "int":
            args["dtype"] = torch.int64
        case "float":
            args["dtype"] = torch.float32
        case "double":
            args["dtype"] = torch.float64
    convert(args, args["device"], args["dtype"])
