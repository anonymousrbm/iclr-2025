import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchrbm.bernoulli_bernoulli.init import init_chains
from torchrbm.bernoulli_bernoulli.io import load_ais_traj_params
from torchrbm.bernoulli_bernoulli.io import load_model
from torchrbm.bernoulli_bernoulli.io import load_rcm
from torchrbm.bernoulli_bernoulli.io import save_model
from torchrbm.bernoulli_bernoulli.methods import compute_log_likelihood
from torchrbm.bernoulli_bernoulli.methods import create_machine
from torchrbm.bernoulli_bernoulli.partition_function import (
    update_weights_ais,
)
from torchrbm.bernoulli_bernoulli.pcd import fit_batch_pcd
from torchrbm.bernoulli_bernoulli.ptt import swap_config_multi
from torchrbm.dataset.dataset_class import RBMDataset
from torchrbm.rcm.rbm import sample_rbm
from torchrbm.utils import get_saved_updates
from torchrbm.utils import get_ll_updates
from torchrbm.utils import get_ptt_updates
from torchrbm.bernoulli_bernoulli.io import load_params
from torchrbm.bernoulli_bernoulli.sampling import sample_state


def train(
    dataset: RBMDataset,
    test_dataset: RBMDataset,
    args: dict,
    dtype: torch.dtype,
    checkpoints: np.ndarray,
):
    filename = args["filename"]

    # Load the data
    dataloader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    # Allows to iterate indefinitely on the dataloader without
    # worrying on the epochs
    dataloader = cycle(dataloader)
    num_visibles = dataset.get_num_visibles()

    # Create a first archive with the initialized model
    if not (args["restore"]):
        create_machine(
            filename=filename,
            num_visibles=num_visibles,
            num_hiddens=args["num_hiddens"],
            num_chains=args["num_chains"],
            batch_size=args["batch_size"],
            gibbs_steps=args["gibbs_steps"],
            learning_rate=args["learning_rate"],
            log=args["log"],
            dataset=dataset,
            test_dataset=test_dataset,
            device=args["device"],
            dtype=dtype,
        )

    # Retrieve the the number of training updates already performed on the model
    updates = get_saved_updates(filename=args["filename"])
    num_updates = updates[-1]
    if args["num_updates"] <= num_updates:
        raise RuntimeError(
            f"The parameter /'num_updates/' ({args['num_updates']}) must be greater than the previous number of epochs ({num_updates})."
        )

    params, parallel_chains, elapsed_time, hyperparameters = load_model(
        args["filename"],
        num_updates,
        device=args["device"],
        dtype=args["dtype"],
        set_rng_state=True,
    )

    # LL AIS trajectory variables
    log_z_init, log_weights = load_ais_traj_params(args["filename"], num_updates)
    log_weights = torch.from_numpy(log_weights).to(device=args["device"], dtype=dtype)
    chains_ll = parallel_chains.clone()

    start_v = None

    rcm = load_rcm(args["filename"], device=args["device"])
    if rcm is not None:
        start_v = sample_rbm(
            rcm["p_m"], rcm["mu"], rcm["U"], 2000, args["device"], dtype
        )
        start_v = (start_v + 1) / 2
    chains_ll = init_chains(parallel_chains.visible.shape[0], params, start_v).to(dtype)
    prev_params = params.clone()

    # Hyperparameters
    for k, v in hyperparameters.items():
        args[k] = v
    learning_rate = args["learning_rate"]

    # Define the optimizer
    optimizer = SGD(params.parameters(), lr=learning_rate, maximize=True)
    start = time.time()

    # Open the log file if it exists
    log_filename = Path(args["filename"]).parent / Path(
        f"log-{Path(args['filename']).stem}.csv"
    )
    args["log"] = log_filename.exists()

    # Continue the training
    pbar = tqdm(
        initial=num_updates,
        total=args["num_updates"],
        colour="red",
        dynamic_ncols=True,
        ascii="-#",
    )
    pbar.set_description("Training RBM")

    # Initialize gradients for the parameters
    params.weight_matrix.grad = torch.zeros_like(params.weight_matrix)
    params.vbias.grad = torch.zeros_like(params.vbias)
    params.hbias.grad = torch.zeros_like(params.hbias)

    if args["restore"]:
        ll_updates = get_ll_updates(filename)
        ptt_updates = get_ptt_updates(filename)

        last_params_ll=load_params(filename, ll_updates[-2],device=args["device"],dtype=dtype)
        last_params_ptt =load_params(filename, ptt_updates[-2],device=args["device"], dtype=dtype)
        
        #print(ll_updates,ptt_updates)
        last_chains_ll= sample_state(
                    gibbs_steps=10000,
                    chains=parallel_chains,
                    params=last_params_ll,
                )
        last_chains_ptt= sample_state(
                    gibbs_steps=10000,
                    chains=parallel_chains,
                    params=last_params_ptt,
                )

    else:
        last_chains_ptt = parallel_chains.clone()
        last_chains_ll = parallel_chains.clone()

        last_params_ptt = params.clone()
        last_params_ll = params.clone()

    target_acc_rate_ptt = args["acc_ptt"]

    target_acc_rate_ll = args["acc_ll"]

    with torch.no_grad():
        for idx in range(num_updates + 1, args["num_updates"] + 1):
            batch = next(dataloader)
            if args["use_numpy"]:
                batch = (
                    batch["data"].to(device=args["device"], dtype=dtype),
                    batch["weights"].to(device=args["device"], dtype=dtype),
                )
            else:
                batch = (batch["data"], batch["weights"])
            optimizer.zero_grad(set_to_none=False)
            parallel_chains, logs = fit_batch_pcd(
                batch=batch,
                parallel_chains=parallel_chains,
                params=params,
                gibbs_steps=args["gibbs_steps"],
                beta=args["beta"],
            )
            optimizer.step()

            # Permute chains to avoid correlations
            last_chains_ptt.permute()
            last_chains_ll.permute()

            curr_chains = parallel_chains.clone()
            curr_params = params.clone()

            _, acc_rate_ll, _ = swap_config_multi(
                params=[last_params_ll, curr_params],
                chains=[last_chains_ll, curr_chains],
                index=None,
            )
            save_ll = acc_rate_ll < target_acc_rate_ll
            if save_ll:
                last_params_ll = params.clone()
                last_chains_ll = parallel_chains.clone()

            _, acc_rate_ptt, _ = swap_config_multi(
                params=[last_params_ptt, curr_params],
                chains=[last_chains_ptt, curr_chains],
                index=None,
            )
            save_ptt = acc_rate_ptt < target_acc_rate_ptt
            #print(acc_rate_ptt)
            if save_ptt:
                # assert False
                last_params_ptt = params.clone()
                last_chains_ptt = parallel_chains.clone()

            # Compute the train and test LL using trajectory AIS

            curr_params = params.clone()
            log_weights, chains_ll = update_weights_ais(
                prev_params=prev_params,
                curr_params=curr_params,
                chains=chains_ll,
                log_weights=log_weights,
            )
            log_z = (
                torch.logsumexp(log_weights, 0)
                - np.log(chains_ll.visible.shape[0])
                + log_z_init
            ).item()

            prev_params = curr_params.clone()

            train_ll = compute_log_likelihood(
                dataset.data,
                curr_params,
                log_z,
            )
            test_ll = compute_log_likelihood(
                test_dataset.data,
                curr_params,
                log_z,
            )


            
            pbar.set_postfix_str(f"train LL: {train_ll:.2f}, test LL: {test_ll:.2f}")
            if idx in checkpoints or save_ll or save_ptt:
                curr_time = time.time() - start
                save_model(
                    filename=args["filename"],
                    params=params,
                    chains=parallel_chains,
                    num_updates=idx,
                    log_z=log_z,
                    log_weights=log_weights,
                    train_ll=train_ll,
                    test_ll=test_ll,
                    time=curr_time + elapsed_time,
                    save_ll=save_ll.item(),
                    save_ptt=save_ptt.item(),
                )
            pbar.update(1)
