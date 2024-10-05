import collections
import time

import h5py
import numpy as np
import torch

from torchrbm.rcm.convert import rcm_to_rbm
from torchrbm.rcm.features import decimation
from torchrbm.rcm.features import eliminate_features
from torchrbm.rcm.features import gen_features_v2
from torchrbm.rcm.features import get_features
from torchrbm.rcm.features import get_random_features
from torchrbm.rcm.rbm import finetune_bias
from torchrbm.rcm.rbm import get_ll_rbm
from torchrbm.rcm.rbm import get_proba_rbm
from torchrbm.rcm.rbm import sample_potts_rcm
from torchrbm.rcm.rbm import sample_rbm
from torchrbm.rcm.solver import train_rcm


Trial = collections.namedtuple(
    "Trial",
    [
        "vbias_rbm",
        "hbias_rbm",
        "W_rbm",
        "features",
        "vbias_rcm",
        "q",
        "all_train_ll",
        "all_test_ll",
    ],
)


def save_trial(
    m,
    mu,
    configurational_entropy,
    U,
    vbias_rcm,
    q,
    features,
    vbias_rbm,
    hbias_rbm,
    W_rbm,
    all_train_ll,
    all_test_ll,
    header,
    args,
):
    device = m.device
    dtype = m.dtype
    pdm, log_z = get_proba_rbm(
        m=m,
        configurational_entropy=configurational_entropy,
        U=U,
        vbias=vbias_rbm,
        hbias=hbias_rbm,
        W=W_rbm,
        return_logZ=True,
    )
    if args["potts"]:
        samples_gen = sample_potts_rcm(
            p_m=pdm,
            mu=mu,
            U=U,
            num_samples=args["num_sample_gen"],
            num_colors=args["num_colors"],
            device=device,
            dtype=dtype,
        )
    else:
        samples_gen = sample_rbm(
            p_m=pdm,
            mu=mu,
            U=U,
            num_samples=args["num_sample_gen"],
            device=device,
            dtype=dtype,
        )
    with h5py.File(args["filename"], "a") as f:
        curr_exp = f.create_group(header)
        curr_exp["q"] = q.cpu().numpy()
        curr_exp["vbias_rcm"] = vbias_rcm.cpu().numpy()
        curr_exp["W_rbm"] = W_rbm.cpu().numpy()
        curr_exp["vbias_rbm"] = vbias_rbm.cpu().numpy()
        curr_exp["hbias_rbm"] = hbias_rbm.cpu().numpy()
        curr_exp["train_ll"] = all_train_ll
        curr_exp["test_ll"] = all_test_ll
        curr_exp["pdm"] = pdm.cpu().numpy()
        curr_exp["samples_gen"] = samples_gen.cpu().numpy()
        curr_exp["features"] = features.cpu().numpy()
        curr_exp["seed"] = args["seed"]
        curr_exp["log_z"] = log_z.item()


def train(
    train_dataset: np.ndarray,
    test_dataset: np.ndarray,
    weights_train: np.ndarray,
    weights_test: np.ndarray,
    args: dict,
    mesh_file: str,
) -> None:
    """Given an input dataset and arguments, train a RCM and save it to HDF5 archive.

    Parameters
    ----------
    dataset : np.ndarray
        Dataset. (num_samples, num_visibles)
    args : dict
        _description_
    device : torch.device
        _description_
    dtype : torch.dtype
        _description_
    """
    match args["dtype"]:
        case "float":
            dtype = torch.float32
        case "float32":
            dtype = torch.float32
        case "double":
            dtype = torch.float64
        case "float64":
            dtype = torch.float64
    device = torch.device(args["device"])

    # Seed the experiments
    if args["seed"] is not None:
        rng = np.random.default_rng(args["seed"])
        torch.manual_seed(args["seed"])
    else:
        rng = np.random.default_rng()

    num_samples, num_visibles = train_dataset.shape

    # Shuffle the dataset
    train_dataset = np.ascontiguousarray(rng.permutation(train_dataset, axis=0))
    test_dataset = np.ascontiguousarray(rng.permutation(test_dataset, axis=0))

    train_set = torch.from_numpy(train_dataset).to(device).to(dtype)
    test_set = torch.from_numpy(test_dataset).to(device).to(dtype)
    weights_train: torch.Tensor = (
        torch.from_numpy(weights_train).to(device).to(dtype) / weights_train.sum()
    )
    weights_test: torch.Tensor = (
        torch.from_numpy(weights_test).to(device).to(dtype) / weights_test.sum()
    )
    start = time.time()

    # Load pre-computed mesh
    with h5py.File(mesh_file, "r") as f:
        m = torch.from_numpy(np.array(f["m"])).to(device).to(dtype)
        mu = torch.from_numpy(np.array(f["mu"])).to(device).to(dtype)
        configurational_entropy = (
            torch.from_numpy(np.array(f["configurational_entropy"]))
            .to(device)
            .to(dtype)
        )
        U = torch.from_numpy(np.array(f["U"])).to(device).to(dtype)
        print(f.keys())
        args["dimension"] = np.asarray(f["hyperparameters"]["dimension"][()], dtype=int)
        args["with_bias"] = np.array(
            f["hyperparameters"]["with_bias"], dtype=bool
        ).item()

        args["potts"] = False
        # num_colors exists only in Potts meshes
        if "num_colors" in f["hyperparameters"].keys():
            args["potts"] = True
            args["num_colors"] = np.array(
                f["hyperparameters"]["num_colors"], dtype=int
            ).item()
        print(configurational_entropy)
        mask = torch.logical_not(configurational_entropy.isinf())
        m = m[mask]
        configurational_entropy = configurational_entropy[mask]
        mu = mu[mask]

    args["intrinsic_dimension"] = len(args["dimension"])

    # Project dataset
    proj_train = train_set @ U.T / num_visibles**0.5
    proj_test = test_set @ U.T / num_visibles**0.5

    # Features generation
    features = gen_features_v2(
        proj_train[:, int(args["with_bias"]) :], args["num_hidden"] // 2
    )

    features_2 = get_features(1000, proj_train.shape[1] - int(args["with_bias"]))

    features_2 = eliminate_features(
        torch.from_numpy(features_2).to(device).to(dtype),
        proj_train[:, int(args["with_bias"]) :],
        args["num_hidden"] // 2,
        device=device,
        dtype=dtype,
    )
    features_3 = (
        torch.from_numpy(
            get_random_features(
                args["num_hidden"],
                proj_train[:, (int(args["with_bias"])) :].cpu().numpy(),
            )
        )
        .to(device)
        .to(dtype)
    )
    features = torch.vstack([features, features_2, features_3])

    features = eliminate_features(
        features,
        proj_train[:, int(args["with_bias"]) :],
        args["num_hidden"],
        device=device,
        dtype=dtype,
    )

    if args["with_bias"]:
        features = torch.hstack(
            [
                torch.zeros(
                    features.shape[0], device=features.device, dtype=features.dtype
                ).unsqueeze(1),
                features,
            ]
        )

    scaled_lr = args["learning_rate"] / num_visibles
    total_iter = 0
    stop = False
    best_trial = None
    best_ll = -1e8
    num_features_removed = 0
    count_trials = 0

    # Log all constants and hyperparameters
    with h5py.File(args["filename"], "w") as f:
        const = f.create_group("const")
        const["m"] = m.cpu().numpy()
        const["mu"] = mu.cpu().numpy()
        const["U"] = U.cpu().numpy()
        const["configurational_entropy"] = configurational_entropy.cpu().numpy()
        const["features"] = features.cpu().numpy()

        hyperparameters = f.create_group("hyperparameters")
        hyperparameters["learning_rate"] = args["learning_rate"]
        hyperparameters["adapt"] = args["adapt"]
        hyperparameters["min_learning_rate"] = args["min_learning_rate"]
        hyperparameters["intrinsic_dimension"] = args["intrinsic_dimension"]
        hyperparameters["train_size"] = train_dataset.shape[0]
        hyperparameters["test_size"] = test_dataset.shape[0]
        # hyperparameters["num_points"] = args["num_points"]
        hyperparameters["stop_ll"] = args["stop_ll"]
        hyperparameters["smooth_rate"] = args["smooth_rate"]
        hyperparameters["eigen_threshold"] = args["eigen_threshold"]
        hyperparameters["features_threshold"] = args["feature_threshold"]
        hyperparameters["decimation"] = args["decimation"]
        hyperparameters["max_iter"] = args["max_iter"]
        hyperparameters["potts"] = args["potts"]
        if args["potts"]:
            hyperparameters["num_colors"] = args["num_colors"]
        while not stop:
            print("=" * 80 + "\n")
            vbias, q, all_train_ll, all_test_ll, total_iter = train_rcm(
                m=m,
                mu=mu,
                proj_train=proj_train,
                proj_test=proj_test,
                weights_train=weights_train,
                weights_test=weights_test,
                U=U,
                configurational_entropy=configurational_entropy,
                features=features,
                max_iter=args["max_iter"],
                adapt=args["adapt"],
                stop_ll=args["stop_ll"],
                min_learning_rate=args["min_learning_rate"],
                num_visibles=num_visibles,
                smooth_rate=args["smooth_rate"],
                learning_rate=scaled_lr,
                total_iter=total_iter,
                device=device,
                dtype=dtype,
                with_bias=args["with_bias"],
            )
            ## Recover current rbm:
            vbias_rbm, hbias_rbm, W_rbm = rcm_to_rbm(
                q=q, proj_vbias=vbias, features=features, U=U
            )

            ## Finetune RBM
            # vbias_rbm, _, _ = finetune_bias(
            #     proj_train=proj_train,
            #     proj_test=proj_test,
            #     weights_train=weights_train,
            #     m=m,
            #     mu=mu,
            #     configurational_entropy=configurational_entropy,
            #     U=U,
            #     vbias=vbias_rbm,
            #     hbias=hbias_rbm,
            #     W=W_rbm,
            #     learning_rate=args["learning_rate"],
            #     max_iter=args["max_iter"],
            #     adapt=args["adapt"],
            #     min_learning_rate=args["min_learning_rate"],
            #     stop_ll=args["stop_ll"],
            #     with_bias=args["with_bias"],
            # )

            curr_ll, log_z = get_ll_rbm(
                configurational_entropy,
                proj_train,
                m,
                W_rbm,
                hbias_rbm,
                vbias_rbm,
                U,
                num_visibles,
                return_logZ=True,
                with_bias=args["with_bias"],
            )
            if curr_ll > best_ll or best_trial is None:
                best_trial = Trial(
                    vbias_rbm=vbias_rbm.clone(),
                    hbias_rbm=hbias_rbm.clone(),
                    W_rbm=W_rbm.clone(),
                    features=features.clone(),
                    vbias_rcm=vbias.clone(),
                    q=q.clone(),
                    all_train_ll=np.array(all_train_ll),
                    all_test_ll=np.array(all_test_ll),
                )
                best_ll = curr_ll
            if args["save_all_trial"]:
                save_trial(
                    m=m,
                    mu=mu,
                    configurational_entropy=configurational_entropy,
                    U=U,
                    vbias_rcm=vbias,
                    q=q,
                    features=features,
                    vbias_rbm=vbias_rbm,
                    hbias_rbm=hbias_rbm,
                    W_rbm=W_rbm,
                    all_train_ll=np.array(all_train_ll),
                    all_test_ll=np.array(all_test_ll),
                    header=f"trial_{count_trials}",
                    args=args,
                )
            if args["decimation"]:
                features, q = decimation(
                    features=features,
                    q=q,
                    feature_threshold=args["feature_threshold"],
                    num_visibles=num_visibles,
                )
                print(f"New number of features: {q.shape[0]}")
                num_features_removed = args["num_hidden"] - q.shape[0]

                # Remove the less important features
                # To match the target number of hidden nodes
                if args["max_num_hiddens"] is not None:
                    if q.shape[0] > args["max_num_hiddens"]:
                        # Force save the best_trial to have the right number of hidden nodes
                        best_trial = None
                        if num_features_removed == 0:
                            num_features_to_remove = (
                                q.shape[0] - args["max_num_hiddens"]
                            )
                            index_to_keep = q.argsort()[num_features_to_remove:]
                            q = q[index_to_keep]
                            features = features[index_to_keep]

                            # To allow for another training
                            num_features_removed = args["num_hidden"] - q.shape[0]
                args["num_hidden"] = q.shape[0]
            if num_features_removed == 0:
                stop = True
            else:
                count_trials += 1
        save_trial(
            m=m,
            mu=mu,
            configurational_entropy=configurational_entropy,
            U=U,
            vbias_rcm=best_trial.vbias_rcm,
            q=best_trial.q,
            features=best_trial.features,
            vbias_rbm=best_trial.vbias_rbm,
            hbias_rbm=best_trial.hbias_rbm,
            W_rbm=best_trial.W_rbm,
            all_train_ll=best_trial.all_train_ll,
            all_test_ll=best_trial.all_test_ll,
            header="best_trial",
            args=args,
        )
    with h5py.File(args["filename"], "a") as f:
        f["num_trial"] = count_trials + 1
        f["time"] = time.time() - start
