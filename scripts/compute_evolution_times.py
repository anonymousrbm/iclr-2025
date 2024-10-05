import argparse

import h5py
import torch
from tqdm import tqdm

from torchrbm.bernoulli_bernoulli.io import load_params
from torchrbm.bernoulli_bernoulli.io import load_rcm
from torchrbm.bernoulli_bernoulli.ptt import init_sampling
from torchrbm.bernoulli_bernoulli.ptt import ptt_sampling
from torchrbm.bernoulli_bernoulli.ptt import sampling_step

from torchrbm.bernoulli_bernoulli.sampling import sample_state

from torchrbm.dataset import load_dataset
from torchrbm.dataset.parser import add_args_dataset
from torchrbm.parser import add_args_pytorch
from torchrbm.utils import get_binary_configurations
from torchrbm.utils import get_ll_updates
from torchrbm.utils import get_ptt_updates
from torchrbm.utils import get_saved_updates

Tensor = torch.Tensor

from scipy.optimize import curve_fit

def exponential_decay(t, C0, tauexp):
    return C0 * np.exp(-t/tauexp)

def compute_tau_int(C):
    
    t_max=len(C)
    t_int = 0.5
    for t in range(1, t_max):
        if t >= 6*t_int:
            break
        t_int += C[t]
    return t_int


def create_parser():
    parser = argparse.ArgumentParser("Compute PTT trajectory of the RBM at the last saved update.")
    parser.add_argument("-i", "--filename", type=str, help="RBM HDF5 archive.")

    parser.add_argument(
        "--num_chains",
        default=1000,
        type=int,
        help="(Defaults to 1000). Number of chains",
    )
    

    parser.add_argument(
        "--num_max_it_Gibbs",
        default=1_000_000,
        type=int,
        help="(Defaults to 1_000_000). Number of maximum steps.",
    )

    parser.add_argument(
        "--times_save",
        default=2_000,
        type=int,
        help="(Defaults to 1000). Number of time measures we save by model.",
    )

    parser.add_argument(
        "--steps_PTT",
        default=10000,
        type=int,
        help="(Defaults to 10000). Number of PTT steps for thermalisation",
    )
    

    parser.add_argument(
        "--use_ll_updates",
        action="store_true",
        default=False,
        help="(Defaults to False). Use the models in the ll_updates file",
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

    U_data, S_data, V_dataT = torch.linalg.svd(train_data  - train_data.mean(0))
    print(train_data.shape)
        

    # Sample the machine using PTT
    n_sample_ptt = args["num_chains"]
    it_mcmc_init_ptt = 100
    max_sample_Gibbs = args["num_max_it_Gibbs"]
    inter=2
    
    
    Nc=5 #number of modes to keep track

    rcm = load_rcm(filename, device, dtype)
    if args["use_ll_updates"]:
        ptt_updates = get_ll_updates(filename)
    else:
        ptt_updates = get_ptt_updates(filename)

    print("Analyzing",len(ptt_updates),"machines")

    list_params = []
    for upd in ptt_updates:
        list_params.append(load_params(filename, upd, device, dtype))

    chains = init_sampling(
                n_gen=n_sample_ptt,
                list_params=list_params,
                rcm=rcm,
                it_mcmc=it_mcmc_init_ptt,
                device=device,
                dtype=dtype,
            )
    
    increment_Gibbs =  torch.ones(len(list_params))

    
    with h5py.File(filename, "a") as f:
        #del f["eq_PTT_vis"]
        #del f["eq_PTT_hid"]

        if "eq_PTT_vis" in f:

            chains_vis=torch.tensor(f["eq_PTT_vis"][:])
            chains_hid=torch.tensor(f["eq_PTT_hid"][:])
            if chains_vis.shape[0]!=len(list_params):
                print("chains are not compatible with the number of parameters")
                return 0
            if chains_vis.shape[1]<n_sample_ptt:
                print("not enough chains saved")
                return 0
            
            for idx, params in enumerate(list_params):
                chains[idx].visible=chains_vis[idx].to(device)[:n_sample_ptt]
                chains[idx].hidden=chains_hid[idx].to(device)[:n_sample_ptt]

            
        else:
            
            #we run a pre-ptt to get a good initialisation where to start to measure
            print("Thermalisation with PTT")
            chains,acc_rates, _ = ptt_sampling(
                    rcm=None,
                    list_params=list_params,
                    chains=chains,
                    index=None,
                    it_mcmc=args["steps_PTT"],
                    increment=1,
                    show_acc_rate=False,
                    show_pbar=True,
                )
            

            chains_vis=[]
            chains_hid=[]
            for idx, params in enumerate(list_params):
                chains_hid.append(chains[idx].hidden.cpu())
                chains_vis.append(chains[idx].visible.cpu())
            f.create_dataset("eq_PTT_vis", data=chains_vis)
            f.create_dataset("eq_PTT_hid", data=chains_hid)
    
    pbar = tqdm(total=len(list_params))
    pbar.set_description("sampling with Gibbs sampling")
    
    n_times=args["times_save"]
    M=torch.zeros((len(list_params),n_times,Nc),device=device)
    MV=torch.zeros((len(list_params),n_times),device=device)
    MH=torch.zeros((len(list_params),n_times),device=device)

    for idx, params in enumerate(list_params):

        not_enough=True
        done=0
        mm=torch.zeros((n_times,Nc))
        mv,mh=torch.zeros((2,n_times))
        while not_enough:
            
            for t in range(done,n_times):
                chains[idx] = sample_state(chains=chains[idx], params=params, gibbs_steps=increment_Gibbs[idx])

                v=chains[idx].visible
                h=chains[idx].hidden
                
                chain_proj = v @ V_dataT.mT[:,0:Nc] / n_visibles**.5
                
                m=torch.mean(chain_proj,0)

                mm[t] =m

                mv[t] =v.mean()
                mh[t] =h.mean()

            C=torch.zeros(n_times)
            #x=mm[:,0]-mm[:,0].mean()
            x=mv[:]-mv[:].mean()
            C0=torch.mean(x**2)
            C[0]=1
            for t in range(1,n_times):
                C[t]=torch.mean(x[t:]*x[:-t])/C0
            
            tau=compute_tau_int(C)
            if 20*tau < n_times or (n_times*inter*increment_Gibbs[idx]>max_sample_Gibbs):
                print(idx,"completed",increment_Gibbs[idx],tau)
                not_enough=False
                M[idx] =mm
                MV[idx] =mv
                MH[idx] =mh
            else:
                #print(idx,tau,20*tau,increment_Gibbs[idx])
                #it was not enough so we increase times
                done=n_times//inter
                not_enough=True
                increment_Gibbs[idx]=increment_Gibbs[idx]*inter
                mm[:done]=mm[torch.arange(0,n_times,inter)]
                mv[:done]=mv[torch.arange(0,n_times,inter)]
                mh[:done]=mh[torch.arange(0,n_times,inter)]


        pbar.update(1)
    
    pbar.close()
    
    with h5py.File(filename, "a") as f:
        for k in ['projections_adapt_time_Gibbs_M','projections_adapt_time_Gibbs_MV','projections_adapt_time_Gibbs_MH','increment_adapt_Gibbs','list_adapt_upd']:
            if k in f.keys():
                del f[k]
        f.create_dataset('projections_adapt_time_Gibbs_M', data=M.cpu())
        f.create_dataset('projections_adapt_time_Gibbs_MV', data=MV.cpu())
        f.create_dataset('projections_adapt_time_Gibbs_MH', data=MH.cpu())
        f.create_dataset('list_adapt_upd', data=ptt_updates)
        f['increment_adapt_Gibbs'] =increment_Gibbs

    
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
