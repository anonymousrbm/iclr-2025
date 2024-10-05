import argparse

import h5py
import torch
from tqdm import tqdm

from torchrbm.bernoulli_bernoulli.io import load_params
from torchrbm.bernoulli_bernoulli.io import load_rcm
from torchrbm.bernoulli_bernoulli.ptt import init_sampling
from torchrbm.bernoulli_bernoulli.ptt import ptt_sampling
from torchrbm.dataset import load_dataset
from torchrbm.dataset.parser import add_args_dataset
from torchrbm.parser import add_args_pytorch
from torchrbm.utils import get_binary_configurations
from torchrbm.utils import get_ll_updates
from torchrbm.utils import get_ptt_updates
from torchrbm.utils import get_saved_updates

Tensor = torch.Tensor


def create_parser():
    parser = argparse.ArgumentParser("Compute PTT trajectory of the RBM at the last saved update.")
    parser.add_argument("-i", "--filename", type=str, help="RBM HDF5 archive.")
    parser.add_argument(
        "--num_ptt_it",
        default=1_000,
        type=int,
        help="(Defaults to 50). Number of PT iterations.",
    )
    parser.add_argument(
        "--num_ptt_therm",
        default=10_000,
        type=int,
        help="(Defaults to 50). Number of PT iterations.",
    )

    parser.add_argument(
        "--use_ll_updates",
        action="store_true",
        default=False,
        help="(Defaults to False). Use the models in the ll_updates file",
    )
    parser.add_argument(
        "--optimize_updates",
        action="store_true",
        default=False,
        help="(Defaults to False). Option to reduce the number of updates for the PTT",
    )
    parser.add_argument(
        "--acc_optimize",
        type=float,
        default=0.25,
        help="(Defaults to 0.25). Optimal acceptance for the optimized PTT",
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


    # Sample the machine using PTT
    n_sample_ptt = 1_000
    it_mcmc_init_ptt = 100
    it_mcmc_sample_ptt = args["num_ptt_it"]
    increment_ptt = 1

    rcm = load_rcm(filename, device, dtype)

    if args["use_ll_updates"]:
        #ptt_updates = get_ll_updates(filename)
        ptt_updates =get_saved_updates(filename)
    else:
        ptt_updates = get_ptt_updates(filename)

    print("Analyzing",len(ptt_updates),"machines")
    list_params = []
    for upd in ptt_updates:
        list_params.append(load_params(filename, upd, device, dtype))
    
    chains = init_sampling(
        n_gen=n_sample_ptt,
        list_params=list_params,
        #rcm=rcm,
        it_mcmc=it_mcmc_init_ptt,
        device=device,
        dtype=dtype,
        #start_v=train_data,
    )

    #we run a pre-ptt to get a good initialisation where to start to measure
    print("Pre-sampling")
    chains,acc_rates, _ = ptt_sampling(
            rcm=rcm,
            list_params=list_params,
            chains=chains,
            index=None,
            it_mcmc= args["num_ptt_therm"],
            increment=increment_ptt,
            show_acc_rate=False,
            show_pbar=True,
        )
    print(acc_rates)

    if args["optimize_updates"]:
        opt='optimized'
        optimal_acc_rate=args["acc_optimize"]
        if min(acc_rates)<optimal_acc_rate:
            print("Run the code with more models because the minimal aceptance rate is:",min(acc_rates))
            return 0
        
        n_p=len(list_params)+1
        while(len(list_params)<n_p):
            n_p=len(list_params)
            i0=0
            list_params_new=list(list_params)
            chains_new=list(chains)
            while max(acc_rates[:-1])>optimal_acc_rate and i0<len(list_params_new)-1:
                
                chains,acc_rates, _ = ptt_sampling(
                    rcm=None,
                    list_params=list_params_new,
                    chains=chains_new,
                    index=None,
                    it_mcmc=1,
                    increment=increment_ptt,
                    show_acc_rate=False,
                    show_pbar=False,
                )
                sorted, indices=torch.sort(acc_rates,descending=True)
                
                i=indices[i0]
                #print(i,acc_rates[i])
                if i<len(indices)-1:
                    list_params_old=list(list_params_new)
                    chains_old=list(chains_new)

                    if i+1==len(list_params_new)-1:
                        print("There is something wrong!!")
                        return 0

                    list_params_new.pop(i+1)
                    chains_new.pop(i+1)

                    chains,acc_rates, _ = ptt_sampling(
                        rcm=None,
                        list_params=list_params_new,
                        chains=chains_new,
                        index=None,
                        it_mcmc=1,
                        increment=increment_ptt,
                        show_acc_rate=False,
                        show_pbar=False,
                    )
                    #print(acc_rates[i],acc_rates[i-1])
                    if acc_rates[i]<optimal_acc_rate or acc_rates[i-1]<optimal_acc_rate:
                        list_params_new=list(list_params_old)
                        chains_new=list(chains_old)
                        i0+=1
                else:
                    i0+=1
                
            print(len(list_params_new),"machines")
            

            list_params=list(list_params_new)

            chains=list(chains_new)
            chains,acc_rates, _ = ptt_sampling(
                    rcm=None,
                    list_params=list_params,
                    chains=chains,
                    index=None,
                    it_mcmc=1,
                    increment=increment_ptt,
                    show_acc_rate=False,
                    show_pbar=False,
                )
        print("Now we have only",len(list_params),"temperatures")
        print(acc_rates)
        
            


    
    proj_t=torch.zeros(it_mcmc_sample_ptt,2,n_sample_ptt)

    pbar = tqdm(total=it_mcmc_sample_ptt)
    pbar.set_description("sampling PTT updates")
    
    rates=torch.zeros(len(list_params)-1)


    for t in range(it_mcmc_sample_ptt):
        
        chains,acc_rates, indexes = ptt_sampling(
            rcm=rcm,
            list_params=list_params,
            chains=chains,
            index=None,
            it_mcmc=1,
            increment=increment_ptt,
            show_acc_rate=False,
            show_pbar=False,
        )
        
        rates+=acc_rates
        chain_proj = chains[-1].visible @ V_dataT.mT[:,0:2] / n_visibles**.5
        
        proj_t[t] = chain_proj.T.cpu()

        pbar.update(1)

    
    
    pbar.close()
    rates/=it_mcmc_sample_ptt
    print("mean_exchange_rate:",rates)
    print("#.updates",ptt_updates)
    
    with h5py.File(filename, "a") as f:
        for k in ['ptt_proj_t','ptt_num_models','ptt_ex_rate','conf_end']:
            if k in f.keys():
                del f[k]
        f.create_dataset('ptt_proj_t', data=proj_t)
        f.create_dataset('ptt_ex_rate', data=rates)
        f.create_dataset('conf_end', data=chains[-1].visible.cpu())
        f['ptt_num_models'] =len(list_params)

    
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
