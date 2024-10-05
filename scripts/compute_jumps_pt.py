import argparse

import h5py
import torch
from tqdm import tqdm


from torchrbm.bernoulli_bernoulli.classes import BBParams
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

#from torchrbm.bernoulli_bernoulli.pt import find_inverse_temperatures

Tensor = torch.Tensor


def create_parser():
    parser = argparse.ArgumentParser("Compute PT trajectory of the RBM at the last saved update.")
    parser.add_argument("-i", "--filename", type=str, help="RBM HDF5 archive.")
    
    parser.add_argument(
        "--num_pt_it",
        default=1_000,
        type=int,
        help="(Defaults to 50). Number of PT iterations.",
    )
    parser.add_argument(
        "--num_beta",
        default=50,
        type=int,
        help="(Defaults to 50). Number of temperatures used for the PT.",
    )
    parser.add_argument(
        "--ais_ref",
        action="store_true",
        default=False,
        help="(Defaults to False). Change the reference distribution for Annealed Importance Sampling.",
    )
    parser.add_argument(
        "--optimize_betas",
        action="store_true",
        default=False,
        help="(Defaults to False). Option to reduce the number of temperatures for the PT",
    )


    parser = add_args_dataset(parser)
    parser = add_args_pytorch(parser)
    return parser


def main(filename, train_dataset, test_dataset, args, device, dtype):

    opt=''
    train_data = train_dataset.data
    test_data = test_dataset.data
    updates = get_saved_updates(filename)
    ll_updates = get_ll_updates(filename)
    params = load_params(filename, updates[0], device, dtype)
    n_visibles, n_hidden = params.vbias.shape[0], params.hbias.shape[0]

    U_data, S_data, V_dataT = torch.linalg.svd(train_data  - train_data.mean(0))

    num_beta=args["num_beta"]
    if num_beta==1:
        all_betas=torch.tensor([1])
    else:
        x = torch.linspace(start=0, end=1, steps=num_beta)
        all_betas=x 

    

    # Sample the machine using PT

    n_sample_ptt = 1_000
    it_mcmc_init_ptt = 10_000
    it_mcmc_sample_ptt = args["num_pt_it"]
    increment_ptt = 1
    it_mcmc_ptt=1

    
    rcm = load_rcm(filename, device, dtype)
    ptt_updates = get_ptt_updates(filename)
    
    upd=ptt_updates[-1]

    params=load_params(filename, upd, device, dtype)

    #all_betas = find_inverse_temperatures(0.25, params)

    num_visibles=params.vbias.shape[0]
    num_hiddens=params.hbias.shape[0]    
    

    if args["ais_ref"]:
        eps = 1e-4
        frequencies = train_data.data.mean(0)
        
        frequencies = torch.clamp(frequencies, min=eps, max=(1.0 - eps))
        vbias_init = (
            torch.log(frequencies/(1.0 - frequencies))
        ).to(device)
        run='_with_ref'
    else:
        vbias_init = torch.zeros_like(params.vbias,device=device)
        run=''

    params_ref = BBParams(
        weight_matrix=torch.zeros_like(params.weight_matrix,device=device),
        vbias=vbias_init.clone(),
        hbias=torch.zeros_like(params.hbias,device=device),
    )


    print("update",upd,"\nlist_betas",all_betas)
    list_params = []
    for i, beta in enumerate(all_betas):
        
        params_new=params.clone()

        params_new.weight_matrix = beta * params.weight_matrix # + (1 - beta) * params_ref.weight_matrix
        params_new.vbias = beta * params.vbias + (1 - beta) * params_ref.vbias
        params_new.hbias = beta * params.hbias #+ (1 - beta) * params_ref.hbias
        

        list_params.append(params_new)

    
    chains = init_sampling(
        rcm=None,
        n_gen=n_sample_ptt,
        list_params=list_params,
        it_mcmc=it_mcmc_init_ptt,
        device=device,
        dtype=dtype,
        #start_v=train_data,
    )

    #we run a pre-ptt to get a good initialisation where to start to measure
    print("Pre-sampling")
    chains,acc_rates, _ = ptt_sampling(
            rcm=None,
            list_params=list_params,
            chains=chains,
            index=None,
            it_mcmc=10000,
            increment=increment_ptt,
            show_acc_rate=False,
            show_pbar=True,
        )


    if args["optimize_betas"]:
        opt='optimized'
        optimal_acc_rate=0.25
        if min(acc_rates)<optimal_acc_rate:
            print("Run the code with more temperatures because the minimal aceptance rate is:",min(acc_rates))
            return 0
        
        for i in range(2):
            i0=0
            list_params_new=list(list_params)
            chains_new=list(chains)
            while max(acc_rates[:-1])>optimal_acc_rate+0.2 and i0<len(list_params_new)-1:
                
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
                print(i,acc_rates[i])
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

                    if min(acc_rates)<optimal_acc_rate:
                        list_params_new=list(list_params_old)
                        chains_new=list(chains_old)
                        i0+=1
                else:
                    i0+=1

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
    pbar.set_description("sampling PT updates")
    
    rates=torch.zeros(len(list_params)-1)

    
    for t in range(it_mcmc_sample_ptt):
        
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

        rates+=acc_rates
        chain_proj = chains[-1].visible @ V_dataT.mT[:,0:2] / n_visibles**.5
        
        proj_t[t] = chain_proj.T.cpu()

        pbar.update(1)
    
    pbar.close()
    rates/=it_mcmc_sample_ptt
    print("mean_exchange_rate:",rates)
    
    with h5py.File(filename, "a") as f:
        for k in ['pt'+str(num_beta)+opt+'_proj_t'+run,'pt'+str(num_beta)+opt+'_num_models'+run,'pt'+str(num_beta)+opt+'_ex_rate'+run]:
            if k in f.keys():
                del f[k]
        f.create_dataset('pt'+str(num_beta)+opt+'_proj_t'+run, data=proj_t)
        f.create_dataset('pt'+str(num_beta)+opt+'_ex_rate'+run, data=rates)
        f['pt'+str(num_beta)+opt+'_num_models'+run] =len(list_params)
        

    
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
