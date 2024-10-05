# fast-RBM
Code for ICLR 2025 submission #2003


## Installation

/!\ To install PyTorch with GPU support, see [this link](https://pytorch.org/).

To install the package, you need to
```bash
git clone https://github.com/anonymousrbm/iclr-2025.git
cd iclr-2025
bash download_dataset.sh
pip install -r requirements.txt
pip install -e .
```

## Usage 
All scripts can be called with `--help` options to get a description of the arguments

### 1. Mesh  
Compute the mesh on the intrinsic space 
 ```bash
python scripts/compute_mesh.py -d MICKEY --variable_type Ising --dimension 0 1 --border_length 0.04 --n_pts_dim 100 --device cuda -o ./mesh.h5 
 ```
#### Arguments 
  - `-d` is the name of the dataset to load ("MICKEY", "MNIST" or "GENE")
  - `--variable_type` should be set to Ising for the mesh. Can be ("Ising", "Bernoulli", "Continuous" or "Potts"). Currently, only "Ising" works for the RCM.
  - `--dimension` is the index of the dimensions of the intrinsic space
  - `--border_length` should be set as 2/50 or less
  - `--n_pts_dim` is the number of points of the mesh for each dimension. The total number of points will be `n_pts_dim**n_dim`
  - `--device` is the pytorch device you want to use. On lower dimensions, the CPU and GPU have similar performance.
  - `-o` The filename for your mesh.

### 2. RCM

The training of the RCM is divided in two parts.

#### 1. Mesh
The first one is to compute a mesh over the so-called longitudinal space.

This space is defined by the $N_d$ first principal components of the dataset. 

The script you need to run is `scripts/compute_mesh.py`. 

We will go over the possible flags you can set with this script. First the **mandatory** ones: 
##### Mandatory options
 - `--dimension` Here you want to set the enumeration of the principal components of the dataset you want to use. The directions are indexed with the first one being 0. For instance, if you want to use the first 3, you should set `--dimension 0 1 2`.

##### Mesh optional options
 - `--n_pts_dim` The maximum number of discretization points on each direction of the longitudinal space. The default is $100$ and is a good value. The maximum number of points you will have in the end will be $N_d^{N_{pts}}$. So for $3$ directions and $100$ points per directions, you will have $10^6$ points. If you want to use more directions, you should reduce it (for example, for $4$ directions, I used $50$ and the computation time remained reasonable)
 - `-o` or `--filename` The path to the file you want to save the mesh in. By default it will save the mesh in a `mesh.h5` file in the current working directory.
 - `--with_bias`. **Important** optional flag. With this flag, the parametrization of the RCM will be slightly different. The longitudinal space will be constructed from the PCA of the dataset (centered) instead of the SVD of the non-centered dataset. This allows to "win" one direction, by constructing the mesh in a clever way. If you want to use this new direction you should still specify the `--dimension` flag. For example: if you set `--dimension 0 1 2 --with_bias` then your RCM will have $4$ directions: the bias directions and the first $3$ principal components.


##### Example
The command I typically run to generate a mesh on `MNIST-01` will be:
```bash
python scripts/compute_mesh.py -d MNIST --subset_labels 0 1 --dimension 0 1 2 --with_bias -o output/mesh/MNIST01_012_wb.h5
```

#### 2. Train the RCM
Now you have your mesh file. You can reuse it for several training of the RCM, to try different hyperparameters.

##### Mandatory flags
 - `--mesh_file` Path to the pre-computed mesh (the one from the previous section)
 - `-d` or `--data` The path to the dataset. Same as in the previous section. 

##### Optional flags
 - `--num_hidden` The starting number of hidden units
 - `--decimation` Use decimation algorithm, will decrease the number of hidden units until all are significant
 - `--feature_threshold` Threshold to decimate features, the default value is $500$ and I never touched it. 
 - `--max_num_hiddens` The maximum number of hidden nodes after training. If the decimation is not enough, will delete the features of lesser importance. 
 - `--stop_ll` Threshold for the early stopping strategy. When the exponential average of the test loss goes below this threshold the training will stop. The default value is $10^{-2}$
 - `--max_iter` The maximum number of gradient updates for each training. Defaults to $10^5$. If the training converges before, it will stop. If the training reaches the maximum number of iterations and you still see significant improvements on the log-likelihood, consider putting more steps.
 - `--learning_rate` The base learning rate, will be divided by the number of dimensions of the dataset. If the training diverges at the beginning, consider putting a lower value (the default one is $10^-2$)
 - `--adapt` **Important** You want to set this flag, it allows for an adaptative learning rate during the training of the RCM.
 - `--smooth_rate` The training of the RCM uses the hessian. However, to avoid too rapidly changing updates, the hessian is computed as the interpolation between the previous hessian and the new one: $H^{(t)} = (1-\lambda) H^{(t-1)} + \lambda H^{new}$ where $\lambda$ is the smooth rate. The default value is $0.1$ and usually works well.
 - `--eigen_threshold` The option is not used anymore, don't bother using it, I should remove it in the near future.
 - `--seed` If you want to seed the training for reproducibility. Otherwise, no seed is set.
 - `--device` Same as above
 - `--dtype` The PyTorch dtype used during training: `double` -> `torch.float64`, `float` -> `torch.float32`,`int` -> `torch.int`  

##### Example
The command I typically use to train a RCM with a maximum amount of 20 hidden nodes on `MNIST-01` will be:
```bash
scripts/train_rcm.py -d MNIST --subset_labels 0 1 --mesh_file ./output/mesh/MNIST01_012_wb.h5 -o output/rcm/MNIST01_012_wb.h5 --adapt --decimation --max_iter 1000000 --stop_ll 0.0001 --num_hidden 100 --max_num_hiddens 20
```

### 3. Convert RCM to RBM
##### Conversion arguments
 - `--path` or '-i' Path to the RCM hdf5 archive
 - `--output` or `-o` Path to the new RBM hdf5 archive. (If the file already exists, it will be overwritten).
 - `--num_hiddens` The number of hidden nodes for the RBM. If it is lower or equal to the number of hidden nodes of the RCM, no hidden nodes will be added. The nodes added have no bias and the weights associated are initialized randomly with a $10^{-4}$ standard deviation.
 - `--therm_steps` The number of Gibbs steps to perform from the RCM sampled configurations. Useless now in theory... 
 - `--trial` Index of the RCM trial to use. The default will select the best trial, i.e. the one where the recovered RBM has the highest log-likelihood.

##### RBM hyperparameters
 - `--learning_rate` Learning rate to use for the recovered RBM when training. 
 - `--gibbs_steps` Number of Gibbs steps to perform between each of the gradient updates during RBM training.
 - `--batch_size` Batch size, during training of the RBM.
 - `--num_chains` Number of parallel chains during training of the RBM.


##### Example
The command I typically run to convert the previously trained RCM to an RBM will be:
```bash
python scripts/rcm_to_rbm.py -d MNIST --subset_labels 0 1  -i ./output/rcm/MNIST01_012_wb.h5 -o output/rbm/MNIST01_012_wb.h5 --num_hiddens 20 --therm_steps 10000 --learning_rate 0.01 --gibbs_steps 100 --batch_size 2000 --num_chains 2000
```

### 4. Restore the training
If you want to continue the training of a RBM (be it one recovered from a RCM or a previously trained one), you can use the same scripts. The differences are that you should add the `--restore` flags. Also some arguments are not useful anymore and can be safely ignored:
 - `--num_hiddens`
 - `--batch_size`
 - `--gibbs_steps`
 - `--learning_rate`
 - `--num_chains`

Finally the updates will be added to the same archive you provide as an input through `--filename`. If the `--restore` flag is set, then the file will **not** be overwritten.

##### Example
To continue the training of the RBM recovered from the RCM I run:
```bash
python scripts/train_rbm.py -d MNIST --subset_labels 0 1 --filename output/rbm/MNIST01_012_wb.h5  --num_updates 10000 --n_save 500 --spacing exp --device cuda --restore
```
### 5. RBM from scratch

To train a RBM, you need to use the [train_rbm.py](https://github.com/anonymousrbm/iclr-2025/tree/main/scripts/train_rbm.py) script.

### RBM hyperparameters
 - `--num_hiddens` Number of hidden nodes for the RBM. Setting it to $20$ or less allows to recover the exact log-likelihood of the model by enumerating on all hidden configurations. 
 - `--batch_size` Batch size, defaults to $2000$. Changing the batch size has an impact on the noise in the estimation of the positive term of the gradient. Setting it to a low value can lead to a very bad estimation and a bad training, but setting it too high can lead to an exact gradient, losing the benefits of the SGD (and remain trapped in a local minima for example).
 - `--num_chains` Number of parallel chains, defaults to $2000$. Setting it to a much higher value than the batch size does not provide benefits, since it only impacts the estimation of the negative term of the gradient. 
 - `--learning_rate` Learning rate. Defaults to $0.01$, setting a larger learning rate often leads to instability. 
 - `--num_updates` The training time is indexed on the number of gradient updates performed and not the number of epochs.
 - `--beta` The inverse temperature to use during training (Defaults to $1$ and should not be changed)

### Save options
 - `--filename` The path to the hdf5 archive to save the RBM during training. It will overwrite previously existing file. 
 - `--n_save` The number of machines to save during the training. 
 - `--spacing` Can be `exp` or `linear`, defaults to `exp`. When `exp` is selected, the time between the save of two models will increase exponentially. (It will look good in log-scale). When `linear` is selected, the time between the save of two models will be constant. 
Saving lots of models can quickly become the computational bottleneck, leading to long execution times. 
 - `--log` For now it is deprecated so you don't care about it 

### PyTorch options
 - `--device` The device on which to run the computations. Follows the PyTorch semantic so you can select which GPU to use with 'cuda:1' for example.
 - `--dtype` The dtype of all the tensors. can be `int`, `double` or `float`. The default is `float` which corresponds to `torch.float32`. 

### Example
The command I typically use to train a RBM on `MNIST-01` will be
```bash
python scripts/train_rbm.py -d MNIST --subset_labels 0 1 --filename output/rbm/MNIST01_from_scratch.h5  --num_updates 10000 --n_save 500 --spacing exp --num_hiddens 20 --batch_size 2000 --num_chains 2000 --learning_rate 0.01 --device cuda --dtype float
``` 

# Restore the training



### 6. PTT
```bash
python scripts/ptt_sampling.py -i RBM.h5 -o sample_RBM_mickey.h5 --num_samples 2000 --target_acc_rate 0.9 --it_mcmc 1000
```
#### Arguments 
- `-i` is the filename of the RBM obtained at step 4 or 5.
- `-o` is the file in which to save the samples.
- `--filename_rcm` the name of the file used to initialize the RBM at step 3. Do not set if the RBM was trained from scratch
- `--target_acc_rate` The target acceptance rate between two consecutive machines
- `--it_mcmc` The number of gibbs steps performed by each machine.

## Analysis
See [rcm_analysis.ipynb](notebooks/rcm_analysis.ipynb) for an analysis of the file obtained at step 2 and [rbm_analysis.ipynb](notebooks/rbm_analysis.ipynb) for an analysis of the files obtained at step 4 or 5, as well as the results for the PTT.