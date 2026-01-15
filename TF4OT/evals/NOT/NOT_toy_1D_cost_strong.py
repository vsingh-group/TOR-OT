import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

import numpy as np
import torch
import torch.nn as nn
import gc

import torch.nn.functional as F
#from IPython.display import clear_output

from TF4OT.evals.NOT.src.tools import unfreeze, freeze
from TF4OT.evals.NOT.src import distributions
from TF4OT.evals.NOT.src.plotters import plot_1D, plot_1D_discrete

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# %load_ext autoreload
# %autoreload 2

def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

def sqeuclidean_cost (X, Y):
  return (X-Y).square().flatten(start_dim=1)
  #return F.mse_loss(X, Y)

def euclidean_cost (X, Y):
  return torch.linalg.vector_norm(X-Y, dim=-1).flatten(start_dim=1)

def cityblock_cost (X, Y):
    #return (X - Y).abs().sum(dim=-1).flatten(start_dim=1)
    return F.l1_loss(X, Y)

def canberra_cost (X, Y):
    eps = 1e-8  # tune if needed
    num = (X - Y).abs()
    den = X.abs() + Y.abs() + eps
    canberra = (num / den).sum(dim=-1)      # sum over feature dim
    return canberra

def chebyshev_cost (X, Y):
    return torch.linalg.vector_norm(X-Y, dim=-1, ord=np.inf).flatten(start_dim=1)

def minkowski_cost (X, Y, p=3):
    return torch.linalg.vector_norm(X-Y, dim=-1, ord=p).flatten(start_dim=1)

def main_routine(X_sampler, Y_sampler, num_bins, rng, metric, max_steps=2501, verbose=False):
    
    #DEVICE_IDS = [2,1,0]
    if metric == "sqeuclidean":
        COST = sqeuclidean_cost
    elif metric == "euclidean":
        COST = euclidean_cost
    elif metric == "cityblock":
        COST = cityblock_cost
    elif metric == "canberra":
        COST = canberra_cost
    elif metric == "chebyshev":
        COST = chebyshev_cost
    elif metric == "minkowski":
        COST = minkowski_cost
    else:
        raise ValueError("Unknown metric: {}".format(metric))

    T_ITERS = 10
    f_LR, T_LR = 1e-4, 1e-4
    
    ZD = 1
    Z_STD = 0.1
    
    BATCH_SIZE = 64
    Z_SIZE = 4
    
    #MAX_STEPS = 5001
    MAX_STEPS = max_steps
    PLOT_INTERVAL = MAX_STEPS-1
    SEED = 0x000000

    DIM = 1
    
    assert torch.cuda.is_available()
    #torch.cuda.set_device(DEVICE_IDS[0])
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # X_sampler = distributions.StandardNormalSampler(dim=1)
    # # Y_sampler = distributions.StandardNormalScaler(distributions.MixNGaussiansSampler(dim=1, n=2))
    # Y_sampler = distributions.StandardNormalScaler(distributions.MixNGaussiansSampler(dim=1, n=3)) # <-- 3 Gaussians
    
    H = 100
    
    T = nn.Sequential(
        nn.Linear(DIM+ZD, H),
        nn.ReLU(True), 
        nn.Linear(H, H),
        nn.ReLU(True),
        nn.Linear(H, H),
        nn.ReLU(True),
        nn.Linear(H, DIM)
    ).cuda()
    
    f = nn.Sequential(
        nn.Linear(DIM, H),
        nn.ReLU(True),
        nn.Linear(H, H),
        nn.ReLU(True),
        nn.Linear(H, H),
        nn.ReLU(True),
        nn.Linear(H, 1)
    ).cuda()
    
    T.apply(weights_init_mlp); f.apply(weights_init_mlp)
    
    # if len(DEVICE_IDS) > 1:
    #     T = nn.DataParallel(T, device_ids=DEVICE_IDS)
    #     f = nn.DataParallel(f, device_ids=DEVICE_IDS)
    
    #print('T params:', np.sum([np.prod(p.shape) for p in T.parameters()]))
    #print('f params:', np.sum([np.prod(p.shape) for p in f.parameters()]))
    
    T_opt = torch.optim.Adam(T.parameters(), lr=T_LR, weight_decay=1e-10)
    f_opt = torch.optim.Adam(f.parameters(), lr=f_LR, weight_decay=1e-10)
    
    iterator = range(MAX_STEPS)
    if verbose:
        iterator = tqdm(iterator)

    for step in iterator:
        # T optimization
        unfreeze(T); freeze(f)
        for t_iter in range(T_ITERS):
            T_opt.zero_grad()
            X = X_sampler.sample(BATCH_SIZE).reshape(-1, 1, DIM).repeat(1, Z_SIZE, 1)
            with torch.no_grad():
                Z = torch.randn(BATCH_SIZE, Z_SIZE, ZD, device='cuda') * Z_STD
                XZ = torch.cat([X, Z], dim=2)
            T_XZ = T(
                XZ.flatten(start_dim=0, end_dim=1)
            ).permute(1, 0).reshape(DIM, -1, Z_SIZE).permute(1, 2, 0)
            
            T_loss = COST(X, T_XZ).mean() - f(T_XZ.flatten(start_dim=0, end_dim=1)).mean()

            #f(T_XZ.flatten(start_dim=0, end_dim=1)).mean() 
            T_loss.backward(); T_opt.step()

        del T_loss, T_XZ, X, Z
        gc.collect(); torch.cuda.empty_cache()

        # f optimization
        
        freeze(T); unfreeze(f)

        X = X_sampler.sample(BATCH_SIZE)
        with torch.no_grad():
            Z = torch.randn(BATCH_SIZE, ZD, device='cuda') * Z_STD
            XZ = torch.cat([X, Z], dim=1)
            T_XZ = T(XZ)
        Y = Y_sampler.sample(BATCH_SIZE)

        f_opt.zero_grad()
        f_loss = f(T_XZ).mean() - f(Y).mean()
        f_loss.backward(); f_opt.step()

        del f_loss, Y, X, T_XZ, Z, XZ
        gc.collect(); torch.cuda.empty_cache()

        # if step % PLOT_INTERVAL == 0:
        #     clear_output(wait=True)
        #     print("Step", step)
        #     print("Plotting")

        #     plot_1D(X_sampler, Y_sampler, T, ZD, Z_STD, num_samples=1024)
        #     plt.show()

        # gc.collect()
        # torch.cuda.empty_cache()
    if verbose:
        print("1d neural ot run completed")
    ot_plan_T, ot_plan_hist = plot_1D(X_sampler, Y_sampler, T, ZD, Z_STD, num_samples=1_000_000, num_bins=num_bins, rng=rng)

    #print("Plotting solutions on DOT (non-unique!)")
    #plot_1D_discrete(X_sampler, Y_sampler, num_samples=1024)
    return ot_plan_T, ot_plan_hist

if __name__ == "__main__":
    main_routine(num_bins=30)