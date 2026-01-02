import os, sys
sys.path.append("..")

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

import numpy as np
import torch
import torch.nn as nn
import gc

import torch.nn.functional as F
# from IPython.display import clear_output

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

def main_routine(X_sampler, Y_sampler, num_bins):
    
#    DEVICE_IDS = [6]

    T_ITERS = 10
    f_LR, T_LR = 1e-4, 1e-4
    
    ZD = 1
    Z_STD = 0.1
    
    BATCH_SIZE = 64
    Z_SIZE = 4

    MAX_STEPS = 5001
    #MAX_STEPS = 201
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
    
#    if len(DEVICE_IDS) > 1:
#        T = nn.DataParallel(T, device_ids=DEVICE_IDS)
#        f = nn.DataParallel(f, device_ids=DEVICE_IDS)
    
    print('T params:', np.sum([np.prod(p.shape) for p in T.parameters()]))
    print('f params:', np.sum([np.prod(p.shape) for p in f.parameters()]))
    
    T_opt = torch.optim.Adam(T.parameters(), lr=T_LR, weight_decay=1e-10)
    f_opt = torch.optim.Adam(f.parameters(), lr=f_LR, weight_decay=1e-10)
    
    for step in tqdm(range(MAX_STEPS), leave=True):
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
            
            T_loss = F.mse_loss(X, T_XZ).mean() - T_XZ.var(dim=1).mean() -\
            f(T_XZ.flatten(start_dim=0, end_dim=1)).mean() 
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
    
        #     ot_plan_T, ot_plan_hist = plot_1D(X_sampler, Y_sampler, T, ZD, Z_STD, num_samples=1024, num_bins=30)
        #     plt.show()
    
        # gc.collect()
        # torch.cuda.empty_cache()
    print("returning results")
    ot_plan_T, ot_plan_hist, xedges, yedges = plot_1D(X_sampler, Y_sampler, T, ZD, Z_STD, num_samples=1024, num_bins=num_bins)

    #print("Plotting solutions on DOT (non-unique!)")
    #plot_1D_discrete(X_sampler, Y_sampler, num_samples=1024)
    return ot_plan_T, ot_plan_hist, xedges, yedges

if __name__ == "__main__":
    main_routine(num_bins=30)