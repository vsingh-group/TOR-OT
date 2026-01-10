import os, sys
sys.path.append("..")

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import gc

from TF4OT.evals.NOT.src import distributions
import torch.nn.functional as F

from IPython.display import clear_output

from TF4OT.evals.NOT.src.tools import unfreeze, freeze
from TF4OT.evals.NOT.src import distributions

from TF4OT.evals.NOT.src.plotters import plot_generated_2D, plot_bar_and_stochastic_2D

from tqdm.notebook import tqdm as tqdm

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

def hamming_cost (X, Y):
    C = ot.dist(X, Y, metric="hamming", use_tensor=True)
    return (X != Y).float().sum(dim=-1)

def main_routine(X_sampler, Y_sampler, num_bins, rng, metric):

    #DEVICE_IDS = [2]
    if metric == "sqeuclidean":
        COST = sqeuclidean_cost
    elif metric == "euclidean":
        COST = euclidean_cost
    elif metric == "cityblock":
        COST = cityblock_cost
    elif metric == "canberra":
        COST = canberra_cost
    elif metric == "hamming":
        COST = hamming_cost
    else:
        raise ValueError("Unknown metric: {}".format(metric))

    T_ITERS = 10
    f_LR, T_LR = 1e-4, 1e-4

    ZD = 2
    Z_STD = 0.1

    BATCH_SIZE = 64
    Z_SIZE = 4

    #PLOT_INTERVAL = 200
    #MAX_STEPS = 10_001
    MAX_STEPS = 5_001
    SEED = 0x000000

    DIM = 2

    assert torch.cuda.is_available()
    #torch.cuda.set_device(DEVICE_IDS[0])
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    #X_sampler = distributions.StandardNormalSampler(dim=DIM)
    #Y_sampler = distributions.StandardNormalScaler(distributions.SwissRollSampler()) # <-- Swiss Roll
    # distributions.StandardNormalScaler(distributions.Mix8GaussiansSampler()) # <-- 8 Gaussians

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

    for step in tqdm(range(MAX_STEPS)):
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
            # STRONG COST:
            T_loss = F.mse_loss(X[:, 0], T_XZ.mean(dim=1)).mean() - f(
                T_XZ.flatten(start_dim=0, end_dim=1)).mean()
            # WEAK COST:
            #T_loss = F.mse_loss(X[:, 0], T_XZ.mean(dim=1)).mean() - f(
            #    T_XZ.flatten(start_dim=0, end_dim=1)).mean() - T_XZ.var(dim=1).mean() / Z_SIZE
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
        #     print("(DOT map appears only on the last step)")

        #     fig, axes, plan_matrix = plot_generated_2D(X_sampler, Y_sampler, T, ZD, Z_STD, num_samples=1_000_000, num_bins=(5,6))
        #     plt.show()
            
        #     # fig, axes = plot_bar_and_stochastic_2D(
        #     #     X_sampler, Y_sampler, T, ZD, Z_STD,
        #     #     plot_discrete=True if step == MAX_STEPS-1 else False
        #     # )
        #     # plt.show()

        # gc.collect()
        # torch.cuda.empty_cache()
    print("2d neural ot run completed")
    plan_matrix = plot_generated_2D(X_sampler, Y_sampler, T, ZD, Z_STD, num_samples=1_000_000, num_bins=num_bins)
    return plan_matrix


if __name__ == "__main__":
    main_routine(num_bins=30, rng=None, metric="sqeuclidean")