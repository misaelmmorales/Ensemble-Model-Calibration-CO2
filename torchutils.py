import os
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy.io as sio
import scipy.linalg as sla
from scipy.spatial import distance_matrix
from pyesmda import ESMDA, ESMDA_RS, approximate_cov_mm

from sklearn.metrics import r2_score, mean_absolute_percentage_error
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from neuralop.models import FNO, UNO
from neuralop.losses import LpLoss
from torch.utils.data import TensorDataset, DataLoader

NREALIZATIONS = 1000
NX, NY = 64, 64
NTT, NT = 28, 20
NTRAIN = 700
NVALID = 150
BATCHSIZE = 16

sec2year   = 365.25 * 24 * 60 * 60
Darcy      = 9.869233e-13
psi2pascal = 6894.76
co2_rho    = 686.5266
milli      = 1e-3
mega       = 1e6

def check_torch(verbose:bool=True):
    if torch.cuda.is_available():
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
        if verbose:
            print('-'*60)
            print('----------------------- VERSION INFO -----------------------')
            print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
            print('# Device(s) available: {}, Name(s): {}'.format(count, name))
            print('-'*60)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device
    else:
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        if verbose:
            print('-'*60)
            print('----------------------- VERSION INFO -----------------------')
            print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
            print('-'*60)
        device = torch.device('cpu')
        return device
device = check_torch(verbose=True)

def plot_losses(losses):
    plt.figure(figsize=(10,4))
    plt.plot(losses.index, losses['train'], label='Train')
    plt.plot(losses.index, losses['valid'], label='Valid')
    plt.grid(True, which='both'); plt.legend()
    plt.tight_layout()
    plt.show()

def split_predictions(X, y, u, n:int=10):
    X_train, X_valid, X_test = X
    y_train, y_valid, y_test = y
    u_train, u_valid, u_test = u
    X_train = X_train[:n].detach().cpu().numpy()
    X_valid = X_valid[:n].detach().cpu().numpy()
    X_test  = X_test[:n].detach().cpu().numpy()
    yy_train = y_train[:n].detach().cpu().numpy()
    yy_valid = y_valid[:n].detach().cpu().numpy()
    yy_test  = y_test[:n].detach().cpu().numpy()
    p_train = yy_train[:,0]
    s_train = yy_train[:,1]
    p_valid = yy_valid[:,0]
    s_valid = yy_valid[:,1]
    p_test = yy_test[:,0]
    s_test = yy_test[:,1]
    p_train_pred = u_train[:,0]
    s_train_pred = u_train[:,1]
    p_valid_pred = u_valid[:,0]
    s_valid_pred = u_valid[:,1]
    p_test_pred = u_test[:,0]
    s_test_pred = u_test[:,1]
    orig = (X_train, X_valid, X_test, yy_train, yy_valid, yy_test)
    pres = (p_train, p_valid, p_test, p_train_pred, p_valid_pred, p_test_pred)
    satr = (s_train, s_valid, s_test, s_train_pred, s_valid_pred, s_test_pred)
    return orig, pres, satr


####################################################################################################
if __name__ == '__main__':
    print('torchutils.py is being run as main')
    timesteps = np.load('simulations/data/time_arr.npy')
    print('t1: {}\nt2:{}'.format(timesteps.round(2)[:8], timesteps.round(2)[8:]))

    times = np.zeros((NX,NY,NT))
    for t in range(NT):
        times[:,:,t] =  timesteps[8:][t]
    times_norm = (times - times.min()) / (times.max() - times.min())
    times_norm = np.expand_dims(np.repeat(np.expand_dims(times_norm, 0), NREALIZATIONS, 0), 1)

    all_data = np.load('data/simulations_64x64x28.npz')
    poro = all_data['poro']
    perm = all_data['perm']
    pressure = all_data['pressure'][:,8:]
    saturation = all_data['saturation'][:,8:]
    print('poro: {} | perm: {}'.format(poro.shape, perm.shape))
    print('pressure: {} | saturation: {}'.format(pressure.shape, saturation.shape))

    pmin, pmax = poro.min(), poro.max()
    kmin, kmax = perm.min(), perm.max()
    rmin, rmax = pressure.min(), pressure.max()
    smin, smax = saturation.min(), saturation.max()

    poro_norm = (poro - pmin) / (pmax - pmin)
    perm_norm = (perm - kmin) / (kmax - kmin)
    pressure_norm = (pressure - rmin) / (rmax - rmin)
    saturation_norm = (saturation - smin) / (smax - smin)

    features = np.repeat(np.expand_dims(np.stack([poro_norm, perm_norm], 1), -1), 20, -1)
    features = np.concatenate([features, times_norm], 1)
    targets  = np.moveaxis(np.stack([pressure_norm, saturation_norm], 1), -3, -1)

    features = torch.tensor(features, dtype=torch.float32)
    targets  = torch.tensor(targets, dtype=torch.float32)
    print('features: {} | targets: {}'.format(features.shape, targets.shape))
    print('-'*40)

    ######################
    idx = np.load('data/random_indices.npz')
    train_idx, valid_idx, test_idx = idx['train_idx'], idx['valid_idx'], idx['test_idx']

    X_train, X_valid, X_test = features[train_idx], features[valid_idx], features[test_idx]
    y_train, y_valid, y_test = targets[train_idx],  targets[valid_idx],  targets[test_idx]
    print('Train - X: {} | y: {}'.format(X_train.shape, y_train.shape))
    print('Valid - X: {} | y: {}'.format(X_valid.shape, y_valid.shape))
    print('Test  - X: {} | y: {}'.format(X_test.shape, y_test.shape))
    print('-'*40)

    ######################
    trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=8, shuffle=True)
    validloader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=8, shuffle=False)

    ######################
    fno = FNO(n_modes=(10,10,6), n_layers=3, non_linearity=F.leaky_relu, use_mlp=True, mlp_dropout=0.1,
            in_channels=3, lifting_channels=64, hidden_channels=256, projection_channels=64, out_channels=2).to(device)
    print('FNO # parameters: {:,}'.format(sum(p.numel() for p in fno.parameters() if p.requires_grad)))

    optimizer = optim.Adam(fno.parameters(), lr=1e-3)
    criterion = nn.MSELoss().to(device)

    epochs, monitor = 51, 10
    train_loss, valid_loss = [], []
    start = time()
    for epoch in range(epochs):
        epoch_train_loss = []
        for i, (x,y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            u = fno(x)
            loss = criterion(u,y)
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item())
        train_loss.append(np.mean(epoch_train_loss))
        fno.eval()
        epoch_valid_loss = []
        with torch.no_grad():
            for i, (x,y) in enumerate(validloader):
                x, y = x.to(device), y.to(device)
                u = fno(x)
                loss = criterion(u,y)
                epoch_valid_loss.append(loss.item())
            valid_loss.append(np.mean(epoch_valid_loss))
        if (epoch+1) % monitor == 0:
            print('Epoch: [{}/{}] | Loss: {:.4f} | Valid Loss: {:.4f}'.format(epoch+1, epochs-1, train_loss[-1], valid_loss[-1]))
    print('Training time: {:.2f} minutes'.format((time()-start)/60))
    losses = pd.DataFrame({'train':train_loss, 'valid':valid_loss})
    losses.to_csv('data/fno_losses.csv')
    torch.save(fno.state_dict(), 'data/fno_model.pth')