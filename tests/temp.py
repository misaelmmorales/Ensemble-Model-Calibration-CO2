import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import gstools as gs
from pde import DiffusionPDE, ScalarField, UnitGrid

from neuralop.models import FNO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

import scipy.linalg as sla
import scipy.optimize as sopt
from pyswarms.single import GlobalBestPSO

from sklearn.metrics import r2_score, mean_absolute_percentage_error
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio

NSAMPLES  = 500
NDIM      = 32
EPOCHS    = 201
MONITOR   = 10
BATCHSIZE = 16

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

DEVICE = check_torch()

#############################################################################
################################ MODEL SPACE ################################
#############################################################################
def make_model_space(variance=1, len_scale=6, showfig:bool=True, figsize=(15,6), cmap='jet'):
    x = y = range(NDIM)
    model = gs.Gaussian(dim=2, var=variance, len_scale=len_scale)

    mm = np.zeros((NSAMPLES, NDIM, NDIM))
    for i in tqdm(range(NSAMPLES), desc='Generating model space'):
        srf = gs.SRF(model)
        mm[i] = srf((x,y), mesh_type='structured')
    np.save('features.npy', mm)

    if showfig:
        plt.figure(figsize=figsize)
        for i in range(36):
            plt.subplot(3, 12, i+1)
            plt.imshow(mm[i], cmap=cmap)
            plt.title('R{}'.format(i))
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('results/features_realizations.png', dpi=600)
        plt.close()

    return mm

def load_model_space(fname:str='features.npy'):
    return np.load(fname)

#############################################################################
################################ DATA SPACE #################################
#############################################################################
def make_data_space(mm, monitor=50, showfig:bool=True, figsize=(15,5), cmap='jet',
                    diffusivity=0.75, bcs=[{'derivative':0},{'value':0}], noise=0.0, t_range=20, dt=0.1,
                    backend='numpy', scheme='rk', ret_info=False, trackers=[None]):
    dd = np.zeros_like(mm)
    grid = UnitGrid([mm.shape[1],mm.shape[2]])
    state = ScalarField.random_uniform(grid, 0.2, 0.3)

    start = time()
    for i in range(mm.shape[0]):
        state.data = mm[i]
        eq = DiffusionPDE(diffusivity=diffusivity, bc=bcs, noise=noise)
        dd[i] = eq.solve(state, t_range=t_range, dt=dt, backend=backend, scheme=scheme, ret_info=ret_info, tracker=trackers).data
        if (i+1) % monitor == 0:
            print('Simulation [{}/{}] done ..'.format(i+1, mm.shape[0]))
    print('Total Simulation Time: {:.2f}'.format((time()-start)/60))
    np.save('targets.npy', dd)

    if showfig:
        labs = ['LogPerm','Diffusion']
        hues = ['black','blue']
        mult = 5
        fig, axs = plt.subplots(2, 12, figsize=figsize, sharex=True, sharey=True)
        for j in range(12):
            k = j*mult
            ax1, ax2 = axs[0,j], axs[1,j]
            im1 = ax1.imshow(mm[k], cmap=cmap)
            im2 = ax2.imshow(dd[k], cmap=cmap)
            ax1.set(title='R{}'.format(k))
            [a.set_ylabel(labs[i], color=hues[i]) for i,a in enumerate([ax1,ax2])] if j==0 else None
        plt.tight_layout()
        plt.savefig('results/features_and_targets.png', dpi=600)
        plt.close()

    return dd

def load_data_space(fname:str='targets.npy'):
    return np.load(fname)

#############################################################################
################################# DATA PREP #################################
#############################################################################
mm = load_model_space()
dd = load_data_space()
print('m: {} | d: {}'.format(mm.shape, dd.shape))
print('Loading data done ..')

mm_norm = (mm - mm.mean()) / mm.std()
dd_norm = (dd - dd.mean()) / dd.std()

features = torch.tensor(np.expand_dims(mm_norm,1), dtype=torch.float32)
targets  = torch.tensor(np.expand_dims(dd_norm,1), dtype=torch.float32)

idx = np.random.choice(range(NSAMPLES), NSAMPLES, replace=False)
n_train, n_valid = 350, 75
train_idx = idx[:n_train]
valid_idx = idx[n_train:n_train+n_valid]
test_idx  = idx[n_train+n_valid:]
np.save('idx.npy', idx)

X_train, X_valid, X_test = features[train_idx], features[valid_idx], features[test_idx]
y_train, y_valid, y_test = targets[train_idx],  targets[valid_idx],  targets[test_idx]

trainset = TensorDataset(X_train, y_train)
trainloader = DataLoader(trainset, batch_size=BATCHSIZE, shuffle=True)

validset = TensorDataset(X_valid, y_valid)
validloader = DataLoader(validset, batch_size=15, shuffle=True)

#############################################################################
################################# SURROGATE #################################
#############################################################################
fno = FNO(n_modes=(12,12), n_layers=4, norm='group_norm', use_mlp=True, mlp_dropout=0.1,
          in_channels=1, lifting_channels=256, hidden_channels=256, projection_channels=256, out_channels=1).to(DEVICE)
nparams = sum(p.numel() for p in fno.parameters() if p.requires_grad)
optimizer = optim.AdamW(fno.parameters(), lr=1e-3, weight_decay=1e-8)
criterion = nn.MSELoss().to(DEVICE)

print('-'*15+' Training FNO [{:,}] '.format(nparams)+'-'*15)
train_loss, valid_loss = [], []
start = time()
for epoch in range(EPOCHS):
    fno.train()
    epoch_train_loss = []
    for i, (x,y) in enumerate(trainloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
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
            x, y = x.to(DEVICE), y.to(DEVICE)
            u = fno(x)
            loss = criterion(u,y)
            epoch_valid_loss.append(loss.item())
        valid_loss.append(np.mean(epoch_valid_loss))
    if (epoch+1) % MONITOR == 0:
        print('Epoch: [{}/{}] | Loss: {:.4f} | Valid Loss: {:.4f}'.format(epoch+1, EPOCHS-1, train_loss[-1], valid_loss[-1]))
print('Trianing Time: {:.2f} minutes'.format((time()-start)/60))
losses = pd.DataFrame({'train':train_loss, 'valid':valid_loss})
losses.to_csv('results/losses.csv')
torch.save(fno.state_dict(), 'results/surrogate.pth')

#############################################################################
################################# INFERENCE #################################
#############################################################################
uu_train = fno(X_train.to(DEVICE)).detach().cpu().numpy()
uu_valid = fno(X_valid.to(DEVICE)).detach().cpu().numpy()
uu_test  = fno(X_test.to(DEVICE)).detach().cpu().numpy()
uu_train_flat = uu_train.reshape(X_train.shape[0], -1)
uu_valid_flat = uu_valid.reshape(X_valid.shape[0], -1)
uu_test_flat  = uu_test.reshape(X_test.shape[0], -1)

yy_train = y_train.detach().cpu().numpy()
yy_valid = y_valid.detach().cpu().numpy()
yy_test  = y_test.detach().cpu().numpy()
yy_train_flat = yy_train.reshape(y_train.shape[0], -1)
yy_valid_flat = yy_valid.reshape(y_valid.shape[0], -1)
yy_test_flat  = yy_test.reshape(y_test.shape[0], -1)

# metrics
r2_train = r2_score(uu_train_flat, yy_train_flat)
r2_valid = r2_score(uu_valid_flat, yy_valid_flat)
r2_test  = r2_score(uu_test_flat, yy_test_flat)

mape_train = mean_absolute_percentage_error(uu_train_flat, yy_train_flat)
mape_valid = mean_absolute_percentage_error(uu_valid_flat, yy_valid_flat)
mape_test  = mean_absolute_percentage_error(uu_test_flat, yy_test_flat)

mse_train = mean_squared_error(uu_train, yy_train)
mse_valid = mean_squared_error(uu_valid, yy_valid)
mse_test  = mean_squared_error(uu_test, yy_test)

ssim_train = structural_similarity(uu_train, yy_train, data_range=1.0, channel_axis=1)
ssim_valid = structural_similarity(uu_valid, yy_valid, data_range=1.0, channel_axis=1)
ssim_test  = structural_similarity(uu_test, yy_test, data_range=1.0, channel_axis=1)

psnr_train = peak_signal_noise_ratio(uu_train, yy_train, data_range=1.0)
psnr_valid = peak_signal_noise_ratio(uu_valid, yy_valid, data_range=1.0)
psnr_test  = peak_signal_noise_ratio(uu_test, yy_test, data_range=1.0)

pd.DataFrame({'train':[r2_train, mape_train, mse_train, ssim_train, psnr_train],
              'valid':[r2_valid, mape_valid, mse_valid, ssim_valid, psnr_valid],
              'test':[r2_test, mape_test, mse_test, ssim_test, psnr_test]}).to_csv('results/metrics.csv')

#############################################################################
################################# VISUALIZE #################################
#############################################################################
plt.figure(figsize=(8,5))
plt.plot(losses.index, losses['train'], color='tab:blue', label='Train')
plt.plot(losses.index, losses['valid'], color='tab:orange', label='Valid')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, which='both')
plt.tight_layout()
plt.savefig('results/training_performance.png', dpi=600)
plt.close()

uu = fno(features.to(DEVICE)).detach().cpu().numpy().squeeze()
np.save('results/predictions.npy', uu)
print('X: {} | y: {} | u: {}'.format(mm_norm.shape, dd_norm.shape, uu.shape))

fig, axs = plt.subplots(4, 8, figsize=(15,6), sharex=True, sharey=True)
labs = ['LogPerm','True','Pred','% Diff']
hues = ['k', 'b', 'r', 'k']
mult = 10
for j in range(8):
    k = j*mult
    ax1, ax2, ax3, ax4 = axs[0,j], axs[1,j], axs[2,j], axs[3,j]
    ax1.set_title('R{}'.format(k))
    [a.set_ylabel(labs[i], color=hues[i]) for i,a in enumerate([ax1,ax2,ax3,ax4])] if j == 0 else None
    ee = sla.norm( np.expand_dims(dd[k], 0) - np.expand_dims(uu[k], 0), axis=0) / sla.norm(dd[k])
    im1 = ax1.imshow(mm_norm[k], cmap='turbo',  vmin=-2.5, vmax=2.5)
    im2 = ax2.imshow(dd_norm[k], cmap='jet',    vmin=-2,   vmax=2)
    im3 = ax3.imshow(uu[k],      cmap='jet',    vmin=-2,   vmax=2)
    im4 = ax4.imshow(ee,         cmap='binary', vmin=0,    vmax=0.1)
    #[plt.colorbar(i, pad=0.04, fraction=0.046) for i in [im1,im2,im3,im4]]
plt.tight_layout()
plt.savefig('results/predictions.png', dpi=600)
plt.close()

print('-'*20+' DONE !!! '+'-'*20+'\n')