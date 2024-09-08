import os
import numpy as np
import pandas as pd
import gstools as gs
from tqdm import tqdm
import scipy.io as sio
from pde import UnitGrid, DiffusionPDE, ScalarField, trackers

from skimage.transform import resize
from sklearn.preprocessing import MinMaxScaler

###################################### 3D perms|facies ######################################
NR, NX, NY, NZ = 1272, 64, 64, 8

fnames = []
for root, dirs, files in os.walk('/mnt/e/MLTrainingImages'):
    for f in files:
        if f.endswith('.npy'):
            fnames.append(os.path.join(root, f))
print('# TIs:', len(fnames))

facies = np.zeros((len(fnames), 4, NX, NY, NZ))
for i in tqdm(range(len(fnames))):
    f = np.load(fnames[i]).reshape(256,256,128)
    f1 = resize(f[...,48:56],   (NX,NY,NZ), anti_aliasing=True, preserve_range=True)
    f2 = resize(f[...,48:76],   (NX,NY,NZ), anti_aliasing=True, preserve_range=True)
    f3 = resize(f[...,88:96],   (NX,NY,NZ), anti_aliasing=True, preserve_range=True)
    f4 = resize(f[...,108:116], (NX,NY,NZ), anti_aliasing=True, preserve_range=True)
    ff = np.stack([f1, f2, f3, f4], axis=0)
    facies[i] = ff
facies = facies.reshape(NR, NX, NY, NZ)
print(facies.shape)

kmin, kmax = np.log10(0.05), np.log10(3000)

perm = np.moveaxis(np.rot90(np.array(pd.read_csv('perm_1272_64x64x8.csv')).reshape(NX,NY,NZ,NR,order='F'), axes=(0,1)), -1, 0)
perm = MinMaxScaler((kmin,kmax)).fit_transform(perm.reshape(NR,-1)).reshape(perm.shape)
print('Perm: {} | min = {:.3f} = {:.3f} | max = {:.3f} = {:.3f}'.format(perm.shape, perm.min(), 10**perm.min(), perm.max(), 10**perm.max()))

facies_norm = MinMaxScaler((0.05, 5)).fit_transform(facies.round(0).reshape(NR,-1)).reshape(facies.shape)
print('Facies: {} | min = {:.3f} | max = {:.3f}'.format(facies_norm.shape, facies_norm.min(), facies_norm.max()))

perm_norm = np.log10(10**perm * facies_norm)
perm_norm = MinMaxScaler((kmin,kmax)).fit_transform(perm_norm.reshape(NR,-1)).reshape(perm_norm.shape)
print('Norm: {} | min = {:.3f} = {:.3f} | max = {:.3f} = {:.3f}'.format(perm_norm.shape, perm_norm.min(), 10**perm_norm.min(), perm_norm.max(), 10**perm_norm.max()))

np.savez('data_1272_64x64x8.npz', facies=facies, facies_norm=facies_norm, perm=perm, perm_norm=perm_norm)
sio.savemat('data_1272_64x64x8.mat', {'perm':perm_norm, 'facies':facies_norm})

###################################### 2D perms|facies ######################################
fnames = []
for root, dirs, files in os.walk('/mnt/e/MLTrainingImages'):
    for f in files:
        if f.endswith('.npy'):
            fnames.append(os.path.join(root, f))

facies = np.zeros((318,4,128,128))
for i in tqdm(range(len(fnames))):
    f = np.load(fnames[i]).reshape(256,256,128)
    f1 = resize(f[..., 31], (128,128), anti_aliasing=True, preserve_range=True)
    f2 = resize(f[..., 62], (128,128), anti_aliasing=True, preserve_range=True)
    f3 = resize(f[..., 93], (128,128), anti_aliasing=True, preserve_range=True)
    f4 = resize(f[..., 124], (128,128), anti_aliasing=True, preserve_range=True)
    facies[i] = np.stack([f1, f2, f3, f4], axis=0)

ff = facies.reshape(-1, 128, 128)
ff_norm = MinMaxScaler((0.1,5)).fit_transform(ff.reshape(-1, 128*128)).reshape(-1, 128, 128)
print('Facies: {} | min = {:.3f} | max = {:.3f}'.format(ff_norm.shape, ff_norm.min(), ff_norm.max()))

perm_all = np.rot90(np.moveaxis(np.array(pd.read_csv('perm_1272_128x128.csv')).reshape(128,128,-1,order='F'), -1, 0), k=1, axes=(1,2))
print('Perm: {} | min = {:.3f} = {:.3f} | max = {:.3f} = {:.3f}'.format(perm_all.shape, perm_all.min(), 10**perm_all.min(), perm_all.max(), 10**perm_all.max()))

perm_norm = np.log10(10**perm_all * ff_norm)
perm_norm = MinMaxScaler((-1.3, 3.4771212)).fit_transform(perm_norm.reshape(-1, 128*128)).reshape(-1, 128, 128)
print('Perm: {} | min = {:.3f} = {:.3f} | max = {:.3f} = {:.3f}'.format(perm_norm.shape, perm_norm.min(), 10**perm_norm.min(), perm_norm.max(), 10**perm_norm.max()))

np.savez('data_1272_128x128.npz', facies=ff, facies_norm=ff_norm, perm=perm_all, perm_norm=perm_norm)
sio.savemat('data_1272_128x128.mat', {'perm': perm_norm, 'facies': ff_norm})

##################################### make perms|facies #####################################
fnames = []
for root, dirs, files in os.walk('/mnt/e/MLTrainingImages'):
    for f in files:
        if f.endswith('.npy'):
            fnames.append(os.path.join(root, f))
facies = np.zeros((len(fnames), 32, 32, 32))
for i in tqdm(range(len(fnames))):
    f = np.load(fnames[i]).reshape(256,256,128)
    facies[i] = resize(f, (32,32,32), anti_aliasing=True, preserve_range=True).round().astype(np.uint8)

perm = np.rot90(np.moveaxis(np.array(pd.read_csv('perm_318_32x32x32.csv')).reshape(32,32,32,318,order='F'), -1, 0), k=1, axes=(1,2))
np.save('perm_318_32x32x32.npy', perm)

facies_norm = MinMaxScaler((0.1, 5)).fit_transform(facies.reshape(len(fnames),-1)).reshape(len(fnames),32,32,32)

mm = np.log10(10**perm * facies_norm)

mm = MinMaxScaler((-1.3, 3.4)).fit_transform(mm.reshape(len(fnames),-1)).reshape(len(fnames),32,32,32)
np.save('k_318_32x32x32.npy', mm)

############################################# 2D #############################################
N_REALIZATIONS = 300
NX, NY = 128, 128

x = y = range(NX)
model = gs.Gaussian(dim=2, var=1, len_scale=20, angles=0, nugget=1e-6)
perms = np.zeros((N_REALIZATIONS, NX, NY))
for i in tqdm(range(N_REALIZATIONS)):
    srf = gs.SRF(model)
    perms[i] = srf((x,y), mesh_type='structured')
perms_norm = (perms - perms.min()) / (perms.max() - perms.min())
np.savez('perms_300_128x128.npz', perms=perms, perms_norm=perms_norm)

grid = UnitGrid([NX, NY])
state = ScalarField.random_uniform(grid, -2, 2)
results = np.zeros((N_REALIZATIONS, NX, NY))
for i in tqdm(range(N_REALIZATIONS)):
    state.data = perms_norm[i]
    eq = DiffusionPDE(diffusivity=1.5, bc=[{'derivative':0},{'derivative':0}], noise=0.0)
    results[i] = eq.solve(state, t_range=100, dt=0.1, tracker=trackers.ConsistencyTracker()).data
np.save('results_300_128x128.npy', results)

############################################# 3D #############################################
N_REALIZATIONS = 300
NX, NY, NZ = 32, 32, 32

x = y = z = range(NX)
model = gs.Gaussian(dim=3, var=1, len_scale=[10,10,10], angles=0, nugget=1e-6)
perms = np.zeros((N_REALIZATIONS, NX, NY, NZ))
for i in tqdm(range(N_REALIZATIONS)):
    srf = gs.SRF(model)
    perms[i] = srf((x,y,z), mesh_type='structured')
perms_norm = (perms - perms.min()) / (perms.max() - perms.min())
np.savez('perms_300_32x32x32.npz', perms=perms, perms_norm=perms_norm)

grid = UnitGrid([NX, NY, NZ])
state = ScalarField.random_uniform(grid, -2, 2)
results = np.zeros((N_REALIZATIONS, NX, NY, NZ))
for i in tqdm(range(N_REALIZATIONS)):
    state.data = perms_norm[i]
    eq = DiffusionPDE(diffusivity=1.5, bc=[{'derivative':0},{'derivative':0},{'derivative':0}], noise=0.0)
    results[i] = eq.solve(state, t_range=100, dt=0.1, tracker=trackers.ConsistencyTracker()).data
np.save('results_300_32x32x32.npy', results)