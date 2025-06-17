import numpy as np
import torch
from torch_harmonics_local import *
from tqdm import tqdm



def normalize(data, diff=False):
    data_mean = np.mean(data)
    data_std = np.std(data)
    if diff:
        data_norm = data / data_std
    else:
        data_norm = (data - data_mean)/ data_std
    
    return data_norm, data_mean, data_std





data = np.load("era5_T30_regridded.npz")

vars = ['temperature', 'humidity', 'u_wind', 'v_wind', 'surface_pressure', 'precipitation', 'tisr', 'orography']
raw_vars = ['temperature', 'humidity', 'u_wind', 'v_wind', 'surface_pressure', 'tisr', 'orography']
diff_vars = ['temperature', 'humidity', 'u_wind', 'v_wind', 'surface_pressure']
diag_vars = ['precipitation']

data_inp = np.zeros((len(data['temperature'])-1, len(vars)-1, 48, 96))
data_tar = np.zeros((len(data['temperature'])-1, len(vars)-2, 48, 96))
raw_means = np.zeros(len(vars))
raw_stds = np.zeros(len(vars))
diag_means = np.zeros(len(diag_vars))
diag_stds = np.zeros(len(diag_vars))
diff_means = np.zeros(len(diff_vars))
diff_stds = np.zeros(len(diff_vars))
true_clim = np.zeros((len(vars)-2, 48, 96))
for idx, var in tqdm(enumerate(vars[:6])):
    true_clim[idx] = np.mean(data[var][:2920])

for idx, var in tqdm(enumerate(raw_vars)):
    data_inp[:,idx,:,:], raw_means[idx], raw_stds[idx] = normalize(data[var][:-1], diff=False)
    if var in diff_vars:
        data_tar[:,idx,:,:], diff_means[idx], diff_stds[idx] = normalize(data[var][1:]-data[var][:-1], diff=True)

for idx, var in tqdm(enumerate(diag_vars)):
    data_tar[:,-1,:,:], diag_means[idx], diag_stds[idx] = normalize(np.log(data[var][1:]/1e-2+1), diff=False)


np.savez("era5_T30_preprocessed.npz", data_inp=data_inp, data_tar=data_tar, raw_means=raw_means, raw_stds=raw_stds, diag_means=diag_means, diag_stds=diag_stds, diff_means=diff_means, diff_stds=diff_stds)

