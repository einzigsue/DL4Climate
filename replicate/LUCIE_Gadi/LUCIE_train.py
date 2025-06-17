from math import ceil, sqrt
from functools import partial
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from typing import Any, Tuple
# import torch_harmonics as th
# import torch_harmonics.distributed as thd

# from torch_harmonics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.utils.checkpoint import checkpoint
from torch.cuda import amp
import math
#from tqdm import tqdm

import torch

from torch.utils.data import Dataset, TensorDataset, DataLoader


from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, StepLR

import sys,os,time
wdir="/g/data/z00/yxs900/neuraloperators/sfno/curriculum_learning/lowRes/experiments/03_LUCIE/"
sys.path.append(wdir)
from torch_harmonics_local import *
from LUCIE_inference import inference

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(0)


def integrate_grid(ugrid, dimensionless=False, polar_opt=0):

    dlon = 2 * torch.pi / nlon
    radius = 1 if dimensionless else radius
    if polar_opt > 0:
        out = torch.sum(ugrid[..., polar_opt:-polar_opt, :] * quad_weights[polar_opt:-polar_opt] * dlon * radius**2, dim=(-2, -1))
    else:
        out = torch.sum(ugrid * quad_weights * dlon * radius**2, dim=(-2, -1))
    return out

def l2loss_sphere(prd, tar, relative=False, squared=True):
    loss = integrate_grid((prd - tar)**2, dimensionless=True).sum(dim=-1)
    if relative:
        loss = loss / integrate_grid(tar**2, dimensionless=True).sum(dim=-1)

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss


def train_model(model, tdl, optimizer, scheduler=None, nepochs=20, loss_fn='l2'):
    infer_bias = 1e+80
    ibs = torch.zeros(1,nepochs)
    best_bias = 1e+80
    recall_count = 0
    acc_losses = []
    epoch_times = []
    ckpt_dir=f"{os.environ['PBS_O_WORKDIR']}/checkpoints/{os.environ['PBS_JOBID']}"
    for epoch in range(nepochs):
        tstamp=time.strftime("%H:%M:%S",time.localtime())
        print(f'--------------------------------------------------------------------------------')
        print(f"{tstamp}: epoch {epoch} start")
        epoch_start = time.time()
        
        if epoch < 149:
            if scheduler is not None:
                scheduler.step()
                print(f'using scheduler: current learning rate = {scheduler.get_lr()}')
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-6
                
            print(f"current learning rate = {optimizer.param_groups[0]['lr']}")
        
        optimizer.zero_grad()

        acc_loss = 0
        model.train()
        #batch_num = 0
        for inp, tar in tdl:
            #batch_num += 1
            #loss = 0

            inp = inp.to(device)
            tar = tar.to(device)
            prd = model(inp)

            loss_delta = l2loss_sphere(prd[:,:5,:,:], tar[:,:5,:,:], relative=True)
            loss_tp = torch.mean((prd[:,5:,:,:]-tar[:,5:,:,:])**2)
            loss = loss_delta + loss_tp / tar.shape[1]

            if epoch > 150:
                #print(f"add spectral loss")
                lat_index = np.r_[7:15, 32:40]
                out_fft = torch.mean(torch.abs(torch.fft.rfft(prd[:,:,lat_index,:],dim=3)),dim=2)
                target_fft = torch.mean(torch.abs(torch.fft.rfft(tar[:,:,lat_index,:],dim=3)),dim=2)
                loss_reg = 0.05 * torch.mean(torch.abs(out_fft - target_fft))
                loss = loss + loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc_loss += loss.item()* inp.size(0)
        
        acc_losses.append(acc_loss / len(tdl))
        #if scheduler is not None:
        #    scheduler.step()

        epoch_times.append(time.time() - epoch_start)
        tstamp=time.strftime("%H:%M:%S",time.localtime())
        print(f'{tstamp}: Epoch {epoch} summary:')
        print(f'time taken: {epoch_times[-1]}')
        print(f'nsamples / sec: {epoch_times[-1]/len(tdl)}')
        print(f'average training loss: {acc_losses[-1]/len(tdl)}')
       
        #if epoch % 50 == 0:
        #    torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optim_state_dict":optimizer.state_dict(),"sch_state_dict":scheduler.state_dict()},f"{ckpt_dir}/lucie_{epoch}.pt")
        
        if epoch >= 60:
            rollout_steps = 2920
            rollout = torch.tensor(inference(model, rollout_steps, data_inp[0:1].to(device), data_inp[:1460,-2:].to(device), 1, prog_means, prog_stds, diag_means, diag_stds, diff_stds)).to(device)
            rollout_clim = torch.mean(rollout[1460:],dim=0)
            clim_bias = torch.mean(torch.abs(rollout_clim - true_clim))
            ibs[0,epoch] = clim_bias
            if len(ibs>0)<=20:
                infer_bias = torch.mean(torch.tensor(ibs[0,60:epoch+1]))
            else:
                infer_bias = torch.mean(ibs[0,epoch-20:epoch+1])

            print(f'clim_bias: {clim_bias}')
            print(f'infer_bias: {infer_bias}')
            if clim_bias <= best_bias:
                print(f"new best clim_bias, save checkpoint")
                best_bias = clim_bias
                torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optim_state_dict":optimizer.state_dict(),"sch_state_dict":scheduler.state_dict()},f"{ckpt_dir}/lucie_{epoch}.pt")
                torch.save(model.state_dict(), f"{ckpt_dir}/regular_training_checkpoint.pth")
            
            if epoch % 10 == 0:
                if ~torch.isnan(clim_bias): 
                    if clim_bias <= infer_bias:
                        #print(f"clim_bias <= {infer_bias}, save checkpoint")
                        #infer_bias = clim_bias
                        #torch.save(model.state_dict(), f"{ckpt_dir}/regular_training_checkpoint.pth")
                        recall_count = 0
                    else:
                        print(f"clim_bias > {infer_bias}, recall from latest checkpoint")
                        state_pth = torch.load(f"{ckpt_dir}/regular_training_checkpoint.pth")
                        model.load_state_dict(state_pth)
                        recall_count += 1
                        if recall_count > 3:
                            break

if __name__ == "__main__":
    data = load_data(f"{wdir}/LUCIE_fix/era5_T30_regridded.npz")[...,:6]
    true_clim = torch.tensor(np.mean(data, axis=0)).to(device).permute(2,0,1)

    data = np.load(f"{wdir}/LUCIE_fix/era5_T30_preprocessed.npz")     # standardized data with mean and stds generated from dataset_generator.py
    data_inp = torch.tensor(data["data_inp"],dtype=torch.float32)     # input data 
    data_tar = torch.tensor(data["data_tar"],dtype=torch.float32)
    raw_means = torch.tensor(data["raw_means"],dtype=torch.float32).reshape(1,-1,1,1).to(device)
    raw_stds = torch.tensor(data["raw_stds"],dtype=torch.float32).reshape(1,-1,1,1).to(device)
    prog_means = raw_means[:,:5]
    prog_stds = raw_stds[:,:5]
    diag_means = torch.tensor(data["diag_means"],dtype=torch.float32).reshape(1,-1,1,1).to(device)
    diag_stds = torch.tensor(data["diag_stds"],dtype=torch.float32).reshape(1,-1,1,1).to(device)
    diff_means = torch.tensor(data["diff_means"],dtype=torch.float32).reshape(1,-1,1,1).to(device)
    diff_stds = torch.tensor(data["diff_stds"],dtype=torch.float32).reshape(1,-1,1,1).to(device)

    ntrain = 16000
    #nval = 100

    train_set = TensorDataset(data_inp[:ntrain],data_tar[:ntrain])
    #val_set = TensorDataset(data_inp[ntrain:ntrain+nval],data_tar[ntrain:ntrain+nval])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    #val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

    grid='legendre-gauss'
    nlat = 48
    nlon = 96
    hard_thresholding_fraction = 0.9
    lmax = ceil(nlat / 1)
    mmax = lmax
    modes_lat = int(nlat * hard_thresholding_fraction)
    modes_lon = int(nlon//2 * hard_thresholding_fraction)
    modes_lat = modes_lon = min(modes_lat, modes_lon)
    sht = RealSHT(nlat, nlon, lmax=modes_lat, mmax=modes_lon, grid=grid, csphase=False)
    radius=6.37122E6
    cost, quad_weights = legendre_gauss_weights(nlat, -1, 1)
    quad_weights = (torch.as_tensor(quad_weights).reshape(-1, 1)).to(device)

    model = SphericalFourierNeuralOperatorNet(params = {}, spectral_transform='sht', filter_type = "linear", operator_type='dhconv', img_shape=(48, 96),
                    num_layers=8, in_chans=7, out_chans=6, scale_factor=1, embed_dim=72, activation_function="silu", big_skip=True, pos_embed="latlon", use_mlp=True,
                                            normalization_layer="instance_norm", hard_thresholding_fraction=hard_thresholding_fraction,
                                            mlp_ratio = 2.).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-5)
    train_model(model, train_loader, optimizer, scheduler=scheduler, nepochs=int(sys.argv[1]), loss_fn = "l2")
    # torch.save(model.state_dict(), 'model.pth')
