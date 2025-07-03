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
nb_dir="/g/data/z00/yxs900/neuraloperators/sfno/curriculum_learning/lowRes"
sys.path.append(nb_dir)
from datasets import WBDataset
wdir="/g/data/z00/yxs900/neuraloperators/sfno/curriculum_learning/lowRes/experiments/05_LUCIE_rm_pos_embed/"
sys.path.append(wdir)
from torch_harmonics_local_v2 import *
#from LUCIE_inference import inference

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

def rollout_model(vdl, model, device="cuda"):
    asteps=len(vdl)
    pred = torch.zeros(asteps,5,128,256)
    targ = torch.zeros(asteps,5,128,256)

    model.eval()
    with torch.no_grad():
        for ii, data in enumerate(vdl, 0):
            #print(f"step {ii}, inp shape = {data[0].shape}, tar shape = {data[1].shape}")
            inp, tar = map(lambda x: x.to(device, dtype = torch.float32), data)
            if ii==0:
                prd = inp
            else:
                prd = torch.concatenate((inp[:,:3,:,:], prd),axis=1)

            prd = model(prd)
            pred[ii,:] = prd.cpu()
            targ[ii,:] = tar.cpu()

    return pred,targ

def rollout_model_tisr_forcing(vdl, model, device="cuda"):
    asteps=len(vdl)
    pred = torch.zeros(asteps,5,128,256)
    targ = torch.zeros(asteps,5,128,256)

    model.eval()
    with torch.no_grad():
        for ii, data in enumerate(vdl, 0):
            #print(f"step {ii}, inp shape = {data[0].shape}, tar shape = {data[1].shape}")
            inp, tar = map(lambda x: x.to(device, dtype = torch.float32), data)
            if ii==0:
                prd = inp
            else:
                prd = torch.concatenate((inp[:,:4,:,:], prd[:,1:,:,:]),axis=1)

            prd = model(prd)
            pred[ii,:] = prd.cpu()
            targ[ii,:] = tar.cpu()

    return pred,targ

def train_model(model, tdl, vdl, optimizer, scheduler=None,epoch0=0, nepochs=20, reg_rate=1e-3,loss_fn='l2'):
    infer_bias = 1e+80
    ibs = torch.zeros(1,nepochs)
    best_bias = 1e+80
    recall_count = 0
    acc_losses = []
    epoch_times = []
    ckpt_dir=f"{os.environ['PBS_O_WORKDIR']}/checkpoints/{os.environ['PBS_JOBID']}"
    for epoch in range(epoch0,nepochs):
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
        
        #optimizer.zero_grad()

        acc_loss = 0
        model.train()
        #batch_num = 0
        for ii, data in enumerate(tdl,0):
            optimizer.zero_grad()
            inp, tar = map(lambda x: x.to(device, dtype = torch.float32), data)
            prd = model(inp)

            loss = l2loss_sphere(prd, tar, relative=True)
            #loss_tp = torch.mean((prd[:,5:,:,:]-tar[:,5:,:,:])**2)
            #loss = loss_delta + loss_tp / tar.shape[1]

            if epoch > 150:
                #print(f"add spectral loss")
                #lat_index = np.r_[7:15, 32:40] # this is the index for 48*96
                lat_index = np.r_[21:43, 85:107] # this is the middle 1/3 index for 128*256

                out_fft = torch.mean(torch.abs(torch.fft.rfft(prd[:,:,lat_index,:],dim=3)),dim=2)
                target_fft = torch.mean(torch.abs(torch.fft.rfft(tar[:,:,lat_index,:],dim=3)),dim=2)
                loss_reg = reg_rate * torch.mean(torch.abs(out_fft - target_fft))
                loss = loss + loss_reg

            #optimizer.zero_grad()
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
        print(f'sec / nsample: {epoch_times[-1]/len(tdl)}')
        print(f'average training loss: {acc_losses[-1]/len(tdl)}')
        
        if epoch >= 60:
            rollout,gtruth = rollout_model_tisr_forcing(vdl, model,device=device)
            rollout_clim = torch.mean(rollout[1460:],dim=0)
            true_clim = torch.mean(gtruth[1460:],dim=0)
            clim_bias = torch.mean(torch.abs(rollout_clim - true_clim))
            ibs[0,epoch] = clim_bias
            if len(ibs>0)<=20:
                if ibs[0,60] == 0:
                    mask = ibs!=0
                    idxc = mask.float().argmax(dim=1)
                else: 
                    idxc = 60
                infer_bias = torch.mean(torch.tensor(ibs[0,idxc:epoch+1]))
            else:
                infer_bias = torch.mean(ibs[0,epoch-20:epoch+1])

            print(f'clim_bias: {clim_bias}')
            print(f'infer_bias: {infer_bias}')
            if clim_bias <= best_bias:
                print(f"new best clim_bias, save checkpoint")
                best_bias = clim_bias
                torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optim_state_dict":optimizer.state_dict(),"sch_state_dict":scheduler.state_dict()},f"{ckpt_dir}/lucie_{epoch}.pt")
                torch.save(model.state_dict(), f"{ckpt_dir}/regular_training_checkpoint.pth")
            elif epoch > 160 and clim_bias < best_bias*1.2:
                # save checkpoints even when it doesn't represent the best clim_bias. 
                torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optim_state_dict":optimizer.state_dict(),"sch_state_dict":scheduler.state_dict()},f"{ckpt_dir}/lucie_{epoch}.pt")

            # ?save checkpoints when the current bias is the best of the last 10 epochs.
            if  all(ibs[0,epoch-9:epoch+1]>0) and ibs[0,epoch] ==  min(ibs[0,epoch-9:epoch+1]):
                torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optim_state_dict":optimizer.state_dict(),"sch_state_dict":scheduler.state_dict()},f"{ckpt_dir}/lucie_{epoch}.pt")

            if epoch % 10 == 0:
                if torch.isnan(clim_bias): 
                    print("clim_bias is NaN, abort")
                    break
                else: 
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
                            print(f"recalled consectively {recall_count-1} times, abort")
                            break

if __name__ == "__main__":
    # set seed
    torch.manual_seed(333)
    torch.cuda.manual_seed(333)

    # get dataloader
    ddir = "/g/data/wb00/admin/testing/t_WeatherBench"
    dpth_str = f"{ddir}/wb2*.nc"
    cpth = f"{ddir}/constants_1.40625deg.nc"
    mpth = f"{ddir}/all_means.npy"
    spth = f"{ddir}/all_stds.npy"
    rpth = f"{ddir}/all_res.npy"
    avs = list(np.load(f"{ddir}/ordering.npy"))
    vs = ["tisr","t850","q850","u500","v500"] # add tp in the future
    idx = [avs.index(vi) for vi in vs]
    vc = ["lat2d","lon2d","lsm"]
    tsamples = 87000 # 10y hourly
    vsamples = 17520 # 2y hourly
    nsamples = tsamples
    samples = WBDataset(dpth_str,cpth,in_chans=vs, out_chans=vs, const_chans=vc, norm_paths=[mpth,spth,rpth],idx=idx, nsamples=nsamples)

    bs=8
    nworkers=10

    t = torch.utils.data.Subset(samples,list(range(0,nsamples,6)))
    v = torch.utils.data.Subset(samples,list(range(3,vsamples,6)))

    tdl = DataLoader(t, shuffle=True, batch_size=bs, num_workers=nworkers,drop_last=True)
    vdl = DataLoader(v, shuffle=False, batch_size=1, num_workers=nworkers,drop_last=True)

    print(f'number of training samples per epoch: {len(tdl)}')
    print(f'number of auto-regressive rollout: {len(vdl)}')
    
    n_in_channels = len(vs)+len(vc)
    n_out_channels = len(vs)

    grid='legendre-gauss'
    nlat = 128
    nlon = 256
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

    #model = SphericalFourierNeuralOperatorNet(params = {}, spectral_transform='sht', filter_type = "linear", operator_type='dhconv', img_shape=(48, 96),num_layers=8, in_chans=7, out_chans=6, scale_factor=1, embed_dim=72, activation_function="silu", big_skip=True, pos_embed="latlon", use_mlp=True,normalization_layer="instance_norm", hard_thresholding_fraction=hard_thresholding_fraction,mlp_ratio = 2.).to(device)
    # create the model without positional embedding by modifying the definition in the local torch_harmonics_local_v2.py file
    # pos_embed=False as a placeholder here

    model = SphericalFourierNeuralOperatorNet(params = {"data_grid":"equiangular"}, spectral_transform='sht', filter_type = "linear", operator_type='dhconv', img_shape=(nlat, nlon),num_layers=8, in_chans=n_in_channels, out_chans=n_out_channels, scale_factor=1, embed_dim=72, activation_function="silu", big_skip=True, pos_embed=False, use_mlp=True,normalization_layer="instance_norm", hard_thresholding_fraction=hard_thresholding_fraction,mlp_ratio = 2.).to(device)

    if len(sys.argv)==4:
        resume = True
    elif len(sys.argv)==3:
        resume=False
        ei=-1

    if resume == True:
        ckpt_path=sys.argv[3]
        sd = torch.load(ckpt_path,map_location=device,weights_only=False)
        ei = sd["epoch"]
        model.load_state_dict(sd["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    if resume == True:
        optimizer.load_state_dict(sd["optim_state_dict"])

    scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-5)
    if resume == True:
        scheduler.load_state_dict(sd["sch_state_dict"])
        print(f"resume from epoch {ei}: learning rate optim={optimizer.param_groups[0]['lr']}, scheduler={scheduler.get_last_lr()}")

    print(f"reg_rate={float(sys.argv[2])}")
    train_model(model, tdl, vdl, optimizer, scheduler=scheduler,epoch0=ei+1,nepochs=int(sys.argv[1]),reg_rate=float(sys.argv[2]))
    # torch.save(model.state_dict(), 'model.pth')
