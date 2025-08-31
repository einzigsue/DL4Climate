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
#nb_dir="/g/data/z00/yxs900/neuraloperators/sfno/curriculum_learning/lowRes"
#sys.path.append(nb_dir)
# load the dataset in the current dir
from datasets_res import WBDataset
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

def rollout_model_tisr_forcing(vdl, model, device="cuda"):
    asteps=len(vdl)
    c,h,w = vdl.dataset[0][1].shape
    pred = torch.zeros(asteps,c,h,w)
    targ = torch.zeros(asteps,c,h,w)

    res = torch.tensor(vdl.dataset.dataset.res).to(device)

    model.eval()
    with torch.no_grad():
        for ii, data in enumerate(vdl, 0):
            #print(f"step {ii}, inp shape = {data[0].shape}, tar shape = {data[1].shape}")
            inp, tar = map(lambda x: x.to(device, dtype = torch.float32), data)
            if ii==0:
                prd = inp
            else:
                prd = torch.concatenate((inp[:,:4,:,:], prd[:,:-1,:,:]*res[0,:-1,:,:]),axis=1)

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
        
        if epoch < 79:
            if scheduler is not None:
                scheduler.step()
                print(f'using scheduler: current learning rate = {scheduler.get_lr()}')
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 5e-6
                
            print(f"current learning rate = {optimizer.param_groups[0]['lr']}")
        
        #optimizer.zero_grad()

        acc_loss = 0
        model.train()
        #batch_num = 0
        for ii, data in enumerate(tdl,0):
            inp, tar = map(lambda x: x.to(device, dtype = torch.float32), data)
            prd = model(inp)

            loss = l2loss_sphere(prd, tar, relative=True)
            #loss_tp = torch.mean((prd[:,5:,:,:]-tar[:,5:,:,:])**2)
            #loss = loss_delta + loss_tp / tar.shape[1]

            if epoch > 80:
                #print(f"add spectral loss")
                #lat_index = np.r_[7:15, 32:40] # this is the index for 48*96
                lat_index = np.r_[21:43, 85:107] # this is the middle 1/3 index for 128*256

                out_fft = torch.mean(torch.abs(torch.fft.rfft(prd[:,:,lat_index,:],dim=3)),dim=2)
                target_fft = torch.mean(torch.abs(torch.fft.rfft(tar[:,:,lat_index,:],dim=3)),dim=2)
                loss_reg = reg_rate * torch.mean(torch.abs(out_fft - target_fft))
                loss = loss + loss_reg

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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
        
        torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optim_state_dict":optimizer.state_dict(),"sch_state_dict":scheduler.state_dict()},f"{ckpt_dir}/lucie_{epoch}.pt")

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

            # ?save checkpoints when the current bias is the best of the last 20 epochs.
            if  all(ibs[0,epoch-19:epoch+1]>0) and ibs[0,epoch] ==  min(ibs[0,epoch-19:epoch+1]):
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
    #torch.manual_seed(333)
    #torch.cuda.manual_seed(333)

    # print torch seed
    print(torch.initial_seed())
    # get dataloader
    ddir = "/g/data/wb00/admin/testing/t_WeatherBench"
    dpth_str = f"{ddir}/a_wb2*.nc"
    cpth = f"{ddir}/constants_1.40625deg.nc"
    didir = "/g/data/z00/yxs900/neuraloperators/sfno/curriculum_learning/lowRes/experiments/061_LUCIE_no_pos_embed_wb1.4/datasets"
    mpth = f"{didir}/all_means.npy"
    spth = f"{didir}/all_stds.npy"
    rpth = f"{didir}/all_res0.npy"
    #avs = list(np.load(f"{didir}/ordering.npy"))
    avs = ["tisr","t1000","t850","q1000","q850","u500","u200","v500","v200","sp","log_tp"]
    vs_in = ["tisr","t850","q850","u500","v500","sp"]
    vs_out = ["t850","q850","u500","v500","sp","log_tp"]
    iidx = [avs.index(vi) for vi in vs_in]
    oidx = [avs.index(vi) for vi in vs_out]
    vc = ["lat2d","lon2d","orography"]
    tsamples = 87000 # 10y hourly
    vsamples = 17520 # 2y hourly
    nsamples = tsamples
    samples = WBDataset(dpth_str,cpth,in_chans=vs_in, out_chans=vs_out, const_chans=vc, norm_paths=[mpth,spth,rpth],iidx=iidx, oidx=oidx, nsamples=nsamples)

    bs=8
    # bs=4 uses only 20% of the memory limit 32GB, try boost to 8 or 16 to 
    # 1. reduce optimizer steps
    # 2. utilize GPU memory
    #bs=8
    nworkers=10

    t = torch.utils.data.Subset(samples,list(range(0,nsamples,6)))
    v = torch.utils.data.Subset(samples,list(range(3,vsamples,6)))

    tdl = DataLoader(t, shuffle=True, batch_size=bs, num_workers=nworkers,drop_last=True)
    vdl = DataLoader(v, shuffle=False, batch_size=1, num_workers=nworkers,drop_last=True)

    print(f'number of training samples per epoch: {len(tdl)}')
    print(f'number of auto-regressive rollout: {len(vdl)}')
    
    n_in_channels = len(vs_in)+len(vc)
    n_out_channels = len(vs_out)

    nlat = 128
    nlon = 256
    hard_thresholding_fraction = 0.9
    cost, quad_weights = legendre_gauss_weights(nlat, -1, 1)
    quad_weights = (torch.as_tensor(quad_weights).reshape(-1, 1)).to(device)

    #model = SphericalFourierNeuralOperatorNet(params = {}, spectral_transform='sht', filter_type = "linear", operator_type='dhconv', img_shape=(48, 96),num_layers=8, in_chans=7, out_chans=6, scale_factor=1, embed_dim=72, activation_function="silu", big_skip=True, pos_embed="latlon", use_mlp=True,normalization_layer="instance_norm", hard_thresholding_fraction=hard_thresholding_fraction,mlp_ratio = 2.).to(device)
    # create the model without positional embedding by modifying the definition in the local torch_harmonics_local_v2.py file
    # pos_embed=False as a placeholder here

    model = SphericalFourierNeuralOperatorNet(params = {"data_grid":"equiangular"}, spectral_transform='sht', filter_type = "linear", operator_type='dhconv', img_shape=(nlat, nlon),num_layers=8, in_chans=n_in_channels, out_chans=n_out_channels, scale_factor=1, embed_dim=128, activation_function="silu", big_skip=True, pos_embed=False, use_mlp=True,normalization_layer="instance_norm", hard_thresholding_fraction=hard_thresholding_fraction,mlp_ratio = 2.).to(device)

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

    scheduler = CosineAnnealingLR(optimizer, T_max=80, eta_min=5e-5)
    if resume == True:
        scheduler.load_state_dict(sd["sch_state_dict"])
        print(f"resume from epoch {ei}: learning rate optim={optimizer.param_groups[0]['lr']}, scheduler={scheduler.get_last_lr()}")

    print(f"reg_rate={float(sys.argv[2])}")
    train_model(model, tdl, vdl, optimizer, scheduler=scheduler,epoch0=ei+1,nepochs=int(sys.argv[1]),reg_rate=float(sys.argv[2]))
    # torch.save(model.state_dict(), 'model.pth')
