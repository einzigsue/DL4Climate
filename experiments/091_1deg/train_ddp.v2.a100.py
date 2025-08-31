from math import ceil, sqrt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys,os,time
nb_dir="/g/data/z00/yxs900/neuraloperators/sfno/curriculum_learning/lowRes/experiments/09_1deg"
sys.path.append(nb_dir)
from datasets_res import WBDataset
wdir="/g/data/z00/yxs900/neuraloperators/sfno/curriculum_learning/lowRes/experiments/05_LUCIE_rm_pos_embed/"
sys.path.append(wdir)
from torch_harmonics_local_v2 import *
#from LUCIE_inference import inference

def integrate_grid(ugrid):
    cost, quad_weights = legendre_gauss_weights(ugrid.shape[-2], -1, 1)
    quad_weights = torch.as_tensor(quad_weights).reshape(-1, 1).cuda()
    dlon = 2 * torch.pi / ugrid.shape[-1]
    out = torch.sum(ugrid * quad_weights * dlon, dim=(-2, -1))
    return out

def l2loss_sphere(prd, tar, relative=False, squared=True):
    loss = integrate_grid((prd - tar)**2).sum(dim=-1)
    if relative:
        loss = loss / integrate_grid(tar**2).sum(dim=-1)

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss

def rollout_model_forcing(samples, model, device="cuda"):
    # rollout for 2 years
    rollout_steps = 2920
    # randomly generate the starting index 
    # make sure the end index within the length of the samples
    start_idx = torch.randint(0,len(samples)-6*rollout_steps,(1,))
    v = torch.utils.data.Subset(samples,list(range(start_idx,start_idx+6*rollout_steps,6)))
    vdl = DataLoader(v, shuffle=False, batch_size=1, num_workers=10,drop_last=True)

    asteps=len(vdl)
    c,h,w = vdl.dataset[0][1].shape
    pred_t = torch.zeros(asteps,c)
    targ_t = torch.zeros(asteps,c)

    res = torch.tensor(vdl.dataset.dataset.res).to(device)
    model.eval()
    with torch.no_grad():
        for ii, data in enumerate(vdl, 0):
            #print(f"step {ii}, inp shape = {data[0].shape}, tar shape = {data[1].shape}")
            inp, tar = map(lambda x: x.to(device, dtype = torch.float32), data)
            if ii==0:
                prd = inp
            else:
                #the first 5 channels are lat2d, lon2d, orography, tisr, and msdwswrf are used as forcing.
                # the last output channel is diagnostic, leavign log_tp out
                #prd = torch.concatenate((inp[:,:5,:,:], prd[:,:-1,:,:]),axis=1)
                prd = torch.concatenate((inp[:,:6,:,:], prd[:,1:-1,:,:]*res[0,1:-1,:,:]),axis=1)

            prd = model(prd)
            pred_t[ii,:] = torch.mean(prd,dim=(-1,-2))
            targ_t[ii,:] = torch.mean(tar,dim=(-1,-2))

    return pred_t.clone().detach().cpu(), targ_t.clone().detach().cpu()

def rollout_score(pp,tt):
    rollout_clim = torch.mean(pp,dim=0)
    true_clim = torch.mean(tt,dim=0)
    scores = torch.abs(rollout_clim - true_clim)
    return torch.tensor(scores)

def ddp_reduce_score(local_score: float, op=dist.ReduceOp.SUM) -> float:
    """Reduces a scalar score across all DDP ranks and synchronizes using a barrier."""
    # Convert the score to a CUDA tensor (assuming model/device is on GPU)
    score_tensor = torch.tensor([local_score], dtype=torch.float32, device=torch.cuda.current_device())
    # Reduce the score across all ranks (e.g., sum, mean, min, max)
    dist.all_reduce(score_tensor, op=op)
    # Synchronize all ranks: no rank proceeds until all have completed reduction
    dist.barrier()
    return torch.tensor(score_tensor.item())


def train_model(rank,model, tdl, samples, sampler, optimizer, scheduler=None,epoch0=0, nepochs=20, reg_rate=1e-3,device="cuda"):
    infer_bias = 1e+80
    clr=0
    ibs = torch.zeros(1,nepochs)
    best_bias = 1e+80
    recall_count = 0
    acc_losses = []
    epoch_times = []
    ckpt_dir=f"{os.environ['PBS_O_WORKDIR']}/checkpoints/{os.environ['PBS_JOBID']}"
    for epoch in range(epoch0,nepochs):
        if rank==0:
            tstamp=time.strftime("%H:%M:%S",time.localtime())
            print(f'--------------------------------------------------------------------------------')
            print(f"{tstamp}: epoch {epoch} start")
            epoch_start = time.time()
        
        if epoch < 59:
            if scheduler is not None:
                scheduler.step()
                clr = scheduler.get_last_lr()
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-6
                clr = optimizer.param_groups[0]['lr']
            
        sampler.set_epoch(epoch)
        acc_loss = 0
        model.train()
        for ii, data in enumerate(tdl,0):
            inp, tar = map(lambda x: x.to(device, dtype = torch.float32), data)
            prd = model(inp)

            loss = l2loss_sphere(prd, tar, relative=True)

            if epoch > 60:
            #if epoch > 51:
                #print(f"add spectral loss")
                #lat_index = np.r_[7:15, 32:40] # this is the index for 48*96
                lat_index = np.r_[30:59, 121:150] # this is the middle 1/3 index for 181*360

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

        if rank==0:
            print(f"current learning rate = {clr}")
            epoch_times.append(time.time() - epoch_start)
            tstamp=time.strftime("%H:%M:%S",time.localtime())
            print(f'{tstamp}: Epoch {epoch} summary:')
            print(f'time taken: {epoch_times[-1]}')
            print(f'sec / nsample: {epoch_times[-1]/len(tdl)}')
            print(f'average training loss: {acc_losses[-1]/len(tdl)}')
        
        if epoch >= 30:
            rollout,gtruth = rollout_model_forcing(samples, model,device=device)
            local_scores = rollout_score(rollout[1460:2920],gtruth[1460:2920])
            local_clim_bias =  local_scores.mean()
            clim_bias = ddp_reduce_score(local_clim_bias,op=dist.ReduceOp.SUM)

            ibs[0,epoch] = clim_bias
            if len(ibs>0)<=20:
                if ibs[0,30] == 0:
                    mask = ibs!=0
                    idxc = mask.float().argmax(dim=1)
                else: 
                    idxc = 30
                infer_bias = torch.mean(torch.tensor(ibs[0,idxc:epoch+1]))
            else:
                infer_bias = torch.mean(ibs[0,epoch-20:epoch+1])

            #print(f"rank {rank}: epoch {epoch}, current bias {clim_bias}, best bias {best_bias}, infer bias {infer_bias}")

            if rank==0:
                print(f'bias by var, {local_scores}')
                print(f'clim_bias: {clim_bias}')
                print(f'infer_bias: {infer_bias}')

            if clim_bias <= best_bias:
                best_bias = clim_bias
                if rank==0:
                    print(f"new best clim_bias, save checkpoint")
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
                        print(f"clim_bias > {infer_bias}, rank {rank} recall from latest checkpoint")
                        state_pth = torch.load(f"{ckpt_dir}/regular_training_checkpoint.pth")
                        model.load_state_dict(state_pth)
                        recall_count += 1
                        if recall_count > 3:
                            print(f"recalled consective {recall_count-1} times, abort")
                            break

def train(rank,world_size,jobid):
    dist.init_process_group("nccl",rank=rank, world_size=world_size)
    print(f"Running DDP example on rank {rank} using torch seed {torch.initial_seed()}")

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device.index)

    # get dataloader
    ddir = "/g/data/wb00/admin/testing/t_WeatherBench_1deg"
    dpth_str = f"{ddir}/wb_gg_1deg_2*.nc"
    cpth = f"{ddir}/gg_const_1deg.nc"
    mpth = f"{ddir}/all_means.npy"
    spth = f"{ddir}/all_stds.npy"
    rpth = f"{ddir}/all_res.npy"
    avs = list(np.load(f"{ddir}/ordering.npy"))
    vs_in=['tisr','msdwswrf','mtnlwrf','2t','skt','10u','10v','sp','msl','u50','u200','u850','v50','v200','v850','t50','t200','t500','t850','t1000','q500','q850','q1000','z150','z500','z850']
    vs_out=['mtnlwrf','2t','skt','10u','10v','sp','msl','u50','u200','u850','v50','v200','v850','t50','t200','t500','t850','t1000','q500','q850','q1000','z150','z500','z850','log_tp']
    iidx = [avs.index(vi) for vi in vs_in]
    oidx = [avs.index(vi) for vi in vs_out]
    vc = ["lat2d","lon2d","oro"]
    tsamples = 87000 # 10y hourly
    vsamples = 17520 # 2y hourly
    nsamples = tsamples
    samples = WBDataset(dpth_str,cpth,in_chans=vs_in, out_chans=vs_out, const_chans=vc, norm_paths=[mpth,spth,rpth],iidx=iidx,oidx=oidx,nsamples=nsamples)

    bs=3
    nworkers=10

    t = torch.utils.data.Subset(samples,list(range(0,nsamples,6)))
    #v = torch.utils.data.Subset(samples,list(range(3,vsamples,6)))

    sampler = DistributedSampler(t)
    tdl = DataLoader(t, sampler=sampler, batch_size=bs, num_workers=nworkers,drop_last=True)
    print(f'number of training samples per epoch: {len(tdl)}')
    
    n_in_channels = len(vs_in)+len(vc)
    n_out_channels = len(vs_out)

    nlat,nlon = t[0][0].shape[-2:]
    hard_thresholding_fraction = 0.9

    model = SphericalFourierNeuralOperatorNet(params = {}, spectral_transform='sht', filter_type = "linear", operator_type='dhconv', img_shape=(nlat, nlon),num_layers=12, in_chans=n_in_channels, out_chans=n_out_channels, scale_factor=1, embed_dim=256, activation_function="silu", big_skip=True, pos_embed=False, use_mlp=True,normalization_layer="layer_norm", hard_thresholding_fraction=hard_thresholding_fraction,mlp_ratio = 2.).to(device)

    ddp_model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=1e-4, weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-5)

    if len(sys.argv)==3:
        #resume=False
        ei=-1
    elif len(sys.argv)==4:
        #resume=True
        #map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        #checkpoint=torch.load(str(sys.argv[3]), map_location=map_location, weights_only=False)
        checkpoint = torch.load(str(sys.argv[3]), map_location=device)
        ei = checkpoint['epoch']
        ddp_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        scheduler.load_state_dict(checkpoint['sch_state_dict'])
    
    train_model(rank, ddp_model, tdl, samples, sampler, optimizer, scheduler=scheduler,epoch0=ei+1,nepochs=int(sys.argv[1]),reg_rate=float(sys.argv[2]),device=device)
    dist.destroy_process_group()


if __name__ == "__main__":
    jobid=os.environ['PBS_JOBID']
    rank=int(os.environ['OMPI_COMM_WORLD_RANK'])
    world_size=int(os.environ['PBS_NGPUS'])
    print(f"rank={rank} out of world_size={world_size}")
    train(rank,world_size,jobid)

