import csv, torch, glob, os,sys,logging,bisect
from itertools import accumulate
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from netCDF4 import Dataset as ncDataset

class WBDataset(Dataset):
    def __init__(self, dpth_str, cpth, in_chans=["tisr","t850","q850","u500","v500"], out_chans=["tisr","t850","q850","u500","v500"], const_chans=["lat2d","lon2d","lsm"], norm_paths=None,iidx=None, oidx=None,dt=6, nsamples=100):
        self.files_paths = glob.glob(dpth_str)
        self.nsamples = nsamples
        self.dt = dt
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.const_chans = self._get_consts(cpth,const_chans)
        self.means = np.load(norm_paths[0])
        self.stds = np.load(norm_paths[1])
        self.res = np.load(norm_paths[2])
        if len(oidx) != self.res.shape[1]:
            sys.exit("the length of the output variable list doesn't match the shape of residual norm factor")

        self.iidx = iidx
        self.oidx = oidx

        self._get_files_stats()

    def __len__(self):
        return self.nsamples

    def _normalize_inp(self, img):
        img -= self.means[0,self.iidx,:,:]
        img /= self.stds[0,self.iidx,:,:]
        return torch.as_tensor(img)

    def _normalize_tar(self, img):
        img -= self.means[0,self.oidx,:,:]
        img /= self.stds[0,self.oidx,:,:]
        img /= self.res[0,:,:,:]
        return torch.as_tensor(img)

    def _minmax(self, img):
        return torch.as_tensor((img-img.min())/(img.max()-img.min()))

    def _get_consts(self,cfile,vs):
        with ncDataset(cfile, 'r') as ds:
            # min-max const
            return np.stack([self._minmax(ds[vi][:].data) for vi in vs])

    def _get_files_stats(self):
        self.files_paths.sort()
        self.years = [int(os.path.splitext(os.path.basename(x))[0][-4:]) for x in self.files_paths]
        self.n_years = len(self.files_paths)
        self.files = [None for _ in range(self.n_years)]
        self.nidx = [(8784 if (yi%4==0 and (yi % 100 != 0 or yi % 400 == 0)) else 8760)  for yi in self.years]
        self.idx0 = [0] + list(accumulate(self.nidx))[:-1]
        #with ncDataset(self.files_paths[0], 'r') as _f:
            #logging.info("Getting file stats from {}".format(self.files_paths[0]))
        #    self.n_samples_per_year = _f['time'].shape[0]

    def _open_file(self, year_idx):
        #print(f"open file for year {self.files_paths[year_idx]}")
        self.files[year_idx] = ncDataset((self.files_paths[year_idx]),'r')
        #self.n_samples_per_year = self.files[year_idx]['time'].shape[0] 

    def __getitem__(self, global_idx):
        #year_idx = int(global_idx / self.n_samples_per_year)  # which year
        #local_idx = int(global_idx % self.n_samples_per_year) # which sample in that year 
        year_idx = bisect.bisect_right(self.idx0, global_idx) - 1
        local_idx = global_idx - self.idx0[year_idx]

        # open image file
        if self.files[year_idx] is None:
            self._open_file(year_idx)

        #step = self.dt # time step

        # boundary conditions to ensure we don't pull data that is not in a specific year
        #local_idx = local_idx % (self.n_samples_per_year - step)
        #if local_idx < step:
        #    local_idx += step

        if local_idx + self.dt < self.nidx[year_idx]:
            year_idx2 = year_idx
            local_idx2 = local_idx + self.dt
        else:
            year_idx2 = year_idx + 1
            local_idx2 = global_idx + self.dt - self.idx0[year_idx2]
            if self.files[year_idx2] is None:
                self._open_file(year_idx2)

        # pre-process and get the image fields
        ins  = np.stack([self.files[year_idx][vi][local_idx,:,:].data for vi in self.in_chans])
        outs = np.stack([self.files[year_idx2][vi][local_idx2,:,:].data for vi in self.out_chans])
        inp, tar = np.concatenate((self.const_chans, self._normalize_inp(ins)),axis=0), self._normalize_tar(outs)

        return inp, tar
