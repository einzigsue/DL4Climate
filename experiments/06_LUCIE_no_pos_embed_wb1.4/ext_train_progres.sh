#!/bin/bash

jobid=$1
inp=$2
src=${inp}.sh.o${jobid}
grep "clim_bias: " ${src} | awk '{print $2}' > clim.bias.o${jobid}.csv
grep loss ${src} | awk '{print $4}' > average.training.loss.o${jobid}.csv
