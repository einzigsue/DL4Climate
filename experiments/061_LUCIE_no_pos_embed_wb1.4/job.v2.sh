#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -q gpuvolta
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=95GB
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -l storage=scratch/z00+gdata/z00+gdata/dk92+gdata/pp66+gdata/wb00
#PBS -P ui41

module purge
module use /g/data/dk92/apps/Modules/modulefiles/
module load NCI-ai-ml/24.08 

wdir="/g/data/z00/yxs900/neuraloperators/sfno/curriculum_learning/lowRes/experiments/061_LUCIE_no_pos_embed_wb1.4"
mkdir -p $wdir/checkpoints/$PBS_JOBID

nepochs=220
reg_rate=0.001
#python3 train.v2.py $nepochs $reg_rate 

# fixed the bug in the rollout
# continue from epoch 60
python3 train.v2.py $nepochs $reg_rate ${wdir}/checkpoints/144980943.gadi-pbs/lucie_60.pt

