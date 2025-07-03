#!/bin/bash
#PBS -l walltime=39:00:00
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

# correct equiangular grid sht, batch size =8, \alpha=1e-3.
nepochs=319
reg_rate=0.001
python3 train.py $nepochs $reg_rate 

# second attempt resuming from previous checkpoint using gaussian grid.
#nepochs=399
# 143943582 doesn't converge, unlike the shorter test. why? 
#reg_rate=0.001
# try a smaller rate
#reg_rate=0.0005
#wdir1="/g/data/z00/yxs900/neuraloperators/sfno/curriculum_learning/lowRes/experiments/06_LUCIE_no_pos_embed_wb1.4"
#checkpoint=$wdir1/checkpoints/143294337.gadi-pbs/lucie_128.pt
#python3 train.py $nepochs $reg_rate $checkpoint
