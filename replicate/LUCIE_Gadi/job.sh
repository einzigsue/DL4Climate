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

wdir="/g/data/z00/yxs900/neuraloperators/sfno/curriculum_learning/lowRes/experiments/03_LUCIE"
mkdir -p $wdir/checkpoints/$PBS_JOBID

nepochs=500
python3 LUCIE_train.py $nepochs
#python3 LUCIE_train_resume.py  $nepochs $wdir/checkpoints/136262312.gadi-pbs/lucie_100.pt

