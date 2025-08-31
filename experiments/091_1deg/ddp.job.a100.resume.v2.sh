#!/bin/bash
#PBS -l walltime=47:59:00
#PBS -q dgxa100
#PBS -l ncpus=64
#PBS -l ngpus=4
#PBS -l mem=1000GB
#PBS -l jobfs=1400GB
#PBS -l wd
#PBS -l storage=scratch/z00+gdata/z00+gdata/dk92+gdata/pp66+gdata/wb00
#PBS -P ui41

module purge
#module use /g/data/dk92/apps/Modules/modulefiles/
#module load NCI-ai-ml/24.08 

module use /g/data/pp66/apps/Modules/modulefiles/
module load nci-ai-ml/25.07

wdir="/g/data/z00/yxs900/neuraloperators/sfno/curriculum_learning/lowRes/experiments/091_1deg"
mkdir -p $wdir/checkpoints/$PBS_JOBID

#
nepochs=150
reg_rate=0.001
hname=$(hostname)
mpirun -np $PBS_NGPUS --map-by numa:SPAN --bind-to numa -x MASTER_ADDR=$hname -x MASTER_PORT=12355 python3 train_ddp.v2.a100.py $nepochs $reg_rate

