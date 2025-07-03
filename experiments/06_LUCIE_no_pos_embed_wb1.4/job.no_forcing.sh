#!/bin/bash
#PBS -l walltime=40:00:00
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

wdir="/g/data/z00/yxs900/neuraloperators/sfno/curriculum_learning/lowRes/experiments/06_LUCIE_no_pos_embed_wb1.4"
mkdir -p $wdir/checkpoints/$PBS_JOBID

#nepochs=230
# this one doesn't do well in terms of rollout sample continuity in time. 
# missing samples at the end of leap years may cause problem
#python3 LUCIE_train_roll_resume.py $nepochs $wdir/checkpoints/141695290.gadi-pbs/lucie_14.pt

#with updated WBDataset making sure the continuity in rollout.
#python3 LUCIE_train_roll_resume.py $nepochs $wdir/checkpoints/141702959.gadi-pbs//lucie_60.pt

# resume from above attempt to 399 epochs
#nepochs=399
#python3 LUCIE_train_roll_resume.py $nepochs $wdir/checkpoints/141790879.gadi-pbs/lucie_125.pt

# save more often.
nepochs=399
python3 LUCIE_train_roll_resume.v1.py $nepochs $wdir/checkpoints/141861760.gadi-pbs/lucie_126.pt

