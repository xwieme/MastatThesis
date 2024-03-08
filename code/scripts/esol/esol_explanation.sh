#!/bin/bash
#PBS -l nodes=1:ppn=8
#PBS -l gpus=1
#PBS -l walltime=01:00:00
#PBS -l mem=16gb

cd $VSC_SCRATCH/MastatThesis/code/scripts/esol
# Make python module XAIChem globaly available
export PYTHONPATH=$VSC_SCRATCH/MastatThesis/code/XAIChem:$PYTHONPATH

# Run script in apptainer container
apptainer exec --nv $VSC_SCRATCH/xai_chemistry_lab.sif \
    /opt/conda/envs/lab/bin/python \
    ../explanation.py \
    "../../../data/ESOL \
    $PBS_ARRAYID