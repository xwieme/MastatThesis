#!/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -l gpus=1
#PBS -l walltime=02:00:00
#PBS -l mem=16gb

cd $VSC_SCRATCH/MastatThesis/code/scripts/aqsoldb_b
# Make python module XAIChem globaly available
export PYTHONPATH=$VSC_SCRATCH/MastatThesis/code/XAIChem:$PYTHONPATH
# Login to wandb to keep track of training progress
source ../../env.sh

# Train the GNN model in an apptainer container
apptainer exec --nv $VSC_SCRATCH/xai_chemistry_lab.sif \
    /opt/conda/envs/lab/bin/python \
    ../train_model.py \
    "../../../data/aqsoldb_B" \
    "aqsoldb_B" \
    $PBS_ARRAYID 