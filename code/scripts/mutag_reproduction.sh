#!/usr/bin/bash
#PBS -l nodes=1:ppn=2
#PBS -l gpus=1
#PBS -l walltime=02:00:00

cd $VSC_SCRATCH
# Make python module XAIChem globaly available
export PYTHONPATH=code/XAIChem:$PYTHONPATH

# Train the GNN model in an apptainer container
apptainer exec --nv /opt/conda/envs/lab/bin/python xai_chemistry_lab MastatThesis/code/scripts/mutage_reproduction.py --data_dir "MastatThesis/data"