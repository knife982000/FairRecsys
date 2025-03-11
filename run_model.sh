#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH --mem=192g
#SBATCH --cpus-per-task=15

singularity exec --nv recbole.sif bash trainer.sh "$1" "$2"