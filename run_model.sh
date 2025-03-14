#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --mem=128g
#SBATCH --cpus-per-task=15
#SBATCH --job-name=Recbole-${2}-${1}
#SBATCH --output=%x-out.log

echo "Running job for dataset: $1, model: $2"
singularity exec --nv recbole.sif python3 main.py -d "$1" -m "$2"