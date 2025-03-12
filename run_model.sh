#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --mem=24g
#SBATCH --cpus-per-task=15
#SBATCH --job-name=Recbole
#SBATCH --output=log-%j.log

singularity exec --nv recbole.sif python3 main.py -d  "$1" -m  "$2" -p "$3"