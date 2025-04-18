#!/bin/bash

#SBATCH --job-name=recbole_runner
#SBATCH --signal=B:SIGTERM@30
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=80
#SBATCH --mem=192G

# Maximum number of restarts allowed
max_restarts=4

# Fetch the current restart count from SLURM job context
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*' | cut -d= -f2)

# If no restarts found, set to 0 (first run)
iteration=${restarts:-0}

# Dynamically set output and error filenames
outfile="${SLURM_JOB_ID}_${iteration}.out"
errfile="${SLURM_JOB_ID}_${iteration}.err"

# Print filenames for debugging
echo "Output file: ${outfile}"
echo "Error file: ${errfile}"

# Define a SIGTERM handler for requeueing
term_handler() {
    echo "SIGTERM received at $(date)"
    if [[ $iteration -lt $max_restarts ]]; then
        echo "Requeuing job (iteration: $iteration)"
        scontrol requeue ${SLURM_JOB_ID}
        exit 0
    else
        echo "Maximum restarts reached, exiting."
        exit 1
    fi
}

# Trap SIGTERM to execute the handler
trap 'term_handler' SIGTERM

# Check if a configuration file is provided
if [ -z "$1" ]; then
    echo "Error: Configuration file is required."
    exit 1
fi

CONFIG_FILE="$1"

# Ensure the configuration file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found."
    exit 1
fi

# Use srun to dynamically specify the output and error files
srun --output="${outfile}" --error="${errfile}" bash run_model.sh -c "$CONFIG_FILE"
