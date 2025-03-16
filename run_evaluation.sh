#!/bin/bash

GPUS="${1:-4}"
MEMORY="$((GPUS * 32))G"
JOB_NAME="Rec-Evaluation"
OUTPUT_FILE="evaluation-out.log"
CPUS="$((GPUS * 10))"
NODE="${2:-}"

echo "Submitting job with the following parameters:"
echo "  GPUs:           $GPUS"
echo "  Memory:         $MEMORY"
echo "  Job Name:       $JOB_NAME"
echo "  Output File:    $OUTPUT_FILE"
echo "  CPUs:           $CPUS"
if [ -n "$NODE" ]; then
  echo "  Node:           $NODE"
fi

sbatch_command="sbatch --job-name=\"$JOB_NAME\" \
       --output=\"$OUTPUT_FILE\" \
       --gres=gpu:$GPUS \
       --mem=$MEMORY \
       --cpus-per-task=$CPUS"

if [ -n "$NODE" ]; then
  sbatch_command="$sbatch_command --nodelist=\"$NODE\""
fi

sbatch_command="$sbatch_command --wrap=\"singularity exec --nv recbole.sif bash evaluation.sh\""

eval $sbatch_command