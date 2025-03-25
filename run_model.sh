#!/bin/bash
DATASET="$1"
MODEL="$2"
GPUS="${3:-4}"
MEMORY="$((GPUS * 32))G"
JOB_NAME="Rec-${MODEL}-${DATASET}-Training"
OUTPUT_FILE="${DATASET}-${MODEL}-Training.log"
CPUS="$((GPUS * 10))"
NODE="${4:-}"

echo "Submitting job with the following parameters:"
echo "  Dataset:        $DATASET"
echo "  Model:          $MODEL"
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

sbatch_command="$sbatch_command --wrap=\"singularity exec --nv recbole.sif python3 main.py -d \"$DATASET\" -m \"$MODEL\"\""
eval $sbatch_command