#!/bin/bash
DATASET="$1"
MODEL="$2"
GPUS="1"
MEMORY="32G"
JOB_NAME="Rec-${MODEL}-${DATASET}-Eval"
OUTPUT_FILE="${DATASET}-${MODEL}-Eval.log"
CPUS="10"
NODE="${3:-}"

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

sbatch_command_eval="sbatch --job-name=\"$JOB_NAME\" \
       --output=\"$OUTPUT_FILE\" \
       --gres=gpu:$GPUS \
       --mem=$MEMORY \
       --cpus-per-task=$CPUS"

if [ -n "$NODE" ]; then
  sbatch_command_eval="$sbatch_command_eval --nodelist=\"$NODE\""
fi

sbatch_command_eval="$sbatch_command_eval --wrap=\"singularity exec --nv recbole.sif python3 main.py -e True -d \"$DATASET\" -m \"$MODEL\"\""
eval $sbatch_command_eval