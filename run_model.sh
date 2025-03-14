#!/bin/bash
DATASET="$1"
MODEL="$2"
JOB_NAME="Rec-${MODEL}-${DATASET}"
OUTPUT_FILE="${DATASET}-${MODEL}-out.log"
GPUS="4"
MEMORY="128g"
CPUS="15"

echo "Submitting job with the following parameters:"
echo "  Dataset:       $DATASET"
echo "  Model:         $MODEL"
echo "  Job Name:      $JOB_NAME"
echo "  Output File:   $OUTPUT_FILE"
echo "  GPUs:          $GPUS"
echo "  Memory:        $MEMORY"
echo "  CPUs:          $CPUS"

sbatch --job-name="$JOB_NAME" \
       --output="$OUTPUT_FILE" \
       --gres=gpu:$GPUS \
       --mem=$MEMORY \
       --cpus-per-task=$CPUS \
       --wrap="singularity exec --nv recbole.sif python3 main.py -d \"$DATASET\" -m \"$MODEL\""