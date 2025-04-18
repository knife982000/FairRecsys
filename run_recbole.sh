#!/bin/bash

GPUS=1
OVERSAMPLE="0.0"
UNDERSAMPLE="0.0"

while getopts ":d:m:eg:n:o:u:h" opt; do
  case $opt in
    d) DATASET="$OPTARG" ;;
    m) MODEL="$OPTARG" ;;
    e) EVAL="-e" ;;
    g) GPUS="$OPTARG" ;;
    n) NODE="$OPTARG" ;;
    o) OVERSAMPLE="$OPTARG" ;;
    u) UNDERSAMPLE="$OPTARG" ;;
    h)
      echo "Usage: $0 -d <DATASET> -m <MODEL> [-g <GPUS=4>] [-n <NODE>] [-o <OVERSAMPLE=0>] [-u <UNDERSAMPLE=0>] [-e]"
      exit 0
      ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

MEMORY="$((GPUS * 32))G"
JOB_NAME="Rec-${MODEL}-${DATASET}"
OUTPUT_FILE="${DATASET}-${MODEL}"
CPUS="$((GPUS * 10))"
NAME="${MODEL}"
if (( $(echo "$OVERSAMPLE != 0.0" | bc) )); then
  NAME="${NAME}_O_${OVERSAMPLE}"
  OUTPUT_FILE="${OUTPUT_FILE}_O_${OVERSAMPLE}"
fi
if (( $(echo "$UNDERSAMPLE != 0.0" | bc) )); then
  NAME="${NAME}_U_${UNDERSAMPLE}"
  OUTPUT_FILE="${OUTPUT_FILE}_U_${UNDERSAMPLE}"
fi
if [ -n "$EVAL" ]; then
  JOB_NAME="${JOB_NAME}_Eval"
  OUTPUT_FILE="${OUTPUT_FILE}_Eval"
fi
OUTPUT_FILE="${OUTPUT_FILE}.log"

echo "Submitting job with the following parameters:"
echo "  Dataset:        $DATASET"
echo "  Model:          $MODEL"
echo "  GPUs:           $GPUS"
echo "  Memory:         $MEMORY"
echo "  Job Name:       $JOB_NAME"
echo "  Output File:    $OUTPUT_FILE"
echo "  CPUs:           $CPUS"
[ -n "$NODE" ] && echo "  Node:           $NODE"
(( $(echo "$OVERSAMPLE != 0.0" | bc) ))  && echo "  Oversample:     $OVERSAMPLE"
(( $(echo "$UNDERSAMPLE != 0.0" | bc) )) && echo "  Undersample:    $UNDERSAMPLE"
[ -n "$EVAL" ] && echo "  Evaluation:     Enabled"

sbatch_command="sbatch --job-name=\"$JOB_NAME\" --output=\"$OUTPUT_FILE\" --gres=gpu:$GPUS --mem=$MEMORY --cpus-per-task=$CPUS"

[ -n "$NODE" ] && sbatch_command+=" --nodelist=\"$NODE\""

sbatch_command="$sbatch_command --wrap=\"singularity exec --nv recbole.sif bash run_model.sh -d $DATASET -m $MODEL -s $NAME -o $OVERSAMPLE -u $UNDERSAMPLE $EVAL\""
eval "$sbatch_command"
