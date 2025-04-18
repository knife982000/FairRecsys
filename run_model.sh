#!/bin/bash

while getopts ":c:d:m:s:o:ueh" opt; do
  case $opt in
    c) CONFIG_FILE="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    m) MODEL="$OPTARG" ;;
    s) NAME="$OPTARG" ;;
    o) OVERSAMPLE="$OPTARG" ;;
    u) UNDERSAMPLE="$OPTARG" ;;
    e) EVAL="-e" ;;
    h)
      echo "Usage: $0 -d <DATASET> -m <MODEL> -s <NAME> [-o <OVERSAMPLE=0>] [-u <UNDERSAMPLE=0>] [-e] [-c <CONFIG_FILE>] [-h]"
      exit 0
      ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

echo "${CONFIG_FILE}"

if [ -n "$CONFIG_FILE" ]; then
  echo "Using configuration file: $CONFIG_FILE"
  while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" =~ ^# ]] && continue

    # Read parameters from the line in the file
    IFS=' ' read -r MODEL DATASET NAME OVERSAMPLE UNDERSAMPLE EVAL <<< "$line"

    echo "Running Python script with the following parameters:"
    echo "  Dataset:        $DATASET"
    echo "  Model:          $MODEL"
    echo "  Name:           $NAME"
    (( $(echo "$OVERSAMPLE != 0.0" | bc) ))   && echo "  Oversample:     $OVERSAMPLE"
    (( $(echo "$UNDERSAMPLE != 0.0" | bc) ))  && echo "  Undersample:    $UNDERSAMPLE"
    [ "$EVAL" = "True" ] && echo "  Evaluation:     Enabled"

    # Construct the python command
    python_command="python3 main.py -d $DATASET -m $MODEL -s $NAME"
    (( $(echo "$OVERSAMPLE != 0.0" | bc) ))   && python_command+=" -o $OVERSAMPLE"
    (( $(echo "$UNDERSAMPLE != 0.0" | bc) ))  && python_command+=" -u $UNDERSAMPLE"
    [ "$EVAL" = "True" ] && python_command+=" -e"
    eval "$python_command"

    # Remove the executed line from the file
    sed -i "/^$line$/d" "$CONFIG_FILE"
  done < "$CONFIG_FILE"
else
  if [ -z "$DATASET" ] || [ -z "$MODEL" ] || [ -z "$NAME" ]; then
    echo "Error: Dataset, Model, and Name are required parameters."
    exit 1
  fi
  echo "Running Python script with the following parameters:"
  echo "  Dataset:        $DATASET"
  echo "  Model:          $MODEL"
  echo "  Name:           $NAME"
  [ -n "$OVERSAMPLE" ]  && (( $(echo "$OVERSAMPLE != 0.0" | bc) ))  && echo "  Oversample:     $OVERSAMPLE"
  [ -n "$UNDERSAMPLE" ] && (( $(echo "$UNDERSAMPLE != 0.0" | bc) )) && echo "  Undersample:    $UNDERSAMPLE"
  [ -n "$EVAL" ] && echo "  Evaluation:     Enabled"

  python_command="python3 main.py -d $DATASET -m $MODEL -s $NAME"
  [ -n "$OVERSAMPLE" ]  && (( $(echo "$OVERSAMPLE != 0.0" | bc) ))  && python_command+=" --oversample $OVERSAMPLE"
  [ -n "$UNDERSAMPLE" ] && (( $(echo "$UNDERSAMPLE != 0.0" | bc) )) && python_command+=" --undersample $UNDERSAMPLE"
  [ -n "$EVAL" ] && python_command+=" $EVAL"
  eval "$python_command"
fi
