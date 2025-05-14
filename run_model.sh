#!/bin/bash

echo "Getting the latest version of RecBole from repository..."
if [ -d "./P6" ]; then
  rm -rdf ./P6
fi
git clone https://github.com/AAU-Dat6-2025/P6.git
pip install -e ./P6/RecBole --verbose

while getopts ":c:d:m:s:o:uerzqh" opt; do
  case $opt in
    c) CONFIG_FILE="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    m) MODEL="$OPTARG" ;;
    s) NAME="$OPTARG" ;;
    o) OVERSAMPLE="$OPTARG" ;;
    u) UNDERSAMPLE="$OPTARG" ;;
    e) EVAL="True" ;;
    r) RERANK="True" ;;
    z) ZIPF="True" ;;
    q) QUEUE="True" ;;
    h)
      echo "Usage: $0 -d <DATASET> -m <MODEL> -s <NAME> [-o <OVERSAMPLE=0>] [-u <UNDERSAMPLE=0>] [-e (evaluate)] [-r (mmr reranking)] [-q (using queue)] [-c <CONFIG_FILE>] [-h]"
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

    echo "Processing line: $line"

    # Read parameters from the line in the file
    IFS=' ' read -r MODEL DATASET NAME OVERSAMPLE UNDERSAMPLE EVAL RERANK ZIPF<<< "$line"

    echo "Running Python script with the following parameters:"
    echo "  Dataset:        $DATASET"
    echo "  Model:          $MODEL"
    echo "  Name:           $NAME"
    echo "  Oversample:     $OVERSAMPLE"
    echo "  Undersample:    $UNDERSAMPLE"
    [ "$EVAL" = "True" ] && echo "  Evaluation:     Enabled"
    [ "$RERANK" = "True" ] && echo "  MMR-Reranking:  Enabled"
    [ "$ZIPF" = "True" ] && echo "  ZIPS Penalty:   Enabled"
    # Construct the python command
    python_command="python3 main.py -d $DATASET -m $MODEL -s $NAME"
    python_command+=" -o $OVERSAMPLE"
    python_command+=" -u $UNDERSAMPLE"
    [ "$EVAL" = "True" ] && python_command+=" -e"
    [ "$RERANK" = "True" ] && python_command+=" -mmr"
    [ "$ZIPF" = "True" ] && python_command+=" -z"
    eval "$python_command"

    # Remove the executed line from the file
    sed -i "/^$line$/d" "$CONFIG_FILE"
    [ "$QUEUE" = "True" ] && kill -s SIGTERM $$
  done < "$CONFIG_FILE"
else
  if [ -z "$DATASET" ] || [ -z "$MODEL" ]; then
    echo "Error: Dataset, Model, and Name are required parameters."
    exit 1
  fi
  echo "Running Python script with the following parameters:"
  echo "  Dataset:        $DATASET"
  echo "  Model:          $MODEL"
  [ -n "$NAME" ] && echo "  Name:           $NAME"
  [ -n "$OVERSAMPLE" ]  && echo "  Oversample:     $OVERSAMPLE"
  [ -n "$UNDERSAMPLE" ] && echo "  Undersample:    $UNDERSAMPLE"
  [ -n "$EVAL" ] && echo "  Evaluation:     Enabled"
  [ -n "$RERANK" ] && echo "  MMR rerank:     Enabled"
  [ -n "$ZIPF" ] && echo "  ZIPS Penalty:   Enabled"

  python_command="python3 main.py -d $DATASET -m $MODEL -s $NAME"
  [ -n "$OVERSAMPLE" ]  && python_command+=" -o $OVERSAMPLE"
  [ -n "$UNDERSAMPLE" ] && python_command+=" -u $UNDERSAMPLE"
  [ -n "$EVAL" ] && python_command+=" -e"
  [ -n "$RERANK" ] && python_command+=" -mmr"
  [ -n "$ZIPF" ] && python_command+=" -z"
  eval "$python_command"
fi
