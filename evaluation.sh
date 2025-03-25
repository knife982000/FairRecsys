#!/bin/bash

for MODEL in "BPR" "LightGCN" "NGCF" "MultiVAE" "Random"; do
  for DATASET in "ml-1m" "gowalla-merged" "steam-merged"; do
    if [ "$MODEL" == "MultiVAE" ] && [ "$DATASET" != "ml-1m" ]; then
      continue
    fi
    python3 main.py -d "$DATASET" -m "$MODEL" -e True
  done
done