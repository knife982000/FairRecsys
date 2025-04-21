#!/bin/bash

if [ ! -d "ml-1m" ]; then
  mkdir -p "ml-1m"
  wget -O "ml-1m/ml-1m.zip" "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/ml-1m.zip"
  unzip -o "ml-1m/ml-1m.zip" -d "ml-1m"
  rm "ml-1m/ml-1m.zip"
else
  echo "MovieLens 1M dataset already exists, skipping..."
fi

if [ ! -d "ml-20m" ]; then
  mkdir -p "ml-20m"
  wget -O "ml-20m/ml-20m.zip" "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/ml-20m.zip"
  unzip -o "ml-20m/ml-20m.zip" -d "ml-20m"
  rm "ml-20m/ml-20m.zip"
else
  echo "MovieLens 20M dataset already exists, skipping..."
fi

if [ ! -d "gowalla-merged" ]; then
  mkdir -p "gowalla-merged"
  wget -O "gowalla-merged/gowalla.zip" "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Gowalla/merged/gowalla.zip"
  unzip -o "gowalla-merged/gowalla.zip" -d "gowalla-merged"
  rm "gowalla-merged/gowalla.zip"
  mv "gowalla-merged/gowalla.inter" "gowalla-merged/gowalla-merged.inter"
else
  echo "Gowalla dataset already exists, skipping..."
fi

if [ ! -d "steam-merged/" ]; then
  mkdir -p "steam-merged/"
  wget -O "steam-merged/steam-merged.zip" "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Steam/merged/steam.zip"
  unzip -o "steam-merged/steam-merged.zip" -d "steam-merged"
  rm "steam-merged/steam-merged.zip"
  mv "steam-merged/steam/steam.item" "steam-merged/steam-merged.item"
  mv "steam-merged/steam/steam.inter" "steam-merged/steam-merged.inter"
  rm -r -f -d "steam-merged/__MACOSX"
  rm -r -f -d "steam-merged/steam"
else
  echo "Steam dataset already exists, skipping..."
fi
