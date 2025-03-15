#!/bin/bash

if [ ! -d "dataset/ml-1m" ]; then
  mkdir -p "dataset/ml-1m"
  wget -O "dataset/ml-1m/ml-1m.zip" "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/ml-1m.zip"
  unzip -o "dataset/ml-1m/ml-1m.zip" -d "dataset/ml-1m"
  rm "dataset/ml-1m/ml-1m.zip"
else
  echo "MovieLens 1M dataset already exists, skipping..."
fi

if [ ! -d "dataset/gowalla-merged" ]; then
  mkdir -p "dataset/gowalla-merged"
  wget -O "dataset/gowalla-merged/gowalla.zip" "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Gowalla/merged/gowalla.zip"
  unzip -o "dataset/gowalla-merged/gowalla.zip" -d "dataset/gowalla-merged"
  rm "dataset/gowalla-merged/gowalla.zip"
  mv "dataset/gowalla-merged/gowalla.inter" "dataset/gowalla-merged/gowalla-merged.inter"
else
  echo "Gowalla dataset already exists, skipping..."
fi

if [ ! -d "dataset/steam-merged/" ]; then
  mkdir -p "dataset/steam-merged/"
  wget -O "dataset/steam-merged/steam-merged.zip" "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Steam/merged/steam.zip"
  unzip -o "dataset/steam-merged/steam-merged.zip" -d "dataset/steam-merged"
  rm "dataset/steam-merged/steam-merged.zip"
  mv "dataset/steam-merged/steam/steam.item" "dataset/steam-merged/steam-merged.item"
  mv "dataset/steam-merged/steam/steam.inter" "dataset/steam-merged/steam-merged.inter"
  rm -r -f -d "dataset/steam-merged/__MACOSX"
  rm -r -f -d "dataset/steam-merged/steam"
else
  echo "Steam dataset already exists, skipping..."
fi