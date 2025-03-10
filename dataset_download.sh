#!/bin/bash

urls=(
    "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/MovieLens/ml-1m.zip"
    "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Yahoo-Music/yahoo-music.zip"
)

mkdir -p dataset

for url in "${urls[@]}"; do
    filename=$(basename "$url")
    foldername=$(echo "$filename" | tr '[:upper:]' '[:lower:]' | sed 's/.zip//')
    mkdir -p "dataset/$foldername"
    wget -O "dataset/$foldername/$filename" "$url"
    unzip -o "dataset/$foldername/$filename" -d "dataset/$foldername"
    rm "dataset/$foldername/$filename"
done

mkdir -p "dataset/gowalla-merged"
wget -O "dataset/gowalla-merged/gowalla.zip" "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Gowalla/merged/gowalla.zip"
unzip -o "dataset/gowalla-merged/gowalla.zip" -d "dataset/gowalla-merged"
rm "dataset/gowalla-merged/gowalla.zip"
mv "dataset/gowalla-merged/gowalla.inter" "dataset/gowalla-merged/gowalla-merged.inter"


mkdir -p "dataset/amazon-books"
wget -O "dataset/amazon-books/Amazon_Books.zip" "https://recbole.s3-accelerate.amazonaws.com/ProcessedDatasets/Amazon_ratings/Amazon_Books.zip"
unzip -o "dataset/amazon-books/Amazon_Books.zip" -d "dataset/amazon-books"
rm "dataset/amazon-books/Amazon_Books.zip"
mv "dataset/amazon-books/Amazon_Books.inter" "dataset/amazon-books/amazon-books.inter"
mv "dataset/amazon-books/Amazon_Books.item" "dataset/amazon-books/amazon-books.item"
