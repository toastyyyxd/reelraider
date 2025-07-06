#!/bin/bash
set -e

DATASET_DIR="$(dirname "$0")/raw"
BASE_URL="https://datasets.imdbws.com"

FILES=(
    "name.basics.tsv.gz"
    "title.ratings.tsv.gz"
)

mkdir -p "$DATASET_DIR"

for file in "${FILES[@]}"; do
    curl -L -o "$DATASET_DIR/$file" "$BASE_URL/$file"
done