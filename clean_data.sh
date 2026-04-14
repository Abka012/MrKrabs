#!/bin/bash

set -e

echo "Deleting old data files..."

find data -type f \( -name "*.csv" -o -name "*.pkl" -o -name "*.npy" \) -delete
find models -type f \( -name "*.keras" -o -name "*.pkl" -o -name "*.json" \) -delete
find results -type f \( -name "*.csv" -o -name "*.json" \) -delete
find logs -type f -name "*.log" -delete
find data -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
find models -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
find results -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
find logs -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
find __pycache__ -type f -name "*.pyc" -delete

echo "Done. Cleaned data/, models/, results/, logs/, __pycache__ directories"
