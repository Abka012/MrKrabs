#!/bin/bash

set -e

echo "Deleting old data files..."

find data -type f \( -name "*.csv" -o -name "*.pkl" -o -name "*.npy" \) -delete
find models -type f \( -name "*.keras" -o -name "*.pkl" \) -delete
find results -type f \( -name "*.csv" -o -name "*.json" \) -delete
find logs -type f -name "*.log" -delete
find __pycache__ -type f -name "*.pyc" -delete

echo "Done. Cleaned data/, models/, results/, logs/, __pycache__ directories"
