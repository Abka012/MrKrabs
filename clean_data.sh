#!/bin/bash

echo "Deleting old data files..."

rm -f data/*.csv data/*.pkl data/*.npy
rm -f models/*.keras models/*.pkl
rm -f results/*.csv
rm -f logs/*.log
rm -f __pycache__/*.pyc

echo "Done. Cleaned data/, models/, results/, logs/, __pycache__ directories"
