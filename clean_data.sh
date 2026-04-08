#!/bin/bash

echo "Deleting old data files..."

rm -f data/*.csv data/*.pkl data/*.npy
rm -f models/*.keras
rm -f results/*.csv
rm -f logs/*.log

echo "Done. Cleaned data/, models/, results/, logs/ directories"