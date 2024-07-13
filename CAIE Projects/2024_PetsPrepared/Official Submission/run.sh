#!/bin/bash
# When error encountered, exit
set -e

cd "$PWD/src/dataprep/"

# Data Cleaning
echo "Cleaning Data"
python3 data_cleaning.py
echo "Cleaning Data Complete"

# EDA Data Engineering
echo "Creating EDA Data"
python3 eda_data_modification.py
echo "EDA Data Complete"

# Data Processing
echo "Processing Data"
python3 data_processing.py
echo "Data Processing Complete"

# ML
cd ../model
echo "Training Model"
python3 pipeline.py
echo "ML Completed"

cd ../../