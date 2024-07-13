#!/bin/bash

# Define the path to the requirements file
requirements_file="requirements.txt"

# Check if the requirements file exists
if [ -f "$requirements_file" ]; then
    echo "$requirements_file exists."
else
    echo "$requirements_file does not exist."
    # Handle the absence of the file, e.g., exit the script with an error code
    exit 1
fi # Closes the if loop

while true; do
    echo "MinZheng 222983R Kedro Pipeline"
    echo "1) Run the whole pipeline (Data+ML)"
    echo "2) Run the data processing pipeline"
    echo "3) Run the ML pipeline"
    echo "4) Run the fine tuning pipeline"
    echo "5) Exit pipeline interface"
    echo "Please enter the number correlated with the process you want to run:"
    read user_input

    # Check the user input and act accordingly
    if [ "$user_input" == "1" ]; then
        kedro run
        echo "Kedro pipeline has been executed."

    elif [ "$user_input" == "2" ]; then
        kedro run --pipeline=dp
        echo "Data processing pipeline has been executed."

    elif [ "$user_input" == "3" ]; then
        kedro run --pipeline=ml
        echo "Machine Learning pipeline has been executed."

    elif [ "$user_input" == "5" ]; then

        echo "Kedro pipeline will now close."
        exit 0
    else
        echo "Invalid input. Please enter 'yes' or 'no'."
    fi # Closes the if loop
done
