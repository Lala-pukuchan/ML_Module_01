#!/bin/bash

if [ ! -d "results" ]; then
    mkdir results
fi

for i in {0..6}
do
    if [ ! -d "results/ex0$i" ]; then
        mkdir "results/ex0$i"
    fi
done

# get current directory
currentDir=$(pwd)

# Append the subdirectory to the current directory
newPath="$currentDir/ex03"

# Export the new path to PYTHONPATH
export PYTHONPATH="$newPath:$PYTHONPATH"

# execute python script
python3 ex00/gradient.py
python3 ex01/vec_gradient.py
python3 ex02/fit.py
python3 ex03/test.py
python3 ex04/linear_model.py
python3 ex05/z_score.py
python3 ex06/minmax.py
