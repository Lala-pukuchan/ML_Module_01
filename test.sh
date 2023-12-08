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

python3 ex00/gradient.py
python3 ex01/vec_gradient.py
python3 ex02/fit.py
python3 ex03/my_linear_regression.py
python3 ex04/linear_model.py
