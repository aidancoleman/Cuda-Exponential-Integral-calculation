#!/bin/bash

make
SIZES=(5000 8192 16384 20000)
FILE="output.csv"
echo "n,m,cpu_time,gpu_time,float_error,double_error,speedup" > $FILE

for sz in "${SIZES[@]}"; do
    for B in 8 16 32 64; do
        ./expIntegral.out -n $sz -m $sz -B $B -t -r -e >> $FILE
    done
done
