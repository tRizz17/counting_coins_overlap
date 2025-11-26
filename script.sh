#!/bin/bash
module load gcc/7.5
/usr/local/apps/cuda/11.2/bin/nvcc countcoins-overlap.cu -o countcoins -lm -Xcompiler -fopenmp
for b in 1 3 117 1024
do
        ./countcoins ./assets/$b"heads.pgm"
done