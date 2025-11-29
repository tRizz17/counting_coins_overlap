#!/bin/bash
module load gcc/7.5
/usr/local/apps/cuda/11.2/bin/nvcc countcoins-overlap.cu -o countcoins -lm -Xcompiler -fopenmp
for b in 1 2 3 4 60 117 1024 1036
do
        ./countcoins ./assets/$b"heads.pgm"
done