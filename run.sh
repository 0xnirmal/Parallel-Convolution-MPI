#!/bin/bash

for run in {1..2}
do
    mpiexec -n $1 -mca btl ^openib ./conv $2 $3 $4
done
