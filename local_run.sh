#!/bin/bash

for run in {1..2}
do
    mpiexec -n $1 ./conv $2 $3 $4
done
