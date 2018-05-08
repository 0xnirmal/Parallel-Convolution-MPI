#!/bin/bash

for run in {1..10}
do
    mpiexec -n $1 ./conv $2 $3
done
