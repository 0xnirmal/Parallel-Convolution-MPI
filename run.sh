#!/bin/bash

mpiexec -n $1 -mca btl ^openib ./conv $2 $3 $4
