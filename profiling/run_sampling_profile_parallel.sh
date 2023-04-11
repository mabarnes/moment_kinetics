#!/bin/bash

# Run the sampling_profile.jl script in parallel
# First argument to the script gives the number of processes, the second gives input file to use
mpirun -np $1 --output-filename profile julia -O3 --check-bounds=no --project sampling_profile.jl $2
