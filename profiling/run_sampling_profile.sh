#!/bin/bash

# Run the sampling_profile.jl script
# First argument to the script gives the input file to use
../bin/julia -O3 --check-bounds=no --project sampling_profile.jl $1 | tee profile.txt
