#!/bin/bash

# Run the memory_profile.jl script with memory profiling activated
# First argument to the script gives the input file to use
julia -O3 --check-bounds=no --project sampling_profile.jl $1
