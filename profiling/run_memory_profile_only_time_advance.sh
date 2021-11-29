#!/bin/bash

# Run the memory_profile.jl script with memory profiling activated
# First argument to the script gives the input file to use
#julia -O3 --check-bounds=no --project --track-allocation=user memory_profile_only_time_advance.jl $1
julia -O3 --check-bounds=no --project --track-allocation=all memory_profile_only_time_advance.jl $1

echo "done profiling, collecting results"
echo ""

# Print top 5 allocation sites for convenience
julia --project collect_memory_stats.jl
