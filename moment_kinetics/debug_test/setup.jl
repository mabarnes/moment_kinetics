"""
Common setup and utility functions for debug test runs

Included in debug test files as `include("setup.jl")`
"""


# Commonly needed packages
##########################
using Test: @testset, @test
using moment_kinetics
using moment_kinetics.communication
using Base.Filesystem: tempname
using MPI

# Test utility functions
########################

"""
Get a single temporary directory that is the same on all MPI ranks
"""
function get_MPI_tempdir()
    if global_rank[] == 0
        test_output_directory = tempname()
        mkpath(test_output_directory)
    else
        test_output_directory = ""
    end
    # Convert test_output_directory to a Vector{Char} so we can Bcast it
    v = fill(Char(0), 1024)
    for (i,c) ∈ enumerate(test_output_directory)
        v[i] = c
    end
    MPI.Bcast!(v, 0, comm_world)
    # Remove null characters wheen converting back to string
    return replace(String(v), "\0"=>"")
end
