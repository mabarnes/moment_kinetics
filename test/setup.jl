"""
Common setup and utility functions for tests

Included in test files as `include("setup.jl")`
"""


# Commonly needed packages
##########################
using Test: @testset, @test
using moment_kinetics


# Test utility functions
########################

"""
Temporarily disable output to stdout.

Intended to be used as a context manager, so call like::

    quietoutput() do
        some code in here
        ...
    end
"""
function quietoutput(body)
    oldstd = stdout
    try
        redirect_stdout(open("/dev/null", "w"))

        # execute body code
        body()

    finally
        # Restore regular output
        redirect_stdout(oldstd)
    end
end
