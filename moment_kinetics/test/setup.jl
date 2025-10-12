"""
Common setup and utility functions for tests

Included in test files as `include("setup.jl")`
"""


# Commonly needed packages
##########################
using Test: @testset, @test
using moment_kinetics
using moment_kinetics.StableRNGs

module MKTestUtilities

export use_verbose, force_optional_dependencies, @long, quietoutput, get_MPI_tempdir,
       global_rank, global_size, maxabs_norm, elementwise_isapprox, @testset_skip,
       recursive_merge, OptionsDict

using moment_kinetics.communication: comm_world, global_rank, global_size
using moment_kinetics.command_line_options: get_options
using moment_kinetics.type_definitions: OptionsDict
using moment_kinetics.utils: recursive_merge

using moment_kinetics.BlockBandedMatrices
using MPI

const use_verbose = get_options()["verbose"]
const force_optional_dependencies = get_options()["force-optional-dependencies"]


# Convenience modifiers for test calls
######################################

"""
Use `@long` to mark tests that should normally be skipped to save time.

Tests marked with `@long` can be enabled by:
* if running tests as a script, by passing `--long` as a command line argument to
    `julia`.
* if running in the REPL, by modifying `ARGS`, e.g.
    ```
    push!(ARGS, "--long")
    ```
* by passing a `test_args` argument to `Pkg.test()`, e.g.
    ```
    Pkg.test(; test_args=["--long"])
    ```
    Note that the semicolon is necessary (not sure why).
"""
macro long(code)
    if get_options()["long"]
        :( $(esc(code)) )
    end
end


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

"""
Pass this function to the `norm` argument of `isapprox()` to test the maximum error
between two arrays.
"""
maxabs_norm(x) = maximum(abs.(x))

"""
    elementwise_isapprox(args...; kwargs...)

Calls `isapprox()` but forces the comparison to be done element-by-element, rather than
testing `norm(x-y)<max(rtol*max(norm(x),norm(y)),atol)`, which would effectively ignore
errors in small values. Takes the same args/kwargs as `isapprox()`, except for the `norm`
kwarg.
"""
@inline function elementwise_isapprox(args...; kwargs...)
    return isapprox(args...; norm=(x)->NaN, kwargs...)
end
@inline function elementwise_isapprox(args::BlockSkylineMatrix...; kwargs...)
    return isapprox((a.data for a ∈ args)...; norm=(x)->NaN, kwargs...)
end

"""
Get a single temporary directory that is the same on all MPI ranks
"""
function get_MPI_tempdir()
    if global_rank[] == 0
        if get_options()["ci"]
            runs_dir = abspath("runs/")
            mkpath(runs_dir)
            test_output_directory = tempname(runs_dir)
        else
            test_output_directory = tempname()
        end
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


# Custom macro to skip a testset
################################

import Test: Test, finish
using Test: DefaultTestSet, Broken
using Test: parse_testset_args

"""
Skip a testset

Use `@testset_skip` to replace `@testset` for some tests which should be skipped.

Usage
-----
Replace `@testset` with `@testset_skip "reason"` where `"reason"` is a string saying why
the test should be skipped (which should come before the description string, if that is
present).
"""
macro testset_skip(args...)
    isempty(args) && error("No arguments to @testset_skip")
    length(args) < 2 && error("First argument to @testset_skip giving reason for "
                              * "skipping is required")

    skip_reason = args[1]

    desc, _, _ = parse_testset_args(args[2:end-1])

    ex = quote
        # record the reason for the skip in the description, and mark the tests as
        # broken, but don't run tests
        local ts = DefaultTestSet(string($desc, " - ", $skip_reason))
        push!(ts.results, Broken(:skipped, "skipped tests"))
        local ret = finish(ts)
        ret
    end

    return ex
end

end

using .MKTestUtilities
