"""
Common setup and utility functions for tests

Included in test files as `include("setup.jl")`
"""


# Commonly needed packages
##########################
using ArgParse
using Test: @testset, @test
using moment_kinetics


# Parse command line arguments to allow settings to be used for tests
#####################################################################
s = ArgParseSettings()
@add_arg_table! s begin
    "--long"
        help = "Include more tests, increasing test run time."
        action = :store_true
end
options = parse_args(s)


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
    if options["long"]
        :( $code )
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


# Provide custom macro to skip a testset
#################################################

# Wrap in a module so we don't need to import extra stuff into the global namespace
module SkipTestSets

export @testset_skip

import Test: Test, finish
using Test: DefaultTestSet, Broken
using Test: parse_testset_args

"""
Skip a testset

Use `@testset_skip` to replace `@testset` for some tests which should be skipped.

Usage
-----
Replace `@testset` with `@testset "reason"` where `"reason"` is a string saying why the
test should be skipped (which should come before the description string, if that is
present).
"""
macro testset_skip(args...)
    isempty(args) && error("No arguments to @testset_skip")
    length(args) < 2 && error("First argument to @testset_skip giving reason for "
                              * "skipping is required")

    skip_reason = args[1]

    desc, testsettype, options = parse_testset_args(args[2:end-1])

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

end # SkipTestSets

using .SkipTestSets
