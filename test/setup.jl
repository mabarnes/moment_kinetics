"""
Common setup and utility functions for tests

Included in test files as `include("setup.jl")`
"""


# Commonly needed packages
##########################
using Test: @testset, @test
using moment_kinetics

module MKTestUtilities

export use_verbose, @long, quietoutput, global_rank, load_test_output, maxabs_norm,
       @testset_skip

using moment_kinetics.communication: global_rank
using moment_kinetics.command_line_options: get_options
using moment_kinetics.load_data: open_netcdf_file
using moment_kinetics.load_data: load_coordinate_data, load_fields_data,
                                 load_moments_data, load_pdf_data

const use_verbose = get_options()["verbose"]


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

"""
    load_test_output(input::Dict, to_load::Tuple{Symbol})

Load the output of a test that was run with settings in `input`. `to_load` specifies
which variables to load - it can include potential `:phi`, moments `:moments`,
distribution function `:f`. Coordinate data is always loaded.

Returns a `Dict` whose keys are `String` containing all loaded data.
"""
function load_test_output(input::Dict, to_load::Tuple{Vararg{Symbol}})
    output = Dict{String, Any}()

    path = joinpath(realpath(input["base_directory"]), input["run_name"], input["run_name"])

    # open the netcdf file and give it the handle 'fid'
    fid = open_netcdf_file(path)

    # load space-time coordinate data
    output["vpa"], output["vperp"], output["z"], output["r"], output["ntime"],
    output["time"] = load_coordinate_data(fid)

    if :phi ∈ to_load
        # Load EM fields
        output["phi"] = load_fields_data(fid)
    end

    if :moments ∈ to_load
        # Load moments
        output["density"], output["parallel_flow"], output["parallel_pressure"],
        output["parallel_heat_flux"], output["thermal_speed"], output["n_species"],
        output["evolve_ppar"] = load_moments_data(fid)
    end

    if :f ∈ to_load
        # load full (vpa,vperp,z,r,species,t) particle distribution function (pdf) data
        output["f"] = load_pdf_data(fid)
    end

    close(fid)

    return output
end

end

using .MKTestUtilities
