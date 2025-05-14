using Pkg
Pkg.activate(".")

using Distributed

@everywhere using moment_kinetics
using moment_kinetics.parameter_scans: get_scan_inputs

"""
    run_parameter_scan(args...)

Run a parameter scan, getting the inputs for each run from
[`moment_kinetics.parameter_scans.get_scan_inputs`](@ref).

If MPI is not used (i.e. each run should be run in serial), then `@distributed`
parallelism can be used to launch several runs at the same time. To do this, start julia
using the `-p` option to set the number of distributed processes, for example
```shell
\$ julia --project -p 8 run_parameter_scan.jl examples/something/scan_foobar.toml
```

When MPI is used, do not pass the `-p` flag to julia. Each run will run in parallel using
MPI, and the different runs in the scan will be started one after the other.
"""
function run_parameter_scan(args...)
    scan_inputs = get_scan_inputs(args...)

    @sync @distributed for s ∈ scan_inputs
        println("running ", s["output"]["run_name"])
        run_moment_kinetics(s)
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_parameter_scan()
end
