# Debugging


## Shared-memory parallelism

For information on race conditions and debugging, see [Shared memory debugging](@ref).


## Identifying location of regressions (debug output)

It can be useful to save output from intermediate points within each timestep.
This can be done by setting `debug_io=true` in the `[timestepping]` section of
the input, which creates a file `debug.dfns.h5` containing output from after
each call (roughly each kinetic equation term or moment equation) in
[`moment_kinetics.time_advance.euler_time_advance!`](@ref) and a few other
useful places (where `write_debug_IO()` is called, which is a local wrapper for
[`moment_kinetics.file_io.write_debug_data_to_binary`](@ref)).

One example where this is useful is when trying to identify a 'regression' -
where the output of some run was expected to be the same before and after a
change to the code, but is not. Comparing the 'debug output' from the
before/after runs can narrow down a lot the (first) place where a difference
occured (using `debug_io=true` should be easier than putting debug prints in
the two versions of the code before/after the change being debugged). To
compare the outputs and identify the earliest difference, use the script
`util/regression_test_debug_comparison.jl`. This can be imported in the command
line to provide a function `regression_test_debug_comparison()` that can be
used to compare debug output from two runs (see its docstring for more
details), or run as a command line script (if the default settings are OK) with
paths to the two debug files as the command line arguments.
