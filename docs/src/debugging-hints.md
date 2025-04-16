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


## Dumping state of physics variables

There is a function [`moment_kinetics.file_io.debug_dump`](@ref) provided in the
[`moment_kinetics.file_io`](@ref) module that can be inserted to save the distribution
function, moments, etc. These can include a label so that if there are several 'dumps'
within a timestep, they can be easily identified.

To use, first import the function
```
using ..file_io: debug_dump
```
then insert calls at the point where you want to save the variables, e.g.
```
debug_dump(f, density, upar, ppar, phi, t, istage=istage, label="foo")
```
where `f`, `density`, `upar`, `ppar`, and `phi` are arrays and `t` is an `mk_float`.
`istage` is an optional `mk_int`, and can be used to identify the stage in the
`ssp_rk!()` function. `label` is optional and can be any string, intended to distinguish
different calls to `debug_dump()`.

There is an alternative method (implementation) of the function that takes
[`moment_kinetics.moment_kinetics_structs.scratch_pdf`](@ref) and
[`moment_kinetics.moment_kinetics_structs.em_fields_struct`](@ref) arguments. This can be
convenient within the functions in [`moment_kinetics.time_advance`](@ref), e.g.
```
debug_dump(fvec_out, fields, t, istage=istage, label="bar")
```

Any of the positional arguments can be replaced by `nothing` if they are not available
in a certain place, or just not needed. If `nothing` is passed, then arrays filled with
`0.0` are written into the output.

The output is written into a NetCDF file `debug_output.cdf` in the current directory
(the filename is currently hard-coded because that was simpler than adding more command
line arguments, etc.).

For debugging, a script `util/compare_debug_files.jl` is provided to compare two output
files, assumed to have similar structure (i.e. the same set of `debug_dump()` calls). An
example workflow could be:
1. Checkout `master` branch.
2. Create new branch, `git checkout -b newfeature`.
3. Work on `newfeature`...
4. At some point, a bug is introduced which breaks some test or example - i.e. changes
   its output. Commit the current state of the code.
5. Add `debug_dump()` calls before and after locations where the bug is likely to be.
6. Run the broken case.
7. `mv debug_output.cdf debug_after.cdf`, so the file does not get overwritten.
8. `git stash` to 'save' the `debug_dump()` statements.
9. `git checkout` a commit where the test/example was working.
10. `git stash pop` to add the `debug_dump()` statements on top of the working commit
    (fingers crossed there are no merge conflicts).
11. Run the test example again.
12. `mv debug_output.cdf debug_before.cdf`
13. Run the comparison script
    ```
    julia util/compare_debug_files.jl debug_before.cdf debug_after.cdf
    ```
14. The script identifies the first point (by `t`, `istage` and `label`) where any
    variable in the two output files has a maximum difference that is larger than some
    very tight (`1.e-14`) absolute and relative tolerances, and reports which
    variables were different and their maximum difference.
