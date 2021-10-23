This `debug_test` directory contains scripts for running a set of short runs, intended
to be used with the `--debug` to check for bugs (e.g. race conditions). The output is
not checked - the intention is just to catch errors raised by the debugging checks.

The inputs only have 2 time-steps, and very few grid points, because the debug checks
are very slow. The actual output is not important, so it does not matter that the runs
are badly under-resolved.

It may be necessary to use the `--compiled-modules=no` flag to Julia for changes
to the `--debug` setting to be picked up correctly.

To run the debug tests, call (from the top-level `moment_kinetics` directory)
something like
```
julia --project --check-bounds=yes --compiled-modules=no debug_test/runtests.jl --debug 99
```
