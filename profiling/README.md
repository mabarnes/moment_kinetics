This subdirectory contains some scripts for profiling `moment_kinetics`.

Memory profiling
----------------
Julia has a `--track-allocation=all` option that saves information on the memory
allocation by each line of code.

`memory_profile.jl` is a script that launches `run_moment_kinetics()` to be profiled. It
uses a short initial run to get compilation out of the way, after which the allocation
counters are reset (so that we are not counting allocations for compilation or for the
initial short run). Finally `run_moment_kinetics()` is called for the actual profiling.

For convenience, the bash script `run_memory_profile.sh` is provided which runs
`memory_profile.jl` as a script while passing the necessary command-line flags to julia.

An input file must be passed as the first command line argument to
`run_memory_profile.sh` or `memory_profile.jl`. Various `*.toml` input files are
included for different types of run.

After the profiling finishes, `.mem` files containing the results are saved alongside
the `.jl` source files. `collect_memory_stats.jl` uses `Coverage.jl` to collect results,
sorted by amount of allocation. The script then prints info and the source lines around
the allocation for the largest allocations (by default 5, but can be changed by passing
a number as the first command line argument - 0 or negative means print all
allocations). Also saves all the results to a file `memory_profile.txt`.
`collect_memory_stats.jl` is run with default arguments by `run_memory_profile.sh` after
`memory_profile.jl` finishes.

`collect_memory_stats.jl` reads all the `.mem` files present, so it is probably most
useful to delete the existing `.mem` files before running another profile.
`cleanup_mem_files.sh` will take care of this, by reading the list of `.mem` files that
were found by `collect_memory_stats.jl` (which are saved to `mem_files_list.txt`) and
deleting all of them.

An example workflow might look like this:
```
# run profiling - top 5 allocation sites are printed
./run_memory_profile.sh sound_wave_chebyshev.toml

# read the output file to see more results
less memory_profile.txt

# clean up .mem files
./cleanup_mem_files.sh

# run profiling for a different case
./run_memory_profile.sh sound_wave_finite_difference.toml

# clean up .mem files again
./cleanup_mem_files.sh
```

Sampling profiler
-----------------
Julia has a built in sampling profiler in the Profile package.

`sampling_profile.jl` is a script that profiles `run_moment_kinetics()` It uses a short
initial run to get compilation out of the way, after which the profile is reset and
`run_moment_kinetics()` is called using the `@profile` macro from `Profile`. The profile
is printed to stdout.

For convenience `run_sampling_profile.sh` calls `sampling_profile.jl` with the necessary
flags passed to `julia`. The first argument to the script gives the input file to use.
Various `*.toml` input files are included for different types of run.
