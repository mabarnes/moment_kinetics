# Getting started

The full documentation is online at [https://mabarnes.github.io/moment_kinetics](https://mabarnes.github.io/moment_kinetics).

## Setup

If you are working on a supported machine, use the `machines/machine_setup.sh`
script, see [Setup for `moment_kinetics` on known clusters](@ref). Otherwise:

1) Ensure that the Julia version is >= 1.7.0 by doing
    ```
    $ julia --version
    ```
    at command line.

2) Dependencies need to be installed to the project
    environment. Start Julia with
    ```
    $ julia --project
    ```
    (which activates the 'project' in the current directory, or after starting with `julia`, in the REPL type `]` to enter `pkg>` mode, enter `activate .` and then backspace to leave `pkg>` mode). Once in the `moment_kinetics` project, enter `pkg>` mode by typing `]` and then run the command
    ```
    (moment_kinetics) pkg> instantiate
    ```
    this should download and install all the dependencies.

3) For julia>=1.6, pre-compiling dependencies manually is not necessary any more due to improvements to the native pre-compilation, so this step can be skipped (although precompiling the whole `moment_kinetics` code may still be useful sometimes). To pre-compile a static image (`dependencies.so`) that includes most of the external packages required for running and post-processing, run
    ```
    $ julia -O3 precompile_dependencies.jl
    ```
    To use the precompiled code, add an option `-Jdependencies.so` when starting julia.
    It is also possible to precompile the whole package into a static image (`moment_kinetics.so`) using
    ```
    $ julia -O3 precompile.jl
    ```
   this significantly decreases the load time but prevents code changes from taking effect when `moment_kinetics.so` is used without repeating the precompilation (to use this option, add an option `-Jmoment_kinetics.so` when starting julia).

4) In the course of development, it is sometimes helpful to upgrade the Julia version. Upgrading the version of Julia or upgrading packages may require a fresh installation of `moment_kinetics`. To make a fresh install with the latest package versions it is necessary to remove (or rename) the `Manifest.jl` file in the main directory, and generate a new `Manifest.jl` with step 1) above. It can sometimes be necessary to remove or rename the `.julia/` folder in your root directory for this step to be successful.

5) One may have to set an environment variable to avoid error messages from the Qt library. If you execute the command
    ```
    $ julia --project run_post_processing.jl runs/your_run_dir/
    ```
    and see the error message
    ```
    qt.qpa.xcb: could not connect to display
    qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
    This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
    ```
    this can be suppressed by setting
    ```
    export QT_QPA_PLATFORM=offscreen
    ```
    in your `.bashrc` or `.bash_profile` files.

## Run a simulation

To run julia with optimization, type
```
$ julia -O3 --project run_moment_kinetics.jl input.toml
```
Options are specified in a TOML file, e.g. `input.toml` here. The defaults are
specified in `moment_kinetics_input.jl`.
* To run in parallel, just put `mpirun -np <n>` in front of the call you would
  normally use, with `<n>` the number of processes to use.
* It may be more convenient when running `moment_kinetics` more than once to
  work from the Julia REPL, e.g.
    ```
    $ julia -O3 --project
    julia> using moment_kinetics
    julia> run_moment_kinetics(input)
    ```
    where `input` is a `Dict()` containing any non-default options desired.
    Input can also be loaded from a TOML file passing the filaname as a String
    to the second argument, e.g.
    ```
    julia> run_moment_kinetics("input.toml")
    ```
    Especially when developing the code, a lot of compilation time can be saved
    by using [Revise.jl](https://timholy.github.io/Revise.jl/stable/), and
    re-running a test case in the REPL (without restarting `julia`).

### Stopping a run

When running in the REPL (especially with MPI) interrupting a run using Ctrl-C
can mess things up, and require you to restart Julia. There is also a chance
that you might interrupt while writing the output files and corrupt them. To
avoid these problems, you can stop the run cleanly (including writing the
distribution functions at the last time point, so that it is possible to
restart the run from where you stopped it), by creating an empty file called
`stop` in the run directory. For example, if the name of your run is
'my\_example'
```shell
$ touch runs/my_example/stop
```
`moment_kinetics` checks for this file when it is going to write output, and if
it is present writes all output and then returns cleanly. The 'stop file' is
deleted when a run is (re-)started, if present, so you do not have to manually
delete it before (re-)starting the run again.

## Restarting

To restart a simulation using `input.toml` from the last time point in the
existing run directory,
```
$ julia -O3 --project run_moment_kinetics --restart input.toml
```
or to restart from a specific output file - either from the same run or (if the
settings are compatible, see below) a different one - here
`runs/example/example.dfns.h5`
```
$ julia -O3 --project run_moment_kinetics input.toml runs/example/example.dfns.h5
```
The output file must include distribution functions. When not using parallel
I/O there will be multiple output files from different MPI ranks - any one of
these can be passed.

To do the same from the Julia REPL
```
$ julia -O3 --project
julia> run_moment_kinetics("input.toml", restart=true)
```
or
```
julia> run_moment_kinetics("input.toml", restart="runs/example/example.dfns.h5")
```

When calling the `run_moment_kinetics()` function you can also choose a
particular time index to restart from, e.g.
```
julia> run_moment_kinetics("input.toml", restart="runs/example/example.dfns.h5", restart_time_index=42)
```

It is possible to restart a run from another output file with different
resolution settings or different moment-kinetic options. This is done by
interpolating variables from the old run onto the new grid.
* When interpolating in spatial dimensions it is not recommended to change the
  length of the domain.
* For velocity space dimensions, changing the size of the domain should be OK.
  Points outside the original domain will be filled with $\propto \exp(-v^2)$
  decreasing values.
* When changing from 1D (no $r$-dimension) to 2D (with $r$-dimension), the
  interpolated values will be constant in $r$.
* When changing from 1V to 2V or 3V, the interpolated values will be
  proportional to $\exp(-v_j^2)$ in the new dimension(s).

When running in parallel, both the old and the new grids must be compatible
with the distributed-MPI parallelisation. When not using [Parallel I/O](@ref),
the distributed-MPI domain decomposition must be identical in the old and new
runs (as each block only reads from a single file).

## Post processing quickstart

To make plots and calculate frequencies/growth rates, run
```
$ julia --project run_post_processing.jl runs/<directory to process>
```
passing the directory to process as a command line argument. Input options
for post-processing can be specified in `post_processing_input.jl`. Note that
even when running interactively, it is necessary to restart Julia after
modifying `post_processing_input.jl`.

Post processing can be done for several directories at once using
```
$ julia --project post_processing_driver.jl runs/<directory1> runs/<directory2> ...
```
passing the directories to process as command line arguments. Optionally pass a
number as the first argument to parallelise post processing of different
directories.

### Alternative post-processing

An alternative post-processing module, written to be a bit more generic and
flexible, and able to be used interactively, is provided in
`makie_post_processing`, see [Post processing](@ref).

## Parallel I/O

Note that to enable parallel I/O, you need to get HDF5.jl to use the system
HDF5 library (which must be MPI-enabled and compiled using the same MPI as you
run Julia with). To do this (see [the HDF5.jl
docs](https://juliaio.github.io/HDF5.jl/stable/#Using-custom-or-system-provided-HDF5-binaries))
run (with the `moment_kinetics` project activated in Julia)
```
using HDF5

HDF5.API.set_libraries!("/path/to/your/hdf5/directory/libhdf5.so",
                        "/path/to/your/hdf5/directory/libhdf5_hl.so")
```
JTO also found that (on a Linux laptop) it was necessary to compile HDF5 from
source. The system-provided, MPI-linked libhdf5 depended on libcurl, and Julia
links to an incompatible libcurl, causing an error. When compiled from source
(enabling MPI!), HDF5 does not require libcurl (guess it is an optional
dependency), avoiding the problem.

## Running parameter scans
Parameter scans (see [Parameter scans](@ref)) can be performed by running
```
$ julia -O3 --project run_parameter_scan.jl path/to/scan/input.toml
```
If running a scan, it can be parallelised by passing the `-p` argument to julia, e.g. to run on 8 processes
```
$ julia -p 8 -O3 --project run_parameter_scan.jl path/to/scan/input.toml
```

## Tests
There is a test suite in the `test/` subdirectory. It can be run in a few ways:
* Run using `Pkg`. Either using `pkg>` mode
    ```
    $ julia -O3 --project
    julia> <press ']' to enter pkg mode>
    (moment_kinetics) pkg> test
    ```
    using `Pkg` in the REPL
    ```
    $ julia -O3 --project
    julia> import Pkg
    julia> Pkg.test()
    ```
    or run on the command line
    ```
    julia -O3 --project -e "import Pkg; Pkg.test()`
    ```
* Execute some or all of the tests as a script. For example in the terminal run
    ```
    $ julia -O3 --project test/runtests.jl
    ```
    or in the REPL run
    ```
    julia> include("test/runtests.jl")
    ```
    Individual test files can also be used instead of `runtests.jl`, which runs all the tests.

By default the test suite should run fairly quickly (in a few minutes). To do
so, it skips many cases. To run more comprehensive tests, you can activate the
`--long` option:
* Using `test_args` argument
    ```
    julia> Pkg.test(; test_args=["--long"])
    ```
    Note the semicolon is necessary.
* In the REPL, run
    ```
    julia> push!(ARGS, "--long")
    ```
    before running the tests.
* Running from the terminal, pass as a command line argument, e.g.
    ```
    $ julia -O3 --project --long test/runtests.jl
    ```

To get more output on what tests were successful, an option `--verbose` (or
`-v`) can be passed in a similar way to `--long` (if any tests fail, the output
is printed by default).
