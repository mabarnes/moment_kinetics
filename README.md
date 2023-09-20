# Getting started

The full documentation is online at [https://mabarnes.github.io/moment_kinetics](https://mabarnes.github.io/moment_kinetics).

## Basics
0) Ensure that the Julia version is >= 1.7.0 by doing
    ```
    $ julia --version
    ```
    at command line.
1) If you are working on a supported machine, use the
    `machines/machine_setup.sh` script, see [Setup for `moment_kinetics` on
    known clusters](@ref).
2) Dependencies need to be installed to the project
    environment. Start Julia with
    ```
    $ julia --project
    ```
    (which activates the 'project' in the current directory) (or after starting with `julia`, in the REPL type `]` to enter `pkg>` mode, enter `activate .` and then backspace to leave `pkg>` mode). Once in the `moment_kinetics` project, enter `pkg>` mode by typing `]` and then run the command
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
4) To run julia with optimization, type
    ```
    $ julia -O3 --project run_moment_kinetics.jl input.toml
    ```
    Options are specified in a TOML file, e.g. `input.toml` here. The defaults are specified in `moment_kinetics_input.jl`.
    * To run in parallel, just put `mpirun -np <n>` in front of the call you would normally use, with `<n>` the number of processes to use.
    * It may be more convenient when running `moment_kinetics` more than once to work from the Julia REPL, e.g.
        ```
        $ julia -O3 --project
        julia> using moment_kinetics
        julia> run_moment_kinetics(input)
        ```
        where `input` is a `Dict()` containing any non-default options desired. Input can also be loaded from a TOML file passing the filaname as a String to the second argument, e.g.
        ```
        julia> run_moment_kinetics("input.toml")
        ```
5) To restart a simulation using `input.toml` from the last time point in the existing run directory,
    ```
    $ julia -O3 --project run_moment_kinetics --restart input.toml
    ```
    or to restart from a specific output file - either from the same run or (if the settings are compatible) a different one - here `runs/example/example.dfns.h5`
    ```
    $ julia -O3 --project run_moment_kinetics input.toml runs/example/example.dfns.h5
    ```
    The output file must include distribution functions. When not using parallel I/O there will be multiple output files from different MPI ranks - any one of these can be passed.
    * To do the same from the Julia REPL
        ```
        $ julia -O3 --project
        julia> run_moment_kinetics("input.toml", restart=true)
        ```
        or
        ```
        julia> run_moment_kinetics("input.toml", restart="runs/example/example.dfns.h5")
        ```
    * When calling the `run_moment_kinetics()` function you can also choose a particular time index to restart from, e.g.
        ```
        julia> run_moment_kinetics("input.toml", restart="runs/example/example.dfns.h5", restart_time_index=42)
        ```
6) To make plots and calculate frequencies/growth rates, run
    ```
    $ julia --project run_post_processing.jl runs/<directory to process>
    ```
    passing the directory to process as a command line argument. Input options for post-processing can be specified in post_processing_input.jl.

7) Parameter scans (see [Running parameter scans](#running-parameter-scans)) or performance tests can be performed by running
    ```
    $ julia -O3 --project driver.jl
    ```
    If running a scan, it can be parallelised by passing the number of processors as an argument. Scan options are set in `scan_inputs.jl`.

8) Post processing can be done for several directories at once using
    ```
    $ julia --project post_processing_driver.jl runs/<directory1> runs/<directory2> ...
    ```
    passing the directories to process as command line arguments. Optionally pass a number as the first argument to parallelise post processing of different directories. Input options for post-processing can be specified in `post_processing_input.jl`.

9) In the course of development, it is sometimes helpful to upgrade the Julia veriosn. Upgrading the version of Julia or upgrading packages may require a fresh installation of `moment_kinetics`. To make a fresh install with the latest package versions it is necessary to remove (or rename) the `Manifest.jl` file in the main directory, and generate a new `Manifest.jl` with step 1) above. It can sometimes be necessary to remove or rename the `.julia/` folder in your root directory for this step to be successful.

10) One may have to set an environment variable to avoid error messages from the Qt library. If you execute the command

    ```
    $ julia --project run_post_processing.jl runs/your_run_dir/
    ```

and see the error message    
    
    
    qt.qpa.xcb: could not connect to display
    qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
    This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
    
    
this can be suppressed by setting 
```
export QT_QPA_PLATFORM=offscreen
```
in your `.bashrc` or `.bash_profile` files. 

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

## Parallel I/O

Note that to enable parallel I/O, you need to get HDF5.jl to use the system
HDF5 library (which must be MPI-enabled and compiled using the same MPI as you
run Julia with). To do this (see [the HDF5.jl
docs](https://juliaio.github.io/HDF5.jl/stable/#Using-custom-or-system-provided-HDF5-binaries))
run (with the `moment_kinetics` project activated in Julia)
```
julia> ENV["JULIA_HDF5_PATH"] = "/path/to/your/hdf5/directory"; using Pkg(); Pkg.build()
```
JTO also found that (on a Linux laptop) it was necessary to compile HDF5 from
source. The system-provided, MPI-linked libhdf5 depended on libcurl, and Julia
links to an incompatible libcurl, causing an error. When compiled from source
(enabling MPI!), HDF5 does not require libcurl (guess it is an optional
dependency), avoiding the problem.

## Running parameter scans
Parameter scans can be run, and can (optionally) use multiple processors. Short summary of implementation and usage:
1) `mk_input()` takes a Dict argument, which can modify values. So `mk_input()` sets the 'defaults' (for a scan), which are overridden by any key/value pairs in the Dict.
2) `mk_scan_inputs()` (in `scan_input.jl`) creates an Array of Dicts that can be passed to `mk_input()`. It first creates a Dict of parameters to scan over (keys are the names of the variable, values are an Array to scan over), then assembles an Array of Dicts (where each entry in the Array is a Dict with a single value for each variable being scanned). Most variables are combined as an 'inner product', e.g. `{:ni=>[0.5, 1.], :nn=>[0.5, 0.]}` gives `[{:ni=>0.5, :nn=>0.5}, {ni=>1., nn=>0.}]`. Any special variables specified in the `combine_outer` array are instead combined with the rest as an 'outer product', i.e. an entry is created for every value of those variables for each entry in the 'inner-producted' list. [This was just complicated enough to run the scans I've done so far without wasted simulations.]
3) The code in `driver.jl` picks between a single run (normal case), a performance_test, or creating a scan by calling `mk_scan_input()` and then looping over the returned array, calling `mk_input()` and running a simulation for each entry. This loop is parallelised (with the set of simulations dispatched over several processes - each simulation is still running serially). Running a scan (on 12 processes - actually 13 but the 'master' process doesn't run any of the loop bodies, so there are 12 'workers'):
    ```
    $ julia -O3 --project driver.jl 12
    ```
    (runs in serial if no argument is given)
4) The scan puts each run in a separate directory, named with a prefix specified by `base_name` in `scan_input.jl` and the rest the names and values of the scanned-over parameters (the names are created in `mk_scan_input()` too, and passed as the `:run_name` entry of the returned Dicts).
5) To run `post_processing.analyze_and_plot_data()` over a bunch of directories (again parallelized trivially, and the number of processes to use is an optional argument, serial if omitted):
    ```
    $ julia -O3 --project post_processing_driver.jl 12 runs/scan_name_*
    ```
6) Plotting the scan is not so general, `plot_comparison.jl` does it, but is only set up for the particular scans I ran - everything except the charge exchange frequencies is hard-coded in.
    ```
    $ julia -O3 --project plot_comparison.jl
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

By default the test suite should run fairly quickly (in a few minutes). To do so, it skips many cases. To run more comprehensive tests, you can activate the `--long` option:
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

To get more output on what tests were successful, an option `--verbose` (or `-v`) can be passed in a similar way to `--long` (if any tests fail, the output is printed by default).
