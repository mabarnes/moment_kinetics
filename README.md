# Getting started

## Basics
0) Ensure that the Julia version is >= 1.7.0 by doing
    ```
    $ julia --version
    ```
    at command line.
1) Dependencies need to be installed to the project environment. Start Julia with
    ```
    $ julia --project
    ```
    (which activates the 'project' in the current directory) (or after starting with `julia`, in the REPL type `]` to enter `pkg>` mode, enter `activate .` and then backspace to leave `pkg>` mode). Once in the `moment_kinetics` project, enter `pkg>` mode by typing `]` and then run the command
    ```
    (moment_kinetics) pkg> instantiate
    ```
    this should download and install all the dependencies.
2) For julia>=1.6, pre-compiling dependencies manually is not necessary any more due to improvements to the native pre-compilation, so this step can be skipped (although precompiling the whole `moment_kinetics` code may still be useful sometimes). To pre-compile a static image (`dependencies.so`) that includes most of the external packages required for running and post-processing, run
    ```
    $ julia -O3 precompile_dependencies.jl
    ```
    To use the precompiled code, add an option `-Jdependencies.so` when starting julia.
    It is also possible to precompile the whole package into a static image (`moment_kinetics.so`) using
    ```
    $ julia -O3 precompile.jl
    ```
   this significantly decreases the load time but prevents code changes from taking effect when `moment_kinetics.so` is used without repeating the precompilation (to use this option, replace `dependencies.so` below with `moment_kinetics.so`).
3) To run julia with optimization, type
    ```
    $julia -O3 --project run_moment_kinetics.jl
    ```
    Default input options are specified in `moment_kinetics_input.jl`. The defaults can be modified for a particular run by setting options in a TOML file, for example `input.toml`, which can be passed as an argument
    ```
    $ julia -O3 --project run_moment_kinetics.jl input.toml
    ```
    * It may be more convenient when running `moment_kinetics` more than once to work from the Julia REPL - see [Developing](#developing).
4) To make plots and calculate frequencies/growth rates, run
    ```
    $ julia --project run_post_processing.jl runs/<directory to process>
    ```
    passing the directory to process as a command line argument. Input options for post-processing can be specified in post_processing_input.jl.

5) Parameter scans (see [Running parameter scans](#running-parameter-scans)) or performance tests can be performed by running
    ```
    $ julia -O3 --project driver.jl
    ```
    If running a scan, it can be parallelised by passing the number of processors as an argument. Scan options are set in `scan_inputs.jl`.

6) Post processing can be done for several directories at once using
    ```
    $ julia --project post_processing_driver.jl runs/<directory1> runs/<directory2> ...
    ```
    passing the directories to process as command line arguments. Optionally pass a number as the first argument to parallelise post processing of different directories. Input options for post-processing can be specified in `post_processing_input.jl`.

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

## Developing
* If you need to add a dependency, start the REPL with the `moment_kinetics` package activated (see [above](#moment_kinetics)), enter `pkg>` mode (press `]`) and then to add, for example, the `FFTW.jl` package enter
    ```
    (moment_kinetics) pkg> add FFTW
    ```
    This should take care of adding the package (`FFTW`) to the `Project.toml` and `Manifest.toml` files.
* When working on the code, one way to avoid waiting for everything to recompile frequently is to load the Revise.jl package
    ```
    julia> using Revise
    ```
    `Revise.jl` will recompile each edited function/method as needed, so it is possible to keep a REPL session open and avoid long recompilation. `moment_kinetics` can be run fairly conveniently from the REPL
    ```
    julia> using moment_kinetics
    julia> run_moment_kinetics(input)
    ```
    where `input` is a `Dict()` containing any non-default options desired. Input can also be loaded from a TOML file passing the filaname as a String to the second argument, e.g.
    ```
    julia> run_moment_kinetics("input.toml")
    ```
    It might be convenient to add `using Revise` to your `startup.jl` file (`~/julia/config/startup.jl`) so it's always loaded.
* Parallelization: the code is parallelized at the moment using MPI and shared-memory arrays. Arrays representing the pdf, moments, etc. are shared between all processes. Using shared memory means, for example, we can take derivatives along one dimension while parallelising the other for any dimension without having to communicate to re-distribute the arrays.
    * To run in parallel, just put `mpirun -np <n>` in front of the call you would normally use, with `<n>` the number of processes to use.
    * It is possible to use a REPL workflow with parallel code:
        * Recommended option is to use [tmpi](https://github.com/Azrael3000/tmpi). This utility (it's a bash script that uses `tmux`) starts an mpi program with each process in a separate pane in a single terminal, and mirrors input to all processes simultaneously (which is normally what you want, there are also commands to 'zoom in' on a single process).
        * Another 'low-tech' possibilty is to use something like `mpirun -np 4 xterm -e julia --project`, but that will start each process in a separate xterm and you would have to enter commands separately in each one. Occasionally useful for debugging when nothing else is available.
    * There is no restriction on the number of processes or number of grid points, although load-balancing may be affected - if there are only very few points per process, and a small fraction of processes have an extra grid point (e.g. splitting 5 points over 4 processes, so 3 process have 1 point but 1 process has 2 points), many processes will spend time waiting for the few with an extra point.
    * Parallelism is implemented through macros that get the local ranges of points that each process should handle. The inner-most level of nested loops is typically not parallelized, to allow efficient FFTs for derivatives, etc. A loop over one (possibly parallelized) dimension can be written as, for example,
        ```
        @loop_s is begin
            f[is] = ...
        end
        ```
        These macros can be nested as needed for relatively complex loops
        ```
        @loop_s is begin
            some_setup(is)
            @loop_z iz begin
                @views do_something(f[:,iz,is])
            end
            @loop_z iz begin
                @views do_something_else(f[:,iz,is])
            end
        end
        ```
        Simpler nested loops can (optionally) be written more compactly
        ```
        @loop_s_z_vpa is iz ivpa begin
            f[ivpa,iz,is] = ...
        end
        ```
        Which dimensions are actually parallelized by these macros is controlled by the 'region' that the code is currently in, as set by the `begin_<dims>_region()` functions, which <dims> are the dimensions that will be parallelized in the following region. For example, after calling `begin_s_z_region()` loop over species and z will be divided up over the processes in a 'block' (currently there is only one block, which contains the whole grid and all the processes being used, as we have not yet implemented distributed-memory parallelism). Every process will loop over all points in the remaining dimensions.
        * In a region after `begin_serial_region()`, the rank 0 process in each block will loop over all points in every dimension, and all other ranks will not loop over any.
        * Inside serial regions, the macro `@serial_region` can also be used to wrap blocks of code so that they only run on rank 0 of the block. This is useful for example to allow the use of array-broadcast expressions during initialization where performance is not critical.
        * To help show how these macros work, a script is provided that print a set of examples where the loop macros are expanded. It can be run from the Julia REPL
            ```
            $ julia --project
                           _
               _       _ _(_)_     |  Documentation: https://docs.julialang.org
              (_)     | (_) (_)    |
               _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
              | | | | | | |/ _` |  |
              | | |_| | | | (_| |  |  Version 1.7.0 (2021-11-30)
             _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
            |__/                   |

            julia> include("util/print-macros.jl")
            ```
            or on the command line
            ```
            $ julia --project util/print-macros.jl
            ```
    * The ranges used are stored in a `LoopRanges` struct in the `Ref` variable `loop_ranges` (which is exported by the `looping` module). Occasionally it is useful to access the range directly. For example the range looped over by the macro `@s_z_loop_s` is `loop_ranges[].s_z_range_s` (same prefix/suffix meanings as the macro).
            * The square brackets `[]` are needed because `loop_ranges` is a reference to a `LoopRanges` object `Ref{LoopRanges}` (a bit like a pointer) - it allows `loop_ranges` to be a `const` variable, so its type is always known at compile time, but the actual `LoopRanges` can be set/modified at run-time.
    * It is also possible to run a block of code in serial (on just the rank-0 member of each block of processes) by wrapping it in a `@serial_region` macro. This is mostly useful for initialization or file I/O where performance is not critical. For example
        ```
        @serial_region begin
            # Do some initialization
            f .= 0.0
        end
        ```
    * In any loops with the same prefix (whether type 1 or type 2) the same points belong to each process, so several loops can be executed without synchronizing the different processes. It is (mostly) only when changing the 'type' of loop (i.e. which dimensions it loops over) that synchronization is necessary, or when changing from 'serial region(s)' to parallel loops. To aid clarity and to allow some debugging routines to be added, the synchronization is done with functions labelled with the loop type. For example `begin_s_z_region()` should be called before `@s_z_loop` or `@s_z_loop_*` is called, and after any `@serial_region` or other type of `@*_loop*` macro. `begin_serial_region()` should be called before `@serial_region`.
        * Internally, the `begin_*_region()` functions call `_block_synchronize()`, which calls `MPI.Barrier()`. When all debugging is disabled they are equivalent to `MPI.Barrier()`. Having different functions allow extra consistency checks to be done when debugging is enabled, see `debug_test/README.md`.
    * For information on race conditions and debugging, see `debug_test/README.md`.
