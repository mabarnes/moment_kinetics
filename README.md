# moment_kinetics
0) Ensure that the Julia version is >= 1.6.1 by doing
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
2) Create a subdirectory to store run output.
    ```
    $ mkdir runs
    ```
3) For julia>=1.6, pre-compiling dependencies manually is not necessary any more due to improvements to the native pre-compilation, so this step can be skipped (although precompiling the whole `moment_kinetics` code may still be useful sometimes). To pre-compile a static image (`dependencies.so`) that includes most of the external packages required for running and post-processing, run
    ```
    $ julia -O3 precompile_dependencies.jl
    ```
    To use the precompiled code, add an option `-Jdependencies.so` when starting julia.
    It is also possible to precompile the whole package into a static image (`moment_kinetics.so`) using
    ```
    $ julia -O3 precompile.jl
    ```
   this significantly decreases the load time but prevents code changes from taking effect when `moment_kinetics.so` is used without repeating the precompilation (to use this option, replace `dependencies.so` below with `moment_kinetics.so`). Note that these commands will fail if the `runs` subdirectory does not exist.
4) To run julia with optimization, type
    ```
    $julia -O3 --project run_moment_kinetics.jl
    ```
    Default input options are specified in `moment_kinetics_input.jl`. The defaults can be modified for a particular run by setting options in a TOML file, for example `input.toml`, which can be passed as an argument
    ```
    $ julia -O3 --project run_moment_kinetics.jl input.toml
    ```
    * It may be more convenient when running `moment_kinetics` more than once to work from the Julia REPL - see [Developing](#developing).
5) To make plots and calculate frequencies/growth rates, run
    ```
    $ julia --project run_post_processing.jl runs/<directory to process>
    ```
    passing the directory to process as a command line argument. Input options for post-processing can be specified in post_processing_input.jl.

6) Parameter scans (see [Running parameter scans](#running-parameter-scans)) or performance tests can be performed by running
    ```
    $ julia -O3 --project driver.jl
    ```
    If running a scan, it can be parallelised by passing the number of processors as an argument. Scan options are set in `scan_inputs.jl`.

7) Post processing can be done for several directories at once using
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
