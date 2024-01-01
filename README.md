# Getting started

The full documentation is online at [https://mabarnes.github.io/moment_kinetics](https://mabarnes.github.io/moment_kinetics).

## Setup

First clone this git repository, e.g. (to clone it into a directory with the
default name `moment_kinetics`)
```bash
$ git clone git@github.com:mabarnes/moment_kinetics
```
The command above assumes that you have an account on Github.com, and that
account has ssh keys set up. If that is not the case you can clone using https
instead
```bash
$ git clone https://github.com/mabarnes/moment_kinetics
```
When using https some things (e.g. pushing to the remote repository) may
require you to use 2-factor authentication, see
https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories#cloning-with-https-urls.

!!! warning
    Do not download the zip-file from the Github.com page. This gives you the
    source code files but does not create a git repository. We get some version
    information from git when running the code, so without the git repository
    you will not be able to run a simulation.

1) If you have already installed Julia, ensure that the Julia version is >=
    1.9.0 by doing
    ```
    $ julia --version
    ```
    at command line. The setup script in step 2 can also download a Julia
    binary if you have not already installed Julia.

2) If you are running on a desktop/laptop (rather than an HPC cluster) ensure
    that you have an MPI implementation installed (using whatever the usual way
    of installing software is on your system). It should not matter which MPI
    implementation - `openmpi` is often a good choice if you have no reason to
    prefer a particular one. Check that the MPI compiler wrapper `mpicc` is
    available, e.g.
    ```
    $ mpicc --version
    ```
    should run without an error.

3) Run the setup script
    ```
    $ machines/machine_setup.sh
    ```
    This script will prompt you for various options. The default choices should
    be sensible in most cases. On a laptop/desktop the 'name of machine to set
    up' will be 'generic-pc' and will set up for interactive use. On supported
    clusters, 'name of machine' will be the name of the cluster. On other
    clusters 'generic-batch' can be used, but requires some manual setup (see
    `machines/generic-batch-template/README.md`).

    For more information, see [`machine_setup` notes](@ref).

    If you want or need to set up 'by hand' without using
    `machines/machine_setup.sh`, see [Manual setup](@ref).

Some other notes that might sometimes be useful:

* To speed up running scripts or the first call of `run_moment_kinetics` in a
    REPL session, it is possible to compile a 'system image'
    (`moment_kinetics.so`). By running
    ```
    $ julia --project -O3 precompile.jl
    ```
    and then start Julia by running for example
    ```
    $ julia --project -O3 -Jmoment_kinetics.so 
    ```
    this significantly decreases the load time but prevents code changes from
    taking effect when `moment_kinetics.so` is used until you repeat the
    compilation of the system image. Note that this also prevents the `Revise`
    package from updating `moment_kinetics` when you edit the code during and
    interactive session.

    System images are created by default on HPC clusters, and are required to
    use the provided `jobscript-*.template` submission scripts (used by
    `submit-run.sh` and `submit-restart.sh`). This is to try and minimise the
    compilation that has to be replicated on all the (possibly thousands of)
    processes in a parallel run. After changing source code, you should run
    ```
    $ precompile-submit.sh
    ```
    (to re-compile the `moment_kinetics.so` system image).

* In the course of development, it is sometimes helpful to upgrade the Julia
    version. Upgrading the version of Julia or upgrading packages may require a
    fresh installation of `moment_kinetics`. To make a fresh install with the
    latest package versions you should be able to just run
    ```julia
    pkg> update
    ```
    (to enter 'Package mode' enter ']' at the `julia>` prompt). It might
    sometimes necessary or helpful to instead remove (or rename) the
    `Manifest.jl` file in the main directory, and re-run the setup from step 2)
    above. It can sometimes be necessary to remove or rename the `.julia/`
    directory (located by default in your home directory) to force all the
    dependencies to be rebuilt.

* When using the `Plots`-based post-processing library, one may have to set an
    environment variable to avoid error messages from the Qt library. If you
    execute the command
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
Note that the middle character in `-O3` is a capital letter 'O', not a zero. (On
HPC clusters, or if you selected the "set up separate packages for post
processing" option from `machines/machine_setup.sh`, you should use `-O3
--check-bounds=no` instead of just `-O3`, and the same in the
[Restarting](@ref) section.)

Options are specified in a TOML file, e.g. `input.toml` here. The defaults are
specified in `moment_kinetics_input.jl`.
* To run in parallel, just put `mpirun -np <n>` in front of the call you would
  normally use, with `<n>` the number of processes to use.
* It may be more convenient when running `moment_kinetics` more than once to
  work from the Julia REPL, e.g.
    ```
    $ julia -O3 --project
    julia> using moment_kinetics
    julia> run_moment_kinetics("input.toml")
    ```
    where `input` is the name of a TOML file containing the desired options. It
    is also possible to pass a `Dict()` containing any non-default options
    desired, which might sometimes be useful in tests or scripts
    ```
    julia> run_moment_kinetics(input)
    ```
    Especially when developing the code, a lot of compilation time can be saved
    by using [Revise.jl](https://timholy.github.io/Revise.jl/stable/), and
    re-running a test case in the REPL (without restarting `julia`) - this is
    enabled by default when setting up using `machines/machine_setup.sh` for
    'generic-pc'.

On an HPC cluster, you can submit a simulation (using the input file
`input.toml`) to the batch queue using the convenience script
```
$ ./submit-run.sh input.toml
```
See the help text
```
$ ./submit-run.sh -h
```
for various command line options to change parameters (e.g. number of nodes,
etc.).

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

On an HPC cluster, you can submit a restart (using the input file
`input.toml`) to the batch queue using the convenience script
```
$ ./submit-restart.sh input.toml
```
or to restart from a particular output file
```
$ ./submit-restart.sh -r runs/example/example.dfns.h5 input.toml
```
See the help text
```
$ ./submit-restart.sh -h
```
for various other command line options to change parameters (e.g. number of
nodes, etc.).

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

## Post-processing with `makie_post_processing`

The default post-processing module, written to be a bit more generic and
flexible than the original Plots-based one, and able to be used interactively,
is provided in `makie_post_processing`, see [Post processing](@ref).

On an HPC cluster, when you call `./submit-run.sh` or `./submit-restart.sh`, a
job will (by default) be submitted to run
[`makie_post_processing.makie_post_process`](@ref) or
[`plots_post_processing.analyze_and_plot_data`](@ref) (depending on which you
have set up, or on whether you pass the `-o` argument when both are set up) on
the output after the run is finished. You can skip this by passing the `-a`
argument to `./submit-run.sh` or `./submit-restart.sh`.

### Original, Plots-based post processing quickstart

This post-processing functionality is now disabled by default, but you can
enable it by entering `y` at the "Would you like to set up
plots\_post\_processing?" prompt in `machines/machine_setup.sh`.

To make plots and calculate frequencies/growth rates, run
```
$ julia --project -O3 run_post_processing.jl runs/<directory to process>
```
passing the directory to process as a command line argument. Input options
for post-processing can be specified in `post_processing_input.jl`. Note that
even when running interactively, it is necessary to restart Julia after
modifying `post_processing_input.jl`.

Post processing can be done for several directories at once using
```
$ julia --project -O3 post_processing_driver.jl runs/<directory1> runs/<directory2> ...
```
passing the directories to process as command line arguments. Optionally pass a
number as the first argument to parallelise post processing of different
directories.

## Parallel I/O

To enable parallel I/O, HDF5.jl needs to be configured to use an HDF5 library
which has MPI enabled and is compiled using the same MPI as you run Julia with.
To ensure this happens, `machines/machine_setup.sh` will download the HDF5
source code and compile a local copy of the library under `machines/artifacts`,
unless you enter `n` at the "Do you want to download, and compile a local
version of HDF5" prompt (except on known HPC clusters where an MPI-enabled HDF5
is provided by a module - this is currently true on ARCHER2 - where the
module-provided HDF5 is used).

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
* Execute some or all of the tests as a script. For example in the terminal run
    ```
    $ julia -O3 --project test/runtests.jl
    ```
    or in the REPL run
    ```
    julia> include("test/runtests.jl")
    ```
    Individual test files can also be used instead of `runtests.jl`, which runs all the tests.
* You can also run the tests using `Pkg`. Either using `pkg>` mode
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
    The downside of this method is that it will cause `NCDatasets` to be
    installed if you did not install it already, which might sometimes cause
    linking errors (related to the HDF5 library, see [Optional
    dependencies](@ref)).

By default the test suite should run fairly quickly (in a few minutes). To do
so, it skips many cases. To run more comprehensive tests, you can activate the
`--long` option:
* In the REPL, run
    ```
    julia> push!(ARGS, "--long")
    ```
    before running the tests.
* Running from the terminal, pass as a command line argument, e.g.
    ```
    $ julia -O3 --project --long test/runtests.jl
    ```
* Using `test_args` argument
    ```
    julia> Pkg.test(; test_args=["--long"])
    ```
    Note the semicolon is necessary.

To get more output on what tests were successful, an option `--verbose` (or
`-v`) can be passed in a similar way to `--long` (if any tests fail, the output
is printed by default).
