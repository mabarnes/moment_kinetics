Setup for `moment_kinetics` on known clusters
=============================================

This subdirectory provides scripts to set up Julia and `moment_kinetics` to run
on some known clusters.

Currently supported:
* "archer" ([ARCHER2](https://www.archer2.ac.uk/))

Quickstart
----------

From the top-level of the `moment_kinetics` repo, run
```shell
$ machines/machine_setup.sh [<path-to-Julia-executable>]
```
If you omit the `<path-to-Julia-executable>` argument, you will be prompted to
enter it, with a default taken from your `$PATH`.

The script will prompt you for several settings, with sensible defaults (where
possible).

You will be prompted to enter a location for your `.julia` directory. If you
are installing on a cluster which allows access to your home directory from
compute nodes, it is fine to leave this as the default. If not (e.g. on
ARCHER2), you need to set a path which is accessible from the compute nodes.
If you want to create a completely self-contained install (e.g. for
reproducibility or for debugging some dependency conflicts), you might want to
put `.julia` within the `moment_kinetics` directory (e.g. enter `.` at the
prompt).

Advanced usage
--------------

The convenience script `machine_setup.sh` is provide because the actual setup
happens in multiple stages, with Julia being restarted in between (as this is
required on some machines):
1. Start Julia (using the Julia executable you want to use in the end for
   simulations) from the top level of the `moment_kinetics` repo, but not
   requiring the `--project` flag; run
   `include("machines/shared/machine_setup.jl")`; call the
   [`moment_kinetics.machine_setup_moment_kinetics`](@ref) function with
   appropriate settings as arguments (see the docstring).
     * This function requires no packages outside the base Julia system image
       (because on some machines it is desirable not to have to install any
       packages before setting `JULIA_DEPOT_PATH`, which can be done by this
       function).
     * It saves some settings into `LocalPreferences.toml`; makes a symlink at
       `bin/julia` to the Julia executable being used; saves environment setup
        into the `julia.env` file (which can include setting the
        `JULIA_DEPOT_PATH` environment variable to ensure the `.julia`
        directory is accessible on compute nodes); and if necessary symlinks a
        machine-specific file from
        `machines/shared/machine_setup_stage_two.jl`.
     * Usually Julia needs to be restarted after running this function, so it
       will call `exit()` to stop Julia. This can be avoided by passing the
       `no_force_exit=true` argument if that is useful.
2. Run `source julia.env` in each terminal session where you want to use
   `moment_kinetics`, or add it to your `.bashrc` (if this does not conflict
   with any other projects).
     * Note that `julia.env` runs `module purge` to remove any already loaded
       modules (to get a clean environment). It is therefore very likely to
       interfere with other projects.
3. If it is necessary to run a second stage of setup, for example after Julia
   is restarted with a special `JULIA_DEPOT_PATH`, start Julia again, this time
   with the `--project` flag. A symlink has been created, so you can use
   ```shell
   $ bin/julia --project
   ```
   and run `include("machines/shared/machine_setup_stage_two.jl")`.
     * This script sets up MPI and HDF5.
     * It exits Julia when it is finished, as Julia must be restarted when
       `MPIPreferences` settings are changed.
4. If you want to build a precompiled system image (to speed up startup of
   simulation runs), run
   ```shell
   $ precompile-submit.sh
   ```
   which will submit a serial or debug job that runs `precompile.jl` to create
   the `moment_kinetics.so` image.
     * This is required if using the `submit-run.sh` script, unless you edit
       the jobscript templates in the appropriate subdirectory
       `machines/<your-machine>`.

API documentation
-----------------

```@autodocs
Modules = [moment_kinetics.machine_setup]
```
