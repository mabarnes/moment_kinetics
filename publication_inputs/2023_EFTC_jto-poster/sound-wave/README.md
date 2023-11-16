Sound wave
==========

Runs
----

All the runs for the sound-wave plots can be done quickly in serial. The
parameter scan functionality allows multiple processes to be used to launch
several runs simultaneously (using Distributed.jl rather than MPI). Here we use
8 processes as an example.

In the top-level `moment_kinetics` directory, run
```julia
$ julia --project -p 8
julia> include("publication_inputs/2023_EFTC_jto-poster/sound-wave/run-all-scans.jl")
```

Plots
-----

Note the steps here should be run after launching `julia` in the
`<moment_kinetics>/publication_inputs/2023_EFTC_jto-poster` directory, not in
the top level `moment_kinetics` directory (and not in the directory containing
this README.md - this avoids some repeated precompilation when plotting the
wall-bc runs).

### Setup

The analysis script in this directory uses James Cook's
PlasmaDispersionFunctions.jl package, as this is more accurate than a naive
implementation of the plasma dispersion function using SpecialFuncions.jl.

We also have to add the `moment_kinetics` package from the top level, which is
`../..` relative to this directory.

To set everything up, do
```julia
$ julia --project
julia>]
(2023_EFTC_jto-poster) pkg> add https://github.com/jwscook/PlasmaDispersionFunctions.jl
(2023_EFTC_jto-poster) pkg> dev ../..
(2023_EFTC_jto-poster) pkg>^D
$
```

You might also need to repeat any setup steps for MPI and HDF5 (or just copy
the `LocalPreferences.toml` file from the top-level `moment_kinetics`
directory).

### Usage

Import the script, do some post-processing, and run `make_plots()` (we again
use `-p 8` as the `post_process_all_scans()` step can run post processing for
several directories simultaneously, although `make_plots()` is serial)
```
$ julia --project -p 8
julia> include("sound-wave/post_process_scans.jl")
julia> post_process_all_scans()
julia> include("sound-wave/plot_dispersion_relation.jl")
julia> make_plots()
```
