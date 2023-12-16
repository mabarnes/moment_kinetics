Introduction
============

The inputs and scripts in this directory were used to produce the figures for
John Omotani's poster at EFTC 2023. This README explains how to reproduce the
runs and plots. Note that the inputs in this directory are not tested
automatically, so they may be broken by future updates to `moment_kinetics`.
They were tested using the branch at
[PR #147](https://github.com/mabarnes/moment_kinetics/pull/147), using
julia-1.9.3 and the Manifest.toml file that was temporarily committed in
344b559c2e79b6586e6fdb8016ee4863c672a3f4.

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

Note the steps here should be run after launching `julia` in the directory
containing this README.md, not in the top level `moment_kinetics` directory.

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

Wall BC
=======

Runs
----

[You might want to use MPI to speed up the runs, see the main `moment_kinetics`
documentation.]

We run the 'full-f' version from scratch, then restart the moment-kinetic
version from that to save time.

Note that the plots shown on the poster used a uniform element spacing, but the
'split3' run does not quite converge to residuals less than 1e-3 with a uniform
grid for the resolutions used here, due to some numerical noise in the elements
next to the sheath edge boundaries (i.e. the first and last element in z).
Using sqrt-scaled element sizes (`z_element_spacing_option = "sqrt"`) resolves
the solution near the sheath entrances better, and in that case the 'split3'
simulation does converge to residuals less than 1e-3, so this option is the one
set now in all the `wall-bc/wall-bc_recyclefraction0.5*.toml` input files (as
the 'split3' simulation needs to be restarted from a simulation that used the
sqrt-spaced grid to avoid numerical instability).

In the top-level `moment_kinetics` directory, run
```julia
$ julia --project
julia> using moment_kinetics
julia> run_moment_kinetics("publication_inputs/2023_EFTC_jto-poster/wall-bc/wall-bc_recyclefraction0.5.toml")
julia> run_moment_kinetics("publication_inputs/2023_EFTC_jto-poster/wall-bc/wall-bc_recyclefraction0.5_split1.toml"; restart="runs/wall-bc_recyclefraction0.5/wall-bc_recyclefraction0.5.dfns.h5")
julia> run_moment_kinetics("publication_inputs/2023_EFTC_jto-poster/wall-bc/wall-bc_recyclefraction0.5_split2.toml"; restart="runs/wall-bc_recyclefraction0.5/wall-bc_recyclefraction0.5.dfns.h5")
julia> run_moment_kinetics("publication_inputs/2023_EFTC_jto-poster/wall-bc/wall-bc_recyclefraction0.5_split3.toml"; restart="runs/wall-bc_recyclefraction0.5/wall-bc_recyclefraction0.5.dfns.h5")
```

Plots
-----

In the directory containing this README.md (after already doing the setup from
the 'Sound wave' section) run
```julia
$ julia --project
julia> include("wall-bc/wall-bc-plots.jl")
```
