'Recycling fraction' examples
=============================

The input files in this directory are for a 1D1V example with a central source
of plasma particles and energy, and a recycling fraction of 0.5 at the walls
providing a source of neutrals. Plasma and neutrals are coupled by charge
exchange and ionization, with ad-hoc constant rates chosen to give a
penetration depth of neutrals into the domain that is similar to the size of
the domain (to avoid as much as possible very steep gradients near the sheath
entrances).

On the numerical side, these examples use a compressed grid near the target -
to provide better resolution of steep gradients near the sheath entrance - and
adaptive, explicit timestepping.

Running the examples
--------------------

To minimise the computer time needed to reach steady state solutions, there is
an input file for an initialisation drift-kinetic run with lower-order finite
elements, which provides a good initial condition for higher resolution runs.
Drift kinetic runs allow longer timesteps, and the 'split2' case (with
separately evolved density and parallel flow) is slightly closer to the steady
state of the full moment-kinetic 'split3' case (with separately evolved
density, parallel flow, and parallel pressure), so provides a better initial
guess. The suggested sequence for runs is:
```julia
julia> using moment_kinetics
julia> run_moment_kinetics("examples/recycling-fraction/wall-bc_recyclefraction0.5-init.toml")
julia> run_moment_kinetics("examples/recycling-fraction/wall-bc_recyclefraction0.5.toml"; restart="runs/wall-bc_recyclefraction0.5-init/wall-bc_recyclefraction0.5-init.dfns.h5")
julia> run_moment_kinetics("examples/recycling-fraction/wall-bc_recyclefraction0.5_split1.toml"; restart="runs/wall-bc_recyclefraction0.5/wall-bc_recyclefraction0.5.dfns.h5")
julia> run_moment_kinetics("examples/recycling-fraction/wall-bc_recyclefraction0.5_split2.toml"; restart="runs/wall-bc_recyclefraction0.5/wall-bc_recyclefraction0.5.dfns.h5")
julia> run_moment_kinetics("examples/recycling-fraction/wall-bc_recyclefraction0.5_split3.toml"; restart="runs/wall-bc_recyclefraction0.5_split2/wall-bc_recyclefraction0.5_split2.dfns.h5")
```
The full moment-kinetic run 'split3' is still quite slow, ~6.5hrs on 8 cores.

Remember that runs can be paused by doing `touch stop` in the output directory, and restarted with, e.g.
```julia
julia> run_moment_kinetics("examples/recycling-fraction/wall-bc_recyclefraction0.5_split3.toml"; restart=true)
```

There are several other input files in this directory that use different
Runge-Kutta timestepping schemes for the full moment-kinetic 'split3' case, but
unless you are experimenting with timestepping schemes the `"Fekete4(3)"` used
for all the runs mentioned above seems to be a reasonable choice.
