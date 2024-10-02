This directory contains input files for some kinetic electron simulations that
are known to run (and probably some other experimental input files too). Inputs
that are expected to work:
* Wall bc with uniform grid. First converge a Boltzmann-electron simulation to
  steady state, then restart kinetic electron simulation from that, e.g.
  ```julia
  run_moment_kinetics("wall-bc_recyclefraction0.5_split3_boltzmann-coarse_tails-uniform-z-init.toml")
  run_moment_kinetics("wall-bc_recyclefraction0.5_split3_boltzmann-coarse_tails-uniform-z.toml; restart="runs/wall-bc_recyclefraction0.5_split3_boltzmann-coarse_tails-uniform-z-init/wall-bc_recyclefraction0.5_split3_boltzmann-coarse_tails-uniform-z-init.dfns.h5")
  run_moment_kinetics("wall-bc_recyclefraction0.5_split3_kinetic-coarse_tails-uniform-z-PareschiRusso2222.toml"; restart="runs/wall-bc_recyclefraction0.5_split3_boltzmann-coarse_tails-uniform-z/wall-bc_recyclefraction0.5_split3_boltzmann-coarse_tails-uniform-z.dfns.h5")
  ```
