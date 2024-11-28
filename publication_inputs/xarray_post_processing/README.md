
## xarray & h5py plotting scripts for publication quality figures

This directory contains python scripts for making publication quality figures.
We briefly describe the contents of the files.

* `plot_mk_utils.py`: A series of plotting functions using matplotlib and pyplot.

* `xarray_mk_utils.py`: A series of utility functions for reading data from `moment_kinetics` output files.

* `plot_wall.py`: A script for comparing sheath-boundary simulations.

* `plot_sd.py`: A script for comparing the numerical solution from `moment_kinetics` to the analytical slowing-down solution.

* `plot_many_collisions.py`: A script for comparing simulations of the relaxation to the Maxwellian distribution in the presence of self collisions.

* `plot_error_data.py` and `plot_integration_error_data.py`: Scripts for plotting data produced by the evaluation tests of the Fokker-Planck collision operator.

The `requirements.txt` file provides a list of required modules at the last used version.
