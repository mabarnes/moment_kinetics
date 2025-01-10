Fokker Planck collision operator
===============================================

We implement the nonlinear Fokker-Planck collision operator for self collisions 
using the weak-form finite-element method. This is documented in the 
[ExCALIBUR/NEPTUNE report 2070839-TN-07](https://excalibur-neptune.github.io/Documents/TN-07_AHigherOrderFiniteElementImplementationFullFlandauFokkerPlanckCollisionOperatorC.html).
A publication based on this report is in progress. Full online documentation will follow.

Input parameters
===============================================

A series of 0D2V Fokker-Planck input files can be found in

    examples/fokker-planck/

and examples of 1D2V pre-sheath simulations with the Fokker-Planck collision operator
can be found in

    examples/fokker-planck-1D2V/
    
noting that the timestepping or resolution parameters may require modification to find
a converged simulation.

The basic input namelist is structured as follows
```
[fokker_planck_collisions]
use_fokker_planck = true
# nuii sets the normalised input C[F,F] Fokker-Planck collision frequency
# for frequency_option = "manual"
nuii = 1.0
frequency_option = "manual"
```
Set `use_fokker_planck=false` to turn off Fokker-Planck collisions 
without commenting out the namelist.
The default option for `frequency_option = "reference_parameters"`, where `nuii` is set
by the reference parameter inputs. Further specialised input parameters can be
seen in the source at  `setup_fkpl_collisions_input()` in `moment_kinetics/src/fokker_planck.jl`.