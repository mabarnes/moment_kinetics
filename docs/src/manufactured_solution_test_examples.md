# List of Manufactured Solutions Test TOML inputs

Here we list the existing manufactured solution test inputs.
These inputs are examples only, and in most cases we only
keep the lowest resolution examples. The user should copy
these inputs and make a series of TOML with increasing resolutions
to generate a series of simulations on which the numerical errors
can be tested and compared to the expected scaling of the 
numerical method employed.

## 1D1V tests

There are 1D1V tests which complement the check-in testing suite.
The example input files in this category are as follows:

* 1D1V simulation of kinetic ions (no neutrals) and numerical
  velocity dissipation.
```
runs/1D-wall_MMS_new_nel_r_1_z_16_vpa_16_vperp_1_diss.toml
```
* 1D1V simulation of kinetic ions (no neutrals) and a krook
  collision operator.
```
runs/1D-wall_MMS_new_nel_r_1_z_16_vpa_16_vperp_1_krook.toml
```

## 1D2V tests

* 1D2V simulation of kinetic ions (no neutrals) and a krook
  collision operator.
```
runs/1D-wall_MMS_new_nel_r_1_z_16_vpa_8_vperp_8_krook.toml
```
* 1D2V simulation of a open field lines in 1D magnetic mirror
  (no neutrals)
```
runs/1D-mirror_MMS_ngrid_9_nel_r_1_z_4_vpa_4_vperp_2_diss.toml
runs/1D-mirror_MMS_ngrid_9_nel_r_1_z_8_vpa_8_vperp_4_diss.toml
runs/1D-mirror_MMS_ngrid_9_nel_r_1_z_16_vpa_16_vperp_8_diss.toml
runs/1D-mirror_MMS_ngrid_9_nel_r_1_z_32_vpa_32_vperp_16_diss.toml
```

## 1D2V/1D3V test (with neutrals)

* A test with periodic boundary conditions in 1D, Boltzmann
  electrons, neutrals, and ions.
```
runs/1D-sound-wave_cheb_nel_r_1_z_2_vpa_4_vperp_4.toml
```

* A test with ions and neutral species and wall boundary conditions, 
  using the Boltzmann electron response to model the electron species.
```
runs/1D-wall_cheb-with-neutrals_nel_r_1_z_2_vpa_4_vperp_4.toml
```

* A test with neutral species and wall boundary conditions, using a simple
  sheath model for electrons based on the Boltzmann electron response.
```
runs/1D-wall_cheb-with-neutrals-with-sheath_nel_r_1_z_2_vpa_4_vperp_4.toml
```

## 2D1V tests

These tests are used to test the spatial advection in simple cases
with wall boundary conditions.

* 2D1V test of wall boundary conditions and the E x B drift.
  Numerical dissipation in the radial domain is imposed to stabilise
  an instability that otherwise appears at the grid scale.
```
runs/2D-wall_MMS_nel_r_32_z_32_vpa_16_vperp_1_diss.toml
```

## 2D2V tests

These tests are used to test the spatial advection in cases
with wall boundary conditions or geometrical features where
two velocity dimensions are necessary.

* 2D2V simulation with periodic boundary conditions, 
  Boltzmann electrons and ions
```
runs/2D-sound-wave_cheb_ion_only_nel_r_2_z_2_vpa_4_vperp_4.toml
```

* 2D2V simulation of a open field lines in 1D magnetic mirror
  (no neutrals)
```
runs/2D-mirror_MMS_ngrid_5_nel_r_8_z_8_vpa_8_vperp_4_diss.toml
runs/2D-mirror_MMS_ngrid_5_nel_r_16_z_16_vpa_16_vperp_8_diss.toml
runs/2D-mirror_MMS_ngrid_5_nel_r_32_z_32_vpa_16_vperp_16_diss.toml
```

## 2D2V/2D3V tests

These tests include a two-dimensional domain, ions, and neutrals 
(which have three velocity dimensions).

* 2D2V/2D3V simulation on a domain with wall boundaries, with
  helical geometry, Boltzmann electrons, neutrals, and ions.
```
runs/2D-wall_cheb-with-neutrals_nel_r_2_z_2_vpa_4_vperp_4.toml
```

* 2D2V/2D3V simulation on a periodic domain, with
  helical geometry, Boltzmann electrons, neutrals, and ions.
```
runs/2D-sound-wave_cheb_nel_r_2_z_2_vpa_4_vperp_4.toml
```

* 2D2V/2D3V simulation on a periodic domain, with
  model charge exchange and ionisation collisions,
  helical geometry, Boltzmann electrons, neutrals, and ions.
```
runs/2D-sound-wave_cheb_cxiz_nel_r_2_z_2_vpa_4_vperp_4.toml
```
