test_type = "Kinetic electron"

test_input = Dict("n_ion_species" => 1,
                  "n_neutral_species" => 1,
                  "electron_physics" => "kinetic_electrons",
                  "run_name" => "kinetic_electron",
                  "base_directory" => test_output_directory,
                  "evolve_moments_density" => true,
                  "evolve_moments_parallel_flow" => true,
                  "evolve_moments_parallel_pressure" => true,
                  "evolve_moments_conservation" => true,
                  "recycling_fraction" => 0.5,
                  "krook_collisions" => true,
                  "T_e" => 0.2,
                  "T_wall" => 0.1,
                  "initial_density1" => 1.0,
                  "initial_temperature1" => 1.0,
                  "z_IC_option1" => "gaussian",
                  "z_IC_density_amplitude1" => 0.001,
                  "z_IC_density_phase1" => 0.0,
                  "z_IC_upar_amplitude1" => 1.0,
                  "z_IC_upar_phase1" => 0.0,
                  "z_IC_temperature_amplitude1" => 0.0,
                  "z_IC_temperature_phase1" => 0.0,
                  "vpa_IC_option1" => "gaussian",
                  "vpa_IC_density_amplitude1" => 1.0,
                  "vpa_IC_density_phase1" => 0.0,
                  "vpa_IC_upar_amplitude1" => 0.0,
                  "vpa_IC_upar_phase1" => 0.0,
                  "vpa_IC_temperature_amplitude1" => 0.0,
                  "vpa_IC_temperature_phase1" => 0.0,
                  "initial_density2" => 1.0,
                  "initial_temperature2" => 1.0,
                  "z_IC_option2" => "gaussian",
                  "z_IC_density_amplitude2" => 0.001,
                  "z_IC_density_phase2" => 0.0,
                  "z_IC_upar_amplitude2" => -1.0,
                  "z_IC_upar_phase2" => 0.0,
                  "z_IC_temperature_amplitude2" => 0.0,
                  "z_IC_temperature_phase2" => 0.0,
                  "vpa_IC_option2" => "gaussian",
                  "vpa_IC_density_amplitude2" => 1.0,
                  "vpa_IC_density_phase2" => 0.0,
                  "vpa_IC_upar_amplitude2" => 0.0,
                  "vpa_IC_upar_phase2" => 0.0,
                  "vpa_IC_temperature_amplitude2" => 0.0,
                  "vpa_IC_temperature_phase2" => 0.0,
                  "charge_exchange_frequency" => 0.75,
                  "ionization_frequency" => 0.5,
                  "constant_ionization_rate" => false,
                  "timestepping" => Dict{String,Any}("type" => "Fekete4(3)",
                                                     "nstep" => 3,
                                                     "dt" => 2.0e-8,
                                                     "minimum_dt" => 1.0e-8,
                                                     "CFL_prefactor" => 1.0,
                                                     "step_update_prefactor" => 0.4,
                                                     "nwrite" => 2,
                                                     "split_operators" => false),
                  "electron_timestepping" => Dict{String,Any}("type" => "Fekete4(3)",
                                                              "nstep" => 10,
                                                              "dt" => 4.0e-11,
                                                              "minimum_dt" => 2.0e-11,
                                                              "initialization_residual_value" => 1.e10,
                                                              "converged_residual_value" => 1.e10,
                                                              "nwrite" => 10000,
                                                              "nwrite_dfns" => 10000,
                                                              "no_restart" => true),
                  "r_ngrid" => 1,
                  "r_nelement" => 1,
                  "z_ngrid" => 3,
                  "z_nelement" => 24,
                  "z_bc" => "wall",
                  "z_discretization" => "chebyshev_pseudospectral",
                  "z_element_spacing_option" => "sqrt",
                  "vpa_ngrid" => 3,
                  "vpa_nelement" => 4,
                  "vpa_L" => 6.0,
                  "vpa_bc" => "zero",
                  "vpa_discretization" => "chebyshev_pseudospectral",
                  "vz_ngrid" => 3,
                  "vz_nelement" => 4,
                  "vz_L" => 6.0,
                  "vz_bc" => "zero",
                  "vz_discretization" => "chebyshev_pseudospectral",
                  "ion_source" => Dict("active" => true,
                                       "z_profile" => "gaussian",
                                       "z_width" => 0.125,
                                       "source_strength" => 2.0,
                                       "source_T" => 2.0),
                  "numerical_dissipation" => Dict{String,Any}("force_minimum_pdf_value" => 0.0,
                                                              "vpa_dissipation_coefficient" => 1e-2))


test_input_list = [
     test_input,
    ]
