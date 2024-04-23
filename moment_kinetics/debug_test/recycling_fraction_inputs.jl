test_type = "Recycling fraction and adaptive timestepping"

test_input = Dict("n_ion_species" => 1,
                  "n_neutral_species" => 1,
                  "boltzmann_electron_response" => true,
                  "run_name" => "full-f",
                  "base_directory" => test_output_directory,
                  "evolve_moments_density" => false,
                  "evolve_moments_parallel_flow" => false,
                  "evolve_moments_parallel_pressure" => false,
                  "evolve_moments_conservation" => false,
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
                                                     "dt" => 1.0e-8,
                                                     "minimum_dt" => 1.0e-8,
                                                     "CFL_prefactor" => 1.0,
                                                     "step_update_prefactor" => 0.5,
                                                     "nwrite" => 2,
                                                     "split_operators" => false),
                  "r_ngrid" => 1,
                  "r_nelement" => 1,
                  "z_ngrid" => 3,
                  "z_nelement" => 2,
                  "z_bc" => "wall",
                  "z_discretization" => "chebyshev_pseudospectral",
                  "z_element_spacing_option" => "sqrt",
                  "vpa_ngrid" => 3,
                  "vpa_nelement" => 2,
                  "vpa_L" => 6.0,
                  "vpa_bc" => "zero",
                  "vpa_discretization" => "chebyshev_pseudospectral",
                  "vz_ngrid" => 3,
                  "vz_nelement" => 2,
                  "vz_L" => 6.0,
                  "vz_bc" => "zero",
                  "vz_discretization" => "chebyshev_pseudospectral",
                  "ion_source" => Dict("active" => true,
                                       "z_profile" => "gaussian",
                                       "z_width" => 0.125,
                                       "source_strength" => 2.0,
                                       "source_T" => 2.0))


test_input_split1 = merge(test_input,
                          Dict("run_name" => "split1",
                               "evolve_moments_density" => true,
                               "evolve_moments_conservation" => true))
test_input_split2 = merge(test_input_split1,
                          Dict("run_name" => "split2",
                               "evolve_moments_parallel_flow" => true))
test_input_split2["timestepping"] = merge(test_input_split2["timestepping"],
                                          Dict{String,Any}("step_update_prefactor" => 0.4))
test_input_split3 = merge(test_input_split2,
                          Dict("run_name" => "split3",
                               "evolve_moments_parallel_pressure" => true,
                               "vpa_nelement" => 8,
                               "vz_nelement" => 8,
                               "numerical_dissipation" => Dict{String,Any}("force_minimum_pdf_value" => 0.0,
                                                                           "vpa_dissipation_coefficient" => 1e-2)))


test_input_list = [
     test_input,
     test_input_split1,
     test_input_split2,
     test_input_split3,
    ]
