test_type = "restart_interpolation"

# default inputs for tests
base_input = Dict(
     "run_name" => "base",
     "n_ion_species" => 2,
     "n_neutral_species" => 2,
     "boltzmann_electron_response" => true,
     "base_directory" => test_output_directory,
     "evolve_moments_density" => false,
     "evolve_moments_parallel_flow" => false,
     "evolve_moments_parallel_pressure" => false,
     "evolve_moments_conservation" => true,
     "T_e" => 1.0,
     "initial_density1" => 0.5,
     "initial_temperature1" => 1.0,
     "initial_density2" => 0.5,
     "initial_temperature2" => 1.0,
     "z_IC_option1" => "sinusoid",
     "z_IC_density_amplitude1" => 0.001,
     "z_IC_density_phase1" => 0.0,
     "z_IC_upar_amplitude1" => 0.0,
     "z_IC_upar_phase1" => 0.0,
     "z_IC_temperature_amplitude1" => 0.0,
     "z_IC_temperature_phase1" => 0.0,
     "z_IC_option2" => "sinusoid",
     "z_IC_density_amplitude2" => 0.001,
     "z_IC_density_phase2" => 0.0,
     "z_IC_upar_amplitude2" => 0.0,
     "z_IC_upar_phase2" => 0.0,
     "z_IC_temperature_amplitude2" => 0.0,
     "z_IC_temperature_phase2" => 0.0,
     "charge_exchange_frequency" => 2*π*0.1,
     "ionization_frequency" => 0.0,
     "nstep" => 3,
     "dt" => 0.0,
     "nwrite" => 2,
     "use_semi_lagrange" => false,
     "n_rk_stages" => 2,
     "split_operators" => false,
     "z_ngrid" => 3,
     "z_nelement" => 2,
     "z_bc" => "periodic",
     "z_discretization" => "chebyshev_pseudospectral",
     "r_ngrid" => 1,
     "r_nelement" => 1,
     "vpa_ngrid" => 3,
     "vpa_nelement" => 2,
     "vpa_L" => 8.0,
     "vpa_bc" => "periodic",
     "vpa_discretization" => "chebyshev_pseudospectral",
     "vperp_ngrid" => 1,
     "vperp_nelement" => 1,
     "vz_ngrid" => 3,
     "vz_nelement" => 2,
     "vz_L" => 8.0,
     "vz_bc" => "periodic",
     "vz_discretization" => "chebyshev_pseudospectral",
     "vzeta_ngrid" => 1,
     "vzeta_nelement" => 1,
     "vr_ngrid" => 1,
     "vr_nelement" => 1,
     "numerical_dissipation" => Dict{String,Any}("force_minimum_pdf_value" => 0.0))

test_input =
    merge(base_input,
          Dict("run_name" => "split1",
               "z_nelement" => 3,
               "vpa_nelement" => 3,
               "vz_nelement" => 3,
               "evolve_moments_density" => true))

test_input_split1 =
    merge(test_input,
          Dict("run_name" => "split1",
               "evolve_moments_density" => true))

test_input_split2 =
    merge(test_input_split1 ,
          Dict("run_name" => "split2",
               "evolve_moments_parallel_flow" => true))

test_input_split3 =
    merge(test_input_split2,
          Dict("run_name" => "split3",
               "evolve_moments_parallel_pressure" => true))

test_input_list = [
     test_input,
     test_input_split1,
     test_input_split2,
     test_input_split3,
    ]
