test_type = "Wall boundary conditions"

# default inputs for tests
test_input_finite_difference_1D1V = Dict(
    "run_name" => "finite_difference_1D1V",
    "n_ion_species" => 2,
    "n_neutral_species" => 2,
    "boltzmann_electron_response" => true,
    "base_directory" => test_output_directory,
    "evolve_moments_density" => false,
    "evolve_moments_parallel_flow" => false,
    "evolve_moments_parallel_pressure" => false,
    "evolve_moments_conservation" => false,
    "electron_physics" => "boltzmann_electron_response",
    "T_e" => 1.0,
    "T_wall" => 1.0,
    "initial_density1" => 1.0,
    "initial_temperature1" => 1.0,
    "z_IC_option1" => "gaussian",
    "z_IC_density_amplitude1" => 0.001,
    "z_IC_density_phase1" => 0.0,
    "z_IC_upar_amplitude1" => 0.0,
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
    "z_IC_upar_amplitude2" => 0.0,
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
    "charge_exchange_frequency" => 2.0,
    "ionization_frequency" => 2.0,
    "constant_ionization_rate" => false,
    "nstep" => 3,
    "dt" => 1.0e-8,
    "nwrite" => 2,
    "use_semi_lagrange" => false,
    "n_rk_stages" => 2,
    "split_operators" => false,
    "z_ngrid" => 4,
    "z_nelement" => 1,
    "z_bc" => "periodic",
    "z_discretization" => "finite_difference",
    "vpa_ngrid" => 4,
    "vpa_nelement" => 1,
    "vpa_L" => 8.0,
    "vpa_bc" => "periodic",
    "vpa_discretization" => "finite_difference")

test_input_finite_difference_simple_sheath_1D1V = merge(
    test_input_finite_difference_1D1V,
    Dict("run_name" => "finite_difference_simple_sheath_1D1V",
         "electron_physics" => "boltzmann_electron_response_with_simple_sheath"))

test_input_finite_difference = merge(
    test_input_finite_difference_1D1V,
    Dict("run_name" => "finite_difference",
         "r_ngrid" => 4,
         "r_nelement" => 1,
         "r_discretization" => "finite_difference",
         "vperp_ngrid" => 4,
         "vperp_nelement" => 1,
         "vperp_discretization" => "finite_difference",
         "vz_ngrid" => 4,
         "vz_nelement" => 1,
         "vz_discretization" => "finite_difference",
         "vr_ngrid" => 4,
         "vr_nelement" => 1,
         "vr_discretization" => "finite_difference",
         "vzeta_ngrid" => 4,
         "vzeta_nelement" => 1,
         "vzeta_discretization" => "finite_difference"))

test_input_finite_difference_simple_sheath = merge(
    test_input_finite_difference,
    Dict("run_name" => "finite_difference_simple_sheath",
         "electron_physics" => "boltzmann_electron_response_with_simple_sheath"))

test_input_chebyshev_1D1V = merge(
    test_input_finite_difference_1D1V,
    Dict("run_name" => "chebyshev_pseudospectral_1D1V",
         "z_discretization" => "chebyshev_pseudospectral",
         "z_ngrid" => 3,
         "z_nelement" => 2,
         "vpa_discretization" => "chebyshev_pseudospectral",
         "vpa_ngrid" => 3,
         "vpa_nelement" => 2))

test_input_chebyshev_simple_sheath_1D1V = merge(
    test_input_chebyshev_1D1V,
    Dict("run_name" => "chebyshev_pseudospectral_simple_sheath_1D1V",
         "electron_physics" => "boltzmann_electron_response_with_simple_sheath"))

test_input_chebyshev = merge(
    test_input_chebyshev_1D1V,
    Dict("run_name" => "chebyshev_pseudospectral",
         "r_discretization" => "chebyshev_pseudospectral",
         "r_ngrid" => 3,
         "r_nelement" => 1,
         "vperp_discretization" => "chebyshev_pseudospectral",
         "vperp_ngrid" => 3,
         "vperp_nelement" => 1,
         "vz_discretization" => "chebyshev_pseudospectral",
         "vz_ngrid" => 3,
         "vz_nelement" => 2,
         "vr_discretization" => "chebyshev_pseudospectral",
         "vr_ngrid" => 3,
         "vr_nelement" => 1,
         "vzeta_discretization" => "chebyshev_pseudospectral",
         "vzeta_ngrid" => 3,
         "vzeta_nelement" => 1))

test_input_chebyshev_simple_sheath = merge(
    test_input_chebyshev,
    Dict("run_name" => "chebyshev_pseudospectral_simple_sheath",
         "electron_physics" => "boltzmann_electron_response_with_simple_sheath"))

test_input_list = [
     #test_input_finite_difference,
     #test_input_finite_difference_simple_sheath,
     #test_input_finite_difference_1D1V,
     #test_input_finite_difference_simple_sheath_1D1V,
     test_input_chebyshev,
     test_input_chebyshev_simple_sheath,
     test_input_chebyshev_1D1V,
     test_input_chebyshev_simple_sheath_1D1V,
    ]
