test_type = "Wall boundary conditions"
using moment_kinetics.type_definitions: OptionsDict

# default inputs for tests
test_input_finite_difference_1D1V = OptionsDict(
    "run_name" => "finite_difference_1D1V",
    "composition" => OptionsDict("n_ion_species" => 2,
                          "n_neutral_species" => 2,
                          "electron_physics" => "boltzmann_electron_response",                      
                          "T_e" => 1.0,
                          "T_wall" => 1.0),
    "base_directory" => test_output_directory,
    "evolve_moments_density" => false,
    "evolve_moments_parallel_flow" => false,
    "evolve_moments_parallel_pressure" => false,
    "evolve_moments_conservation" => true,
    "charge_exchange_frequency" => 2.0,
    "ionization_frequency" => 2.0,
    "constant_ionization_rate" => false,
    "timestepping" => OptionsDict("nstep" => 3,
                                       "dt" => 1.0e-8,
                                       "nwrite" => 2,
                                       "type" => "SSPRK2",
                                       "split_operators" => false),
    "r_ngrid" => 1,
    "r_nelement" => 1,
    "z_ngrid" => 4,
    "z_nelement" => 1,
    "z_bc" => "periodic",
    "z_discretization" => "finite_difference",
    "vpa_ngrid" => 4,
    "vpa_nelement" => 8,
    "vpa_L" => 8.0,
    "vpa_bc" => "zero",
    "vpa_discretization" => "finite_difference",
    "vperp_ngrid" => 1,
    "vperp_nelement" => 1,
    "vz_ngrid" => 4,
    "vz_nelement" => 1,
    "vz_L" => 8.0,
    "vz_bc" => "periodic",
    "vz_discretization" => "finite_difference",
    "vzeta_ngrid" => 1,
    "vzeta_nelement" => 1,
    "vr_ngrid" => 1,
    "vr_nelement" => 1)

test_input_finite_difference_simple_sheath_1D1V = merge(
    test_input_finite_difference_1D1V,
    OptionsDict("run_name" => "finite_difference_simple_sheath_1D1V",
         "electron_physics" => "boltzmann_electron_response_with_simple_sheath"))

test_input_finite_difference = merge(
    test_input_finite_difference_1D1V,
    OptionsDict("run_name" => "finite_difference",
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
    OptionsDict("run_name" => "finite_difference_simple_sheath",
         "electron_physics" => "boltzmann_electron_response_with_simple_sheath"))

test_input_chebyshev_1D1V = merge(
    test_input_finite_difference_1D1V,
    OptionsDict("run_name" => "chebyshev_pseudospectral_1D1V",
         "z_discretization" => "chebyshev_pseudospectral",
         "z_ngrid" => 3,
         "z_nelement" => 2,
         "vpa_discretization" => "chebyshev_pseudospectral",
         "vpa_ngrid" => 3,
         "vpa_nelement" => 2,
         "vz_discretization" => "chebyshev_pseudospectral",
         "vz_ngrid" => 3,
         "vz_nelement" => 2))

test_input_chebyshev_split1_1D1V = merge(test_input_chebyshev_1D1V,
                                     OptionsDict("run_name" => "chebyshev_pseudospectral_split1_1D1V",
                                          "evolve_moments_density" => true))

test_input_chebyshev_split2_1D1V = merge(test_input_chebyshev_split1_1D1V,
                                     OptionsDict("run_name" => "chebyshev_pseudospectral_split2_1D1V",
                                          "evolve_moments_parallel_flow" => true))

test_input_chebyshev_split3_1D1V = merge(test_input_chebyshev_split2_1D1V,
                                     OptionsDict("run_name" => "chebyshev_pseudospectral_split3_1D1V",
                                          "evolve_moments_parallel_pressure" => true))


test_input_chebyshev_simple_sheath_1D1V = merge(
    test_input_chebyshev_1D1V,
    OptionsDict("run_name" => "chebyshev_pseudospectral_simple_sheath_1D1V",
         "electron_physics" => "boltzmann_electron_response_with_simple_sheath"))

test_input_chebyshev = merge(
    test_input_chebyshev_1D1V,
    OptionsDict("run_name" => "chebyshev_pseudospectral",
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
    OptionsDict("run_name" => "chebyshev_pseudospectral_simple_sheath",
         "electron_physics" => "boltzmann_electron_response_with_simple_sheath"))

test_input_list = [
     #test_input_finite_difference,
     #test_input_finite_difference_simple_sheath,
     #test_input_finite_difference_1D1V,
     #test_input_finite_difference_simple_sheath_1D1V,
     #test_input_chebyshev,
     test_input_chebyshev_simple_sheath,
     #test_input_chebyshev_1D1V,
     #test_input_chebyshev_split1_1D1V,
     #test_input_chebyshev_split2_1D1V,
     #test_input_chebyshev_split3_1D1V,
     test_input_chebyshev_simple_sheath_1D1V,
    ]
