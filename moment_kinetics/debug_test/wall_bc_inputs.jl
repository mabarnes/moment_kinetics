test_type = "Wall boundary conditions"
using moment_kinetics.type_definitions: OptionsDict
using moment_kinetics.utils: recursive_merge

# default inputs for tests
test_input_finite_difference_1D1V = OptionsDict(
    "output" => OptionsDict("run_name" => "finite_difference_1D1V",
                            "base_directory" => test_output_directory),
    "composition" => OptionsDict("n_ion_species" => 2,
                          "n_neutral_species" => 2,
                          "electron_physics" => "boltzmann_electron_response",                      
                          "T_e" => 1.0,
                          "T_wall" => 1.0),
    "evolve_moments" => OptionsDict("density" => false,
                                    "parallel_flow" => false,
                                    "pressure" => false),
    "reactions" => OptionsDict("charge_exchange_frequency" => 2.0,
                               "ionization_frequency" => 2.0),
    "timestepping" => OptionsDict("nstep" => 3,
                                       "dt" => 1.0e-8,
                                       "nwrite" => 2,
                                       "type" => "SSPRK2",
                                       "split_operators" => false),
    "r" => OptionsDict("ngrid" => 1,
                       "nelement" => 1),
    "z" => OptionsDict("ngrid" => 4,
                       "nelement" => 1,
                       "bc" => "periodic",
                       "discretization" => "finite_difference"),
    "vpa" => OptionsDict("ngrid" => 4,
                         "nelement" => 8,
                         "L" => 8.0,
                         "bc" => "zero",
                         "discretization" => "finite_difference"),
    "vperp" => OptionsDict("ngrid" => 1,
                           "nelement" => 1),
    "vz" => OptionsDict("ngrid" => 4,
                        "nelement" => 1,
                        "L" => 8.0,
                        "bc" => "periodic",
                        "discretization" => "finite_difference"),
    "vzeta" => OptionsDict("ngrid" => 1,
                           "nelement" => 1),
    "vr" => OptionsDict("ngrid" => 1,
                        "nelement" => 1),
   )

test_input_finite_difference_simple_sheath_1D1V = recursive_merge(
    test_input_finite_difference_1D1V,
    OptionsDict("output" => OptionsDict("run_name" => "finite_difference_simple_sheath_1D1V"),
                "composition" => OptionsDict("electron_physics" => "boltzmann_electron_response_with_simple_sheath"))) 

test_input_finite_difference = recursive_merge(
    test_input_finite_difference_1D1V,
    OptionsDict("output" => OptionsDict("run_name" => "finite_difference"),
                "r" => OptionsDict("ngrid" => 4,
                                   "nelement" => 1,
                                   "discretization" => "finite_difference"),
                "vperp" => OptionsDict("ngrid" => 4,
                                       "nelement" => 1,
                                       "discretization" => "finite_difference"),
                "vz" => OptionsDict("ngrid" => 4,
                                    "nelement" => 1,
                                    "discretization" => "finite_difference"),
                "vr" => OptionsDict("ngrid" => 4,
                                    "nelement" => 1,
                                    "discretization" => "finite_difference"),
                "vzeta" => OptionsDict("ngrid" => 4,
                                       "nelement" => 1,
                                       "discretization" => "finite_difference"),
               ))

test_input_finite_difference_simple_sheath = recursive_merge(
    test_input_finite_difference,
    OptionsDict("output" => OptionsDict("run_name" => "finite_difference_simple_sheath"),
         "composition" => OptionsDict("electron_physics" => "boltzmann_electron_response_with_simple_sheath")))

test_input_chebyshev_1D1V = recursive_merge(
    test_input_finite_difference_1D1V,
    OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_1D1V"),
                "z" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                   "ngrid" => 3,
                                   "nelement" => 2),
                "vpa" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                     "ngrid" => 3,
                                     "nelement" => 2),
                "vz" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                    "ngrid" => 3,
                                    "nelement" => 2),
               ))

test_input_chebyshev_split1_1D1V = recursive_merge(test_input_chebyshev_1D1V,
                                                   OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_split1_1D1V"),
                                                               "evolve_moments" => OptionsDict("density" => true)))

test_input_chebyshev_split2_1D1V = recursive_merge(test_input_chebyshev_split1_1D1V,
                                                   OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_split2_1D1V"),
                                                               "evolve_moments" => OptionsDict("parallel_flow" => true)))

test_input_chebyshev_split3_1D1V = recursive_merge(test_input_chebyshev_split2_1D1V,
                                                   OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_split3_1D1V"),
                                                               "evolve_moments" => OptionsDict("pressure" => true)))


test_input_chebyshev_simple_sheath_1D1V = recursive_merge(
    test_input_chebyshev_1D1V,
    OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_simple_sheath_1D1V"),
                "composition" => OptionsDict("electron_physics" => "boltzmann_electron_response_with_simple_sheath")))

test_input_chebyshev = recursive_merge(
    test_input_chebyshev_1D1V,
    OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral"),
                "r" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                   "ngrid" => 3,
                                   "nelement" => 1),
                "vperp" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                       "ngrid" => 3,
                                       "nelement" => 1),
                "vz" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                    "ngrid" => 3,
                                    "nelement" => 2),
                "vr" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                    "ngrid" => 3,
                                    "nelement" => 1),
                "vzeta" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                       "ngrid" => 3,
                                       "nelement" => 1),
               ))

test_input_chebyshev_simple_sheath = recursive_merge(
    test_input_chebyshev,
    OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_simple_sheath"),
         "composition" => OptionsDict("electron_physics" => "boltzmann_electron_response_with_simple_sheath")))

test_input_chebyshev_debug_io = recursive_merge(
    test_input_chebyshev,
    OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_debug_io"),
                "timestepping" => OptionsDict("debug_io" => true)))

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
     test_input_chebyshev_split3_1D1V,
     #test_input_chebyshev_simple_sheath_1D1V,
     test_input_chebyshev_debug_io,
    ]
