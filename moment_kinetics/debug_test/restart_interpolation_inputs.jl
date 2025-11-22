test_type = "restart_interpolation"
using moment_kinetics.type_definitions: OptionsDict
using moment_kinetics.utils: recursive_merge

# default inputs for tests
base_input = OptionsDict(
     "output" => OptionsDict("run_name" => "base",
                             "base_directory" => test_output_directory),
     "composition" => OptionsDict("n_ion_species" => 2,
                                  "n_neutral_species" => 2,
                                  "electron_physics" => "boltzmann_electron_response",
                                  "T_e" => 1.0),
     "evolve_moments" => OptionsDict("density" => false,
                                     "parallel_flow" => false,
                                     "pressure" => false),
     "reactions" => OptionsDict("charge_exchange_frequency" => 2*π*0.1,
                                "ionization_frequency" => 0.0),
     "timestepping" => OptionsDict("nstep" => 3,
                                   "dt" => 0.0,
                                   "nwrite" => 2,
                                   "type" => "SSPRK2",
                                   "split_operators" => false),
     "z" => OptionsDict("ngrid" => 3,
                        "nelement" => 2,
                        "bc" => "periodic",
                        "discretization" => "chebyshev_pseudospectral"),
     "r" => OptionsDict("ngrid" => 1,
                        "nelement" => 1),
     "vpa" => OptionsDict("ngrid" => 3,
                          "nelement" => 2,
                          "L" => 8.0,
                          "bc" => "periodic",
                          "discretization" => "chebyshev_pseudospectral"),
     "vperp" => OptionsDict("ngrid" => 1,
                            "nelement" => 1),
     "vz" => OptionsDict("ngrid" => 3,
                         "nelement" => 2,
                         "L" => 8.0,
                         "bc" => "periodic",
                         "discretization" => "chebyshev_pseudospectral"),
     "vzeta" => OptionsDict("ngrid" => 1,
                            "nelement" => 1),
     "vr" => OptionsDict("ngrid" => 1,
                         "nelement" => 1),
     "ion_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0),
     "neutral_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0))

test_input =
    recursive_merge(base_input,
                    OptionsDict("output" => OptionsDict("run_name" => "full-f"),
                                "z" => OptionsDict("nelement" => 3),
                                "vpa" => OptionsDict("nelement" => 3),
                                "vz" => OptionsDict("nelement" => 3)))

test_input_split1 =
    recursive_merge(test_input,
                    OptionsDict("output" => OptionsDict("run_name" => "split1"),
                                "evolve_moments" => OptionsDict("density" => true)))

    test_input_split2 =
    recursive_merge(test_input_split1 ,
                    OptionsDict("output" => OptionsDict("run_name" => "split2"),
                                "evolve_moments" => OptionsDict("parallel_flow" => true)))

    test_input_split3 =
    recursive_merge(test_input_split2,
                    OptionsDict("output" => OptionsDict("run_name" => "split3"),
                                "evolve_moments" => OptionsDict("pressure" => true)))

test_input_list = [
     test_input,
     test_input_split1,
     test_input_split2,
     test_input_split3,
    ]
