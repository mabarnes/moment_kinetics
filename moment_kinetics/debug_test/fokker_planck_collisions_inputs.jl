using moment_kinetics.type_definitions: OptionsDict
test_type = "Fokker-Planck collisions"

# default input for test
test_input_full_f = OptionsDict(
     "output" => OptionsDict("run_name" => "full_f"),
     "timestepping" => OptionsDict("dt" => 0.0,
                                        "nstep" => 3,
                                        "nwrite" => 2,
                                        "nwrite_dfns" => 2),
     "composition" => OptionsDict("n_ion_species" => 1,
                                       "n_neutral_species" => 0,
                                       "T_e" => 1.0,
                                       "T_wall" => 1.0,
                                       "electron_physics" => "boltzmann_electron_response"),
     "evolve_moments" => OptionsDict("moments_conservation" => false,
                                     "density" => false,
                                     "parallel_flow" => false,
                                     "pressure" => false),
     "ion_species_1" => OptionsDict("initial_density" => 0.5,      
                                         "initial_temperature" => 1.0),
     "z_IC_ion_species_1" => OptionsDict("density_amplitude" => 0.001,
                                              "density_phase" => 0.0,
                                              "initialization_option" => "sinusoid",
                                              "temperature_amplitude" => 0.0,
                                              "temperature_phase" => 0.0,
                                              "upar_amplitude" => 0.0,
                                              "upar_phase" => 0.0),     
     "reactions" => OptionsDict("charge_exchange_frequency" => 0.0,
                                "ionization_frequency" => 0.0),
     "fokker_planck_collisions" => OptionsDict("use_fokker_planck" => true,
                                                    "nuii" => 1.0,
                                                    "frequency_option" => "manual"),
     "r" => OptionsDict("bc" => "periodic",
                        "discretization" => "chebyshev_pseudospectral",
                        "nelement" => 1,
                        "ngrid" => 3),
     "vpa" => OptionsDict("L" => 6.0,
                          "bc" => "zero",
                          "discretization" => "gausslegendre_pseudospectral",
                          "nelement" => 2,
                          "ngrid" => 3),
     "vperp" => OptionsDict("L" => 3.0,
                            "discretization" => "gausslegendre_pseudospectral",
                            "nelement" => 2,
                            "ngrid" => 3),
     "z" => OptionsDict("bc" => "wall",
                        "discretization" => "chebyshev_pseudospectral",
                        "nelement" => 1,
                        "ngrid" => 3),
    )

test_input_list = [
     test_input_full_f ,
    ]
