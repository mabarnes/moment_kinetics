test_type = "Kinetic electron"
using moment_kinetics.type_definitions: OptionsDict

test_input = OptionsDict("composition" => OptionsDict("n_ion_species" => 1,
                                                      "n_neutral_species" => 0, #1,
                                                      "electron_physics" => "kinetic_electrons",
                                                      "recycling_fraction" => 0.5,
                                                      "T_e" => 0.2,
                                                      "T_wall" => 0.1),
                         "output" => OptionsDict("run_name" => "kinetic_electron",
                                                 "base_directory" => test_output_directory),
                         "evolve_moments" => OptionsDict("density" => true,
                                                         "parallel_flow" => true,
                                                         "pressure" => true,
                                                         "moments_conservation" => true),
                         "ion_species_1" => OptionsDict("initial_density" => 1.0,
                                                        "initial_temperature" => 1.0),
                         "z_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                             "density_amplitude" => 0.001,
                                                             "density_phase" => 0.0,
                                                             "upar_amplitude" => 1.0,
                                                             "upar_phase" => 0.0,
                                                             "temperature_amplitude" => 0.0,
                                                             "temperature_phase" => 0.0),
                         "vpa_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                               "density_amplitude" => 1.0,
                                                               "density_phase" => 0.0,
                                                               "upar_amplitude" => 0.0,
                                                               "upar_phase" => 0.0,
                                                               "temperature_amplitude" => 0.0,
                                                               "temperature_phase" => 0.0),
                         #"neutral_species_1" => OptionsDict("initial_density" => 1.0,
                         #                                   "initial_temperature" => 1.0),
                         #"z_IC_neutral_species_1" => OptionsDict("initialization_option" => "gaussian",
                         #                                        "density_amplitude" => 0.001,
                         #                                        "density_phase" => 0.0,
                         #                                        "upar_amplitude" => -1.0,
                         #                                        "upar_phase" => 0.0,
                         #                                        "temperature_amplitude" => 0.0,
                         #                                        "temperature_phase" => 0.0),
                         #"vz_IC_neutral_species_1" => OptionsDict("initialization_option" => "gaussian",
                         #                                         "density_amplitude" => 1.0,
                         #                                         "density_phase" => 0.0,
                         #                                         "upar_amplitude" => 0.0,
                         #                                         "upar_phase" => 0.0,
                         #                                         "temperature_amplitude" => 0.0,
                         #                                         "temperature_phase" => 0.0),
                         "reactions" => OptionsDict("charge_exchange_frequency" => 0.75,
                                                    "ionization_frequency" => 0.5),
                         "timestepping" => OptionsDict("type" => "PareschiRusso2(2,2,2)",
                                                       "kinetic_electron_solver" => "implicit_p_implicit_pseudotimestep",
                                                       "nstep" => 3,
                                                       "dt" => 1.0e-9,
                                                       "nwrite" => 2,),
                         "electron_timestepping" => OptionsDict("dt" => 1.0e-6,
                                                                "initialization_residual_value" => 2.e3,
                                                                "converged_residual_value" => 1.e3,
                                                                "nwrite" => 10000,
                                                                "nwrite_dfns" => 10000,
                                                                "no_restart" => true),
                         #"nonlinear_solver" => OptionsDict("rtol" => 1.0e-2,
                         #                                  "atol" => 1.0e-3,),
                         "r" => OptionsDict("ngrid" => 1,
                                            "nelement" => 1),
                         "z" => OptionsDict("ngrid" => 3,
                                            "nelement" => 1,
                                            "bc" => "wall",
                                            "discretization" => "gausslegendre_pseudospectral",
                                            "element_spacing_option" => "uniform"),
                         "vpa" => OptionsDict("ngrid" => 4,
                                              "nelement" => 5,
                                              "L" => 6.0,
                                              "bc" => "zero",
                                              "element_spacing_option" => "coarse_tails",
                                              "discretization" => "gausslegendre_pseudospectral"),
                         "vz" => OptionsDict("ngrid" => 4,
                                             "nelement" => 5,
                                             "L" => 6.0,
                                             "bc" => "zero",
                                              "element_spacing_option" => "coarse_tails",
                                             "discretization" => "gausslegendre_pseudospectral"),
                         "ion_source_1" => OptionsDict("active" => true,
                                                     "z_profile" => "gaussian",
                                                     "z_width" => 0.125,
                                                     "source_strength" => 2.0,
                                                     "source_T" => 2.0),
                         "krook_collisions" => OptionsDict("use_krook" => true),
                         "ion_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0,
                                                                    "vpa_dissipation_coefficient" => 1e-2),
                         "electron_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0,
                                                                         "vpa_dissipation_coefficient" => 1e-2),
                         "neutral_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0,
                                                                        "vz_dissipation_coefficient" => 1e-2))

test_input_adi = deepcopy(test_input)
test_input_adi["output"]["run_name"] = "kinetic_electron_adi"
test_input_adi["timestepping"]["kinetic_electron_preconditioner"] = "adi"


test_input_list = [
     test_input,
     test_input_adi,
    ]
