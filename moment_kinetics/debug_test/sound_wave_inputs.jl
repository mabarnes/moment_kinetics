test_type = "sound_wave"
using moment_kinetics.type_definitions: OptionsDict

# default inputs for tests
test_input_finite_difference_1D1V = OptionsDict("run_name" => "finite_difference_1D1V",
                                                "composition" => OptionsDict("n_ion_species" => 2,
                                                                             "n_neutral_species" => 2,
                                                                             "electron_physics" => "boltzmann_electron_response",
                                                                             "T_e" => 1.0),
                                                "base_directory" => test_output_directory,
                                                "evolve_moments_density" => false,
                                                "evolve_moments_parallel_flow" => false,
                                                "evolve_moments_parallel_pressure" => false,
                                                "evolve_moments_conservation" => true,
                                                "charge_exchange_frequency" => 2*π*0.1,
                                                "ionization_frequency" => 0.0,
                                                "timestepping" => OptionsDict("nstep" => 3,
                                                                              "dt" => 1.e-8,
                                                                              "nwrite" => 2,
                                                                              "type" => "SSPRK2",
                                                                              "split_operators" => false),
                                                "z" => OptionsDict("ngrid" => 4,
                                                                   "nelement" => 1,
                                                                   "bc" => "periodic",
                                                                   "discretization" => "finite_difference"),
                                                "r" => OptionsDict("ngrid" => 1,
                                                                   "nelement" => 1),
                                                "vpa" => OptionsDict("ngrid" => 4,
                                                                     "nelement" => 1,
                                                                     "L" => 8.0,
                                                                     "bc" => "periodic",
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

test_input_finite_difference_1D1V_split_1_moment =
    merge(test_input_finite_difference_1D1V,
          OptionsDict("run_name" => "finite_difference_1D1V_split_1_moment",
                      "evolve_moments_density" => true))

test_input_finite_difference_1D1V_split_2_moments =
    merge(test_input_finite_difference_1D1V_split_1_moment,
          OptionsDict("run_name" => "finite_difference_1D1V_split_2_moments",
                      "evolve_moments_parallel_flow" => true))

test_input_finite_difference_1D1V_split_3_moments =
    merge(test_input_finite_difference_1D1V_split_2_moments,
          OptionsDict("run_name" => "finite_difference_1D1V_split_3_moments",
                      "evolve_moments_parallel_pressure" => true))

test_input_finite_difference_cx0_1D1V =
    merge(test_input_finite_difference_1D1V,
          OptionsDict("run_name" => "finite_difference_cx0_1D1V",
                      "charge_exchange_frequency" => 0.0))

test_input_finite_difference_cx0_1D1V_split_1_moment =
    merge(test_input_finite_difference_cx0_1D1V,
          OptionsDict("run_name" => "finite_difference_cx0_1D1V_split_1_moment",
                      "evolve_moments_density" => true))

test_input_finite_difference_cx0_1D1V_split_2_moments =
    merge(test_input_finite_difference_cx0_1D1V_split_1_moment,
          OptionsDict("run_name" => "finite_difference_cx0_1D1V_split_2_moments",
                      "evolve_moments_parallel_flow" => true))

test_input_finite_difference_cx0_1D1V_split_3_moments =
    merge(test_input_finite_difference_cx0_1D1V_split_2_moments,
          OptionsDict("run_name" => "finite_difference_cx0_1D1V_split_3_moments",
                      "evolve_moments_parallel_pressure" => true))

test_input_finite_difference =
    merge(test_input_finite_difference_1D1V,
          OptionsDict("run_name" => "finite_difference",
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

test_input_finite_difference_split_1_moment =
    merge(test_input_finite_difference,
          OptionsDict("run_name" => "finite_difference_split_1_moment",
                      "evolve_moments_density" => true))

test_input_finite_difference_split_2_moments =
    merge(test_input_finite_difference_split_1_moment,
          OptionsDict("run_name" => "finite_difference_split_2_moments",
                      "evolve_moments_parallel_flow" => true))

test_input_finite_difference_split_3_moments =
    merge(test_input_finite_difference_split_2_moments,
          OptionsDict("run_name" => "finite_difference_split_3_moments",
                      "evolve_moments_parallel_pressure" => true))

test_input_finite_difference_cx0 =
    merge(test_input_finite_difference,
          OptionsDict("run_name" => "finite_difference_cx0",
                      "charge_exchange_frequency" => 0.0))

test_input_finite_difference_cx0_split_1_moment =
    merge(test_input_finite_difference_cx0,
          OptionsDict("run_name" => "finite_difference_cx0_split_1_moment",
                      "evolve_moments_density" => true))

test_input_finite_difference_cx0_split_2_moments =
    merge(test_input_finite_difference_cx0_split_1_moment,
          OptionsDict("run_name" => "finite_difference_cx0_split_2_moments",
                      "evolve_moments_parallel_flow" => true))

test_input_finite_difference_cx0_split_3_moments =
    merge(test_input_finite_difference_cx0_split_2_moments,
          OptionsDict("run_name" => "finite_difference_cx0_split_3_moments",
                      "evolve_moments_parallel_pressure" => true))

test_input_chebyshev = merge(test_input_finite_difference,
                             OptionsDict("run_name" => "chebyshev_pseudospectral",
                                         "r" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                            "ngrid" => 3,
                                                            "nelement" => 1),
                                         "z" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                            "ngrid" => 3,
                                                            "nelement" => 2),
                                         "vperp" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                "ngrid" => 3,
                                                                "nelement" => 1),
                                         "vpa" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                              "ngrid" => 3,
                                                              "nelement" => 2),
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

test_input_chebyshev_split_1_moment =
    merge(test_input_chebyshev,
          OptionsDict("run_name" => "chebyshev_pseudospectral_split_1_moment",
                      "evolve_moments_density" => true))

test_input_chebyshev_split_2_moments =
    merge(test_input_chebyshev_split_1_moment,
          OptionsDict("run_name" => "chebyshev_pseudospectral_split_2_moments",
                      "evolve_moments_parallel_flow" => true))

test_input_chebyshev_split_3_moments =
    merge(test_input_chebyshev_split_2_moments,
          OptionsDict("run_name" => "chebyshev_pseudospectral_split_3_moments",
                      "evolve_moments_parallel_pressure" => true))

test_input_chebyshev_cx0 =
    merge(test_input_chebyshev,
          OptionsDict("run_name" => "chebyshev_pseudospectral_cx0",
                      "charge_exchange_frequency" => 0.0))

test_input_chebyshev_cx0_split_1_moment =
    merge(test_input_chebyshev_cx0,
          OptionsDict("run_name" => "chebyshev_pseudospectral_cx0_split_1_moment",
                      "evolve_moments_density" => true))

test_input_chebyshev_cx0_split_2_moments =
    merge(test_input_chebyshev_cx0_split_1_moment,
          OptionsDict("run_name" => "chebyshev_pseudospectral_cx0_split_2_moments",
                      "evolve_moments_parallel_flow" => true))

test_input_chebyshev_cx0_split_3_moments =
    merge(test_input_chebyshev_cx0_split_2_moments,
          OptionsDict("run_name" => "chebyshev_pseudospectral_cx0_split_3_moments",
                      "evolve_moments_parallel_pressure" => true))

test_input_chebyshev_1D1V =
    merge(test_input_finite_difference_1D1V,
          OptionsDict("run_name" => "chebyshev_pseudospectral_1D1V",
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

test_input_chebyshev_1D1V_split_1_moment =
    merge(test_input_chebyshev_1D1V,
          OptionsDict("run_name" => "chebyshev_pseudospectral_1D1V_split_1_moment",
                      "evolve_moments_density" => true))

test_input_chebyshev_1D1V_split_2_moments =
    merge(test_input_chebyshev_1D1V_split_1_moment,
          OptionsDict("run_name" => "chebyshev_pseudospectral_1D1V_split_2_moments",
                      "evolve_moments_parallel_flow" => true))

test_input_chebyshev_1D1V_split_3_moments =
    merge(test_input_chebyshev_1D1V_split_2_moments,
          OptionsDict("run_name" => "chebyshev_pseudospectral_1D1V_split_3_moments",
                      "evolve_moments_parallel_pressure" => true, "runtime_plots" => true))

test_input_chebyshev_cx0_1D1V =
    merge(test_input_chebyshev_1D1V,
          OptionsDict("run_name" => "chebyshev_pseudospectral_cx0_1D1V",
                      "charge_exchange_frequency" => 0.0))

test_input_chebyshev_cx0_1D1V_split_1_moment =
    merge(test_input_chebyshev_cx0_1D1V,
          OptionsDict("run_name" => "chebyshev_pseudospectral_cx0_1D1V_split_1_moment",
                      "evolve_moments_density" => true))

test_input_chebyshev_cx0_1D1V_split_2_moments =
    merge(test_input_chebyshev_cx0_1D1V_split_1_moment,
          OptionsDict("run_name" => "chebyshev_pseudospectral_cx0_1D1V_split_2_moments",
                      "evolve_moments_parallel_flow" => true))

test_input_chebyshev_cx0_1D1V_split_3_moments =
    merge(test_input_chebyshev_cx0_1D1V_split_2_moments,
          OptionsDict("run_name" => "chebyshev_pseudospectral_cx0_1D1V_split_3_moments",
                      "evolve_moments_parallel_pressure" => true))

test_input_list = [
     test_input_finite_difference,
     #test_input_finite_difference_split_1_moment,
     #test_input_finite_difference_split_2_moments,
     #test_input_finite_difference_split_3_moments,
     #test_input_finite_difference_cx0,
     #test_input_finite_difference_cx0_split_1_moment,
     #test_input_finite_difference_cx0_split_2_moments,
     #test_input_finite_difference_cx0_split_3_moments,
     #test_input_finite_difference_1D1V,
     #test_input_finite_difference_1D1V_split_1_moment,
     #test_input_finite_difference_1D1V_split_2_moments,
     test_input_finite_difference_1D1V_split_3_moments,
     #test_input_finite_difference_cx0_1D1V,
     #test_input_finite_difference_cx0_1D1V_split_1_moment,
     #test_input_finite_difference_cx0_1D1V_split_2_moments,
     #test_input_finite_difference_cx0_1D1V_split_3_moments,
     test_input_chebyshev,
     #test_input_chebyshev_split_1_moment,
     #test_input_chebyshev_split_2_moments,
     #test_input_chebyshev_split_3_moments,
     test_input_chebyshev_cx0,
     #test_input_chebyshev_cx0_split_1_moment,
     #test_input_chebyshev_cx0_split_2_moments,
     #test_input_chebyshev_cx0_split_3_moments,
     test_input_chebyshev_1D1V,
     test_input_chebyshev_1D1V_split_1_moment,
     test_input_chebyshev_1D1V_split_2_moments,
     test_input_chebyshev_1D1V_split_3_moments,
     test_input_chebyshev_cx0_1D1V,
     #test_input_chebyshev_cx0_1D1V_split_1_moment,
     #test_input_chebyshev_cx0_1D1V_split_2_moments,
     test_input_chebyshev_cx0_1D1V_split_3_moments,
    ]
