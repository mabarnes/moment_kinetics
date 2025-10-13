test_type = "sound_wave"
using moment_kinetics.type_definitions: OptionsDict
using moment_kinetics.utils: recursive_merge
using moment_kinetics.file_io: io_has_implementation
using moment_kinetics.input_structs: netcdf

# default inputs for tests
test_input_finite_difference_1D1V = OptionsDict("output" => OptionsDict("run_name" => "finite_difference_1D1V",
                                                                        "base_directory" => test_output_directory),
                                                "composition" => OptionsDict("n_ion_species" => 2,
                                                                             "n_neutral_species" => 2,
                                                                             "electron_physics" => "boltzmann_electron_response",
                                                                             "T_e" => 1.0),
                                                "evolve_moments" => OptionsDict("density" => false,
                                                                                "parallel_flow" => false,
                                                                                "pressure" => false,
                                                                                "moments_conservation" => true),
                                                "reactions" => OptionsDict("charge_exchange_frequency" => 2*π*0.1,
                                                                           "ionization_frequency" => 0.0),
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
recursive_merge(test_input_finite_difference_1D1V,
                OptionsDict("output" => OptionsDict("run_name" => "finite_difference_1D1V_split_1_moment"),
                            "evolve_moments" => OptionsDict("density" => true)))

test_input_finite_difference_1D1V_split_2_moments =
recursive_merge(test_input_finite_difference_1D1V_split_1_moment,
                OptionsDict("output" => OptionsDict("run_name" => "finite_difference_1D1V_split_2_moments"),
                            "evolve_moments" => OptionsDict("parallel_flow" => true)))

test_input_finite_difference_1D1V_split_3_moments =
recursive_merge(test_input_finite_difference_1D1V_split_2_moments,
                OptionsDict("output" => OptionsDict("run_name" => "finite_difference_1D1V_split_3_moments"),
                            "evolve_moments" => OptionsDict("pressure" => true)))

test_input_finite_difference_cx0_1D1V =
recursive_merge(test_input_finite_difference_1D1V,
                OptionsDict("output" => OptionsDict("run_name" => "finite_difference_cx0_1D1V"),
                            "reactions" => OptionsDict("charge_exchange_frequency" => 0.0)))

test_input_finite_difference_cx0_1D1V_split_1_moment =
recursive_merge(test_input_finite_difference_cx0_1D1V,
                OptionsDict("output" => OptionsDict("run_name" => "finite_difference_cx0_1D1V_split_1_moment"),
                            "evolve_moments" => OptionsDict("density" => true)))

test_input_finite_difference_cx0_1D1V_split_2_moments =
recursive_merge(test_input_finite_difference_cx0_1D1V_split_1_moment,
                OptionsDict("output" => OptionsDict("run_name" => "finite_difference_cx0_1D1V_split_2_moments"),
                            "evolve_moments" => OptionsDict("parallel_flow" => true)))

test_input_finite_difference_cx0_1D1V_split_3_moments =
recursive_merge(test_input_finite_difference_cx0_1D1V_split_2_moments,
                OptionsDict("output" => OptionsDict("run_name" => "finite_difference_cx0_1D1V_split_3_moments"),
                            "evolve_moments" => OptionsDict("pressure" => true)))

test_input_finite_difference =
recursive_merge(test_input_finite_difference_1D1V,
                OptionsDict("output" => OptionsDict("run_name" => "finite_difference"),
                            "r" => OptionsDict("ngrid" => 4,
                                               "nelement" => 1,
                                               "discretization" => "chebyshev_pseudospectral"), # finite difference discretization does not currently support `r_boundary_section` implementation of radial boundary conditions
                            "inner_r_bc_1" => OptionsDict("bc" => "periodic"),
                            "outer_r_bc_1" => OptionsDict("bc" => "periodic"),
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
recursive_merge(test_input_finite_difference,
                OptionsDict("output" => OptionsDict("run_name" => "finite_difference_split_1_moment"),
                            "evolve_moments" => OptionsDict("density" => true)))

test_input_finite_difference_split_2_moments =
recursive_merge(test_input_finite_difference_split_1_moment,
                OptionsDict("output" => OptionsDict("run_name" => "finite_difference_split_2_moments"),
                            "evolve_moments" => OptionsDict("parallel_flow" => true)))

test_input_finite_difference_split_3_moments =
recursive_merge(test_input_finite_difference_split_2_moments,
                OptionsDict("output" => OptionsDict("run_name" => "finite_difference_split_3_moments"),
                            "evolve_moments" => OptionsDict("pressure" => true)))

test_input_finite_difference_cx0 =
recursive_merge(test_input_finite_difference,
                OptionsDict("output" => OptionsDict("run_name" => "finite_difference_cx0"),
                            "reactions" => OptionsDict("charge_exchange_frequency" => 0.0)))

test_input_finite_difference_cx0_split_1_moment =
recursive_merge(test_input_finite_difference_cx0,
                OptionsDict("output" => OptionsDict("run_name" => "finite_difference_cx0_split_1_moment"),
                            "evolve_moments" => OptionsDict("density" => true)))

test_input_finite_difference_cx0_split_2_moments =
recursive_merge(test_input_finite_difference_cx0_split_1_moment,
                OptionsDict("output" => OptionsDict("run_name" => "finite_difference_cx0_split_2_moments"),
                            "evolve_moments" => OptionsDict("parallel_flow" => true)))

test_input_finite_difference_cx0_split_3_moments =
recursive_merge(test_input_finite_difference_cx0_split_2_moments,
                OptionsDict("output" => OptionsDict("run_name" => "finite_difference_cx0_split_3_moments"),
                            "evolve_moments" => OptionsDict("pressure" => true)))

test_input_chebyshev = recursive_merge(test_input_finite_difference,
                                       OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral"),
                                                   "r" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                      "ngrid" => 3,
                                                                      "nelement" => 1),
                                                   "inner_r_bc_1" => OptionsDict("bc" => "Neumann"),
                                                   "outer_r_bc_1" => OptionsDict("bc" => "Neumann"),
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
recursive_merge(test_input_chebyshev,
                OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_split_1_moment"),
                            "evolve_moments" => OptionsDict("density" => true)))

test_input_chebyshev_split_2_moments =
recursive_merge(test_input_chebyshev_split_1_moment,
                OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_split_2_moments"),
                            "evolve_moments" => OptionsDict("parallel_flow" => true)))

test_input_chebyshev_split_3_moments =
recursive_merge(test_input_chebyshev_split_2_moments,
                OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_split_3_moments"),
                            "evolve_moments" => OptionsDict("pressure" => true)))

test_input_chebyshev_cx0 =
recursive_merge(test_input_chebyshev,
                OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_cx0"),
                            "reactions" => OptionsDict("charge_exchange_frequency" => 0.0)))

test_input_chebyshev_cx0_split_1_moment =
recursive_merge(test_input_chebyshev_cx0,
                OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_cx0_split_1_moment"),
                            "evolve_moments" => OptionsDict("density" => true)))

test_input_chebyshev_cx0_split_2_moments =
recursive_merge(test_input_chebyshev_cx0_split_1_moment,
                OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_cx0_split_2_moments"),
                            "evolve_moments" => OptionsDict("parallel_flow" => true)))

test_input_chebyshev_cx0_split_3_moments =
recursive_merge(test_input_chebyshev_cx0_split_2_moments,
                OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_cx0_split_3_moments"),
                            "evolve_moments" => OptionsDict("pressure" => true)))

test_input_chebyshev_1D1V =
recursive_merge(test_input_finite_difference_1D1V,
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

test_input_chebyshev_1D1V_split_1_moment =
recursive_merge(test_input_chebyshev_1D1V,
                OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_1D1V_split_1_moment",
                                                    "parallel_io" => false),
                            "evolve_moments" => OptionsDict("density" => true)))

test_input_chebyshev_1D1V_split_2_moments =
recursive_merge(test_input_chebyshev_1D1V_split_1_moment,
                OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_1D1V_split_2_moments"),
                            "evolve_moments" => OptionsDict("parallel_flow" => true)))

test_input_chebyshev_1D1V_split_3_moments =
recursive_merge(test_input_chebyshev_1D1V_split_2_moments,
                OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_1D1V_split_3_moments"),
                            "evolve_moments" => OptionsDict("pressure" => true)))

# Use "netcdf" for a few tests to test the NetCDF I/O if it is available.
const binary_format = io_has_implementation(Val(netcdf)) ? "netcdf" : "hdf5"

test_input_chebyshev_cx0_1D1V =
recursive_merge(test_input_chebyshev_1D1V,
                OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_cx0_1D1V",
                                                    "binary_format" => binary_format),
                            "reactions" => OptionsDict("charge_exchange_frequency" => 0.0)))

test_input_chebyshev_cx0_1D1V_split_1_moment =
recursive_merge(test_input_chebyshev_cx0_1D1V,
                OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_cx0_1D1V_split_1_moment"),
                            "evolve_moments" => OptionsDict("density" => true)))

test_input_chebyshev_cx0_1D1V_split_2_moments =
recursive_merge(test_input_chebyshev_cx0_1D1V_split_1_moment,
                OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_cx0_1D1V_split_2_moments"),
                            "evolve_moments" => OptionsDict("parallel_flow" => true)))

test_input_chebyshev_cx0_1D1V_split_3_moments =
recursive_merge(test_input_chebyshev_cx0_1D1V_split_2_moments,
                OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_cx0_1D1V_split_3_moments"),
                            "evolve_moments" => OptionsDict("pressure" => true)))

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
