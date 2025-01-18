# provide option of running from command line via 'julia moment_kinetics.jl'
using Pkg
Pkg.activate(".")

using moment_kinetics
using moment_kinetics.type_definitions: OptionsDict

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

input = OptionsDict("output" => OptionsDict("run_name" => "precompilation",
                                            "base_directory" => test_output_directory),
                    "evolve_moments" => OptionsDict("density" => true,
                                                    "parallel_flow" => true,
                                                    "parallel_pressure" => true),
                    "composition" => OptionsDict("electron_physics" => "kinetic_electrons"),
                    "r" => OptionsDict("ngrid" => 1,
                                       "nelement" => 1,
                                       "bc" => "periodic",
                                       "discretization" => "gausslegendre_pseudospectral"),
                    "z" => OptionsDict("ngrid" => 5,
                                       "nelement" => 4,
                                       "bc" => "wall",
                                       "discretization" => "gausslegendre_pseudospectral"),
                    "vperp" => OptionsDict("ngrid" => 1,
                                           "nelement" => 1,
                                           "bc" => "zero",
                                           "L" => 4.0,
                                           "discretization" => "gausslegendre_pseudospectral"),
                    "vpa" => OptionsDict("ngrid" => 7,
                                         "nelement" => 8,
                                         "bc" => "zero",
                                         "L" => 8.0,
                                         "discretization" => "gausslegendre_pseudospectral"),
                    "vzeta" => OptionsDict("ngrid" => 1,
                                           "nelement" => 1,
                                           "bc" => "zero",
                                           "L" => 4.0,
                                           "discretization" => "gausslegendre_pseudospectral"),
                    "vr" => OptionsDict("ngrid" => 1,
                                        "nelement" => 1,
                                        "bc" => "zero",
                                        "L" => 4.0,
                                        "discretization" => "gausslegendre_pseudospectral"),
                    "vz" => OptionsDict("ngrid" => 7,
                                        "nelement" => 8,
                                        "bc" => "zero",
                                        "L" => 8.0,
                                        "discretization" => "gausslegendre_pseudospectral"),
                    "timestepping" => OptionsDict("type" => "KennedyCarpenterARK324",
                                                  "nstep" => 1,
                                                  "dt" => 2.0e-11),
                    "electron_timestepping" => OptionsDict("nstep" => 1,
                                                           "dt" => 2.0e-11,
                                                           "initialization_residual_value" => 1.0e10,
                                                           "converged_residual_value" => 1.0e10,
                                                           "rtol" => 1.0e10,
                                                           "no_restart" => true))


run_moment_kinetics(input)
