module DebugIOTest

using moment_kinetics
using moment_kinetics.type_definitions: OptionsDict

test_input = OptionsDict("output" => OptionsDict("run_name" => "debug_IO_test"),
                         "composition" => OptionsDict("n_ion_species" => 2,
                                                      "n_neutral_species" => 2,
                                                      "electron_physics" => "boltzmann_electron_response",
                                                      "T_e" => 1.0),
                         "evolve_moments" => OptionsDict("density" => true,
                                                         "parallel_flow" => true,
                                                         "pressure" => true,
                                                         "moments_conservation" => true),
                         "reactions" => OptionsDict("charge_exchange_frequency" => 2*π*0.1,
                                                    "ionization_frequency" => 0.0),
                         "timestepping" => OptionsDict("nstep" => 3,
                                                       "dt" => 1.e-8,
                                                       "nwrite" => 2,
                                                       "type" => "SSPRK2",
                                                       "split_operators" => false,
                                                       "debug_io" => true),
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
                        )

function runtests()
    run_moment_kinetics(test_input)
    return nothing
end

end # DebugIOTest


using .DebugIOTest

DebugIOTest.runtests()
