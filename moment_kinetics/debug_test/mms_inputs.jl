using moment_kinetics.type_definitions: OptionsDict
test_type = "MMS"

# default inputs for tests
test_input = OptionsDict(
    "manufactured_solns" => OptionsDict("use_for_advance"=>true),
    "evolve_moments" => OptionsDict("density" => false,
                                    "parallel_flow" => false,
                                    "pressure" => false,
                                    "moments_conservation" => false),
    "composition" => OptionsDict("n_ion_species" => 1,
                          "n_neutral_species" => 1,
                          "electron_physics" => "boltzmann_electron_response",
                          "T_e" => 1.0,
                          "T_wall" => 1.0),    
    "output" => OptionsDict("run_name" => "MMS-2D-wall_cheb-with-neutrals"),
    "reactions" => OptionsDict("charge_exchange_frequency" => 0.0,
                               "ionization_frequency" => 0.0),
    "timestepping" => OptionsDict("nstep" => 3,
                                       "dt" => 1.e-8,
                                       "nwrite" => 2,
                                       "type" => "SSPRK2",
                                       "split_operators" => false),
    "z" => OptionsDict("ngrid" => 3,
                       "nelement" => 2,
                       "bc" => "wall",
                       "discretization" => "chebyshev_pseudospectral"),
    "r" => OptionsDict("ngrid" => 3,
                       "nelement" => 2,
                       "discretization" => "chebyshev_pseudospectral"),
    "inner_r_bc_1" => OptionsDict("bc" => "periodic"),
    "outer_r_bc_1" => OptionsDict("bc" => "periodic"),
    "vpa" => OptionsDict("ngrid" => 3,
                         "nelement" => 2,
                         "L" => 12.0,
                         "bc" => "periodic",
                         "discretization" => "chebyshev_pseudospectral"),
    "vperp" => OptionsDict("ngrid" => 3,
                           "nelement" => 2,
                           "L" => 6.0,
                           "discretization" => "chebyshev_pseudospectral"),
    "vz" => OptionsDict("ngrid" => 3,
                        "nelement" => 2,
                        "L" => 12.0,
                        "bc" => "none",
                        "discretization" => "chebyshev_pseudospectral"),
    "vr" => OptionsDict("ngrid" => 3,
                        "nelement" => 2,
                        "L" => 12.0,
                        "bc" => "none",
                        "discretization" => "chebyshev_pseudospectral"),
    "vzeta" => OptionsDict("ngrid" => 3,
                           "nelement" => 2,
                           "L" => 12.0,
                           "bc" => "none",
                           "discretization" => "chebyshev_pseudospectral"),
    "geometry" => OptionsDict("rhostar" => 1.0,),
)

test_input_Dirichlet_r = deepcopy(test_input)
test_input_Dirichlet_r["output"]["run_name"] = "MMS-2D-wall_cheb-with-neutrals-Dirichlet"
test_input_Dirichlet_r["inner_r_bc_1"]["bc"] => "Dirichlet"
test_input_Dirichlet_r["outer_r_bc_1"]["bc"] => "Dirichlet"

test_input_list = [
     test_input,
     test_input_Dirichlet_r,
    ]
