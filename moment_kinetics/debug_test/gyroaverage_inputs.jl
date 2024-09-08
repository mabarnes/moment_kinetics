using moment_kinetics.type_definitions: OptionsDict
test_type = "gyroaverage"

# default inputs for tests
test_input = OptionsDict(
    "run_name" => "gyroaverage",
    "composition" => OptionsDict("n_ion_species" => 1,
                          "n_neutral_species" => 0,
                          "gyrokinetic_ions" => true,
                          "T_e" => 1.0,
                          "T_wall" => 1.0),
    "evolve_moments" => OptionsDict("density" => false,
                                    "parallel_flow" => false,
                                    "parallel_pressure" => false,
                                    "moments_conservation" => false),
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
    "ion_source" => OptionsDict("z_profile" => "constant",
                                "source_strength" => 0.05,
                                "source_T" => 0.25),
    "timestepping" => OptionsDict("nstep" => 3,
                                       "dt" => 1.0e-12,
                                       "nwrite" => 2,
                                       "nwrite_dfns" => 2,),
    "r" => OptionsDict("ngrid" => 5,
                       "nelement" => 2,
                       "bc" => "periodic"),
    "z" => OptionsDict("ngrid" => 5,
                       "nelement" => 2,
                       "bc" => "periodic",
                       "discretization" => "chebyshev_pseudospectral"),
    "vpa" => OptionsDict("ngrid" => 5,
                         "nelement" => 2,
                         "L" => 6.0,
                         "bc" => "zero",
                         "discretization" => "chebyshev_pseudospectral"),
    "vperp" => OptionsDict("ngrid" => 5,
                           "nelement" => 1,
                           "L" => 3.0,
                           "bc" => "zero",
                           "discretization" => "chebyshev_pseudospectral"),
    "vz" => OptionsDict("ngrid" => 5,
                        "nelement" => 2,
                        "L" => 6.0,
                        "bc" => "zero",
                        "discretization" => "chebyshev_pseudospectral"),
    "numerical_dissipation" => OptionsDict("vpa_dissipation_coefficient" => 1.0e-3,
                                    "vperp_dissipation_coefficient" => 1.0e-3),
    "geometry" => OptionsDict("DeltaB"=>0.0,
                       "option"=>"constant-helical",
                       "pitch"=>0.1,
                       "rhostar"=> 0.1),
)

test_input_list = [
     test_input,
    ]
