using moment_kinetics
using moment_kinetics.type_definitions: OptionsDict
using makie_post_processing

test_input = OptionsDict(
    "output" => OptionsDict("run_name" => "makie_post_processing_check_data"),
    "evolve_moments" => OptionsDict("density" => true,
                                    "moments_conservation" => true,
                                    "parallel_flow" => true,
                                    "pressure" => true),
    "r" => OptionsDict("ngrid" => 1,
                       "nelement" => 1),
    "z" => OptionsDict("ngrid" => 3,
                       "discretization" => "gausslegendre_pseudospectral",
                       "nelement" => 2,
                       "bc" => "wall"),
    "vpa" => OptionsDict("ngrid" => 6,
                         "discretization" => "gausslegendre_pseudospectral",
                         "nelement" => 9,
                         "L" => 41.569219381653056,
                         "bc" => "zero",
                         "element_spacing_option" => "coarse_tails8.660254037844386"),
    "vz" => OptionsDict("ngrid" => 6,
                         "discretization" => "gausslegendre_pseudospectral",
                         "nelement" => 9,
                         "L" => 41.569219381653056,
                         "bc" => "zero",
                         "element_spacing_option" => "coarse_tails8.660254037844386"),
    "composition" => OptionsDict("T_e" => 0.2,
                                 "n_ion_species" => 1,
                                 "n_neutral_species" => 1,
                                 "electron_physics" => "kinetic_electrons"),
    "ion_species_1" => OptionsDict("initial_temperature" => 0.06666666666666667,
                                   "initial_density" => 1.0),
    "z_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                        "density_amplitude" => 1.0,
                                        "temperature_amplitude" => 0.0,
                                        "density_phase" => 0.0,
                                        "upar_amplitude" => 1.4142135623730951,
                                        "temperature_phase" => 0.0,
                                        "upar_phase" => 0.0),
    "vpa_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                          "density_amplitude" => 1.0,
                                          "temperature_amplitude" => 0.0,
                                          "density_phase" => 0.0,
                                          "upar_amplitude" => 0.0,
                                          "temperature_phase" => 0.0,
                                          "upar_phase" => 0.0),
    "krook_collisions" => OptionsDict("use_krook" => true),
    "reactions" => OptionsDict("electron_ionization_frequency" => 0.0,
                               "ionization_frequency" => 0.7071067811865476,
                               "charge_exchange_frequency" => 1.0606601717798214),
    "ion_source_1" => OptionsDict("active" => true,
                                  "z_profile" => "gaussian",
                                  "z_width" => 0.25,
                                  "source_strength" => 2.8284271247461903,
                                  "source_T" => 2.0),
    "ion_source_2" => OptionsDict("active" => true,
                                  "z_profile" => "wall_exp_decay",
                                  "z_width" => 0.25,
                                  "source_strength" => 0.7071067811865476,
                                  "source_T" => 0.2),
    "timestepping" => OptionsDict("type" => "KennedyCarpenterARK324",
                                  "kinetic_electron_solver" => "implicit_p_implicit_pseudotimestep",
                                  "kinetic_ion_solver" => "full_explicit_ion_advance",
                                  "nstep" => 1,
                                  "dt" => 7.0710678118654756e-6,
                                  "maximum_dt" => 7.0710678118654756e-6,
                                  "nwrite" => 1,
                                  "nwrite_dfns" => 1),
    "electron_timestepping" => OptionsDict("nstep" => 5000000,
                                           "dt" => 1.4142135623730951e-5,
                                           "maximum_dt" => Inf,
                                           "nwrite" => 10000,
                                           "nwrite_dfns" => 100000,
                                           "cap_factor_ion_dt" => 10.0,
                                           "initialization_residual_value" => 1.0e3,
                                           "converged_residual_value" => 1.0e3),
     "nonlinear_solver" => OptionsDict("nonlinear_max_iterations" => 1000,
                                       "rtol" => 1.0e-8,
                                       "atol" => 1.0e-14,
                                       "linear_restart" => 5,
                                       "preconditioner_update_interval" => 100),
    "ion_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0),
    "electron_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0),
   )

println("\nCheck makie_post_process\n")

postproc_input_filename = "check-makie_post_processing-input.toml"
if !isfile(postproc_input_filename)
    makie_post_processing.generate_maximal_input_file(postproc_input_filename)
end

run_moment_kinetics(test_input)

example_output_directory = joinpath("runs", test_input["output"]["run_name"])

makie_post_process(example_output_directory; input_file=postproc_input_filename)

# Also check comparing multiple outputs.
makie_post_process(example_output_directory, example_output_directory; input_file=postproc_input_filename)
makie_post_process(example_output_directory, example_output_directory, example_output_directory; input_file=postproc_input_filename)
