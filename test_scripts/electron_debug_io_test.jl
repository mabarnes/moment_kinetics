module DebugElectronIOTest

using moment_kinetics
using moment_kinetics.type_definitions: OptionsDict

fixed_timestep_input = OptionsDict(
    "output" => OptionsDict("run_name" => "electron_debug_io_test_fixed_timestep"),
    "evolve_moments" => OptionsDict("density" => true,
                                    "moments_conservation" => true,
                                    "parallel_flow" => true,
                                    "pressure" => true),
    "r" => OptionsDict("ngrid" => 1,
                       "nelement" => 1),
    "z" => OptionsDict("ngrid" => 5,
                       "discretization" => "gausslegendre_pseudospectral",
                       "nelement" => 2,
                       "bc" => "wall"),
    "vpa" => OptionsDict("ngrid" => 6,
                         "discretization" => "gausslegendre_pseudospectral",
                         "nelement" => 9,
                         "L" => 40.0,
                         "bc" => "zero",
                         "element_spacing_option" => "coarse_tails8"),
    "composition" => OptionsDict("T_e" => 0.2,
                                 "electron_physics" => "kinetic_electrons",
                                 "n_ion_species" => 1,
                                 "n_neutral_species" => 0),
    "ion_species_1" => OptionsDict("initial_temperature" => 0.1,
                                   "initial_density" => 1.0),
    "z_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                        "density_amplitude" => 1.0,
                                        "temperature_amplitude" => 0.0,
                                        "density_phase" => 0.0,
                                        "upar_amplitude" => 1.0,
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
                               "ionization_frequency" => 1.0,
                               "charge_exchange_frequency" => 1.0),
    "ion_source_1" => OptionsDict("active" => true,
                                  "z_profile" => "gaussian",
                                  "z_width" => 0.25,
                                  "source_strength" => 3.0,
                                  "source_T" => 2.0),
    "ion_source_2" => OptionsDict("active" => true,
                                  "z_profile" => "wall_exp_decay",
                                  "z_width" => 0.25,
                                  "source_strength" => 1.0,
                                  "source_T" => 0.2),
    "timestepping" => OptionsDict("type" => "PareschiRusso2(2,2,2)",
                                  "kinetic_electron_solver" => "implicit_p_implicit_pseudotimestep",
                                  "kinetic_ion_solver" => "full_explicit_ion_advance",
                                  "nstep" => 1,
                                  "dt" => 1e-10,
                                  "nwrite" => 1,
                                  "nwrite_dfns" => 1,
                                 ),
    "electron_timestepping" => OptionsDict("nstep" => 5000000,
                                           "dt" => 1e-10,
                                           "maximum_dt" => Inf,
                                           "nwrite" => 10000,
                                           "nwrite_dfns" => 100000,
                                           "decrease_dt_iteration_threshold" => 5000,
                                           "increase_dt_iteration_threshold" => 0,
                                           "cap_factor_ion_dt" => 10.0,
                                           "initialization_residual_value" => 1.0e6,
                                           "converged_residual_value" => 1.0e6,
                                           "debug_io" => 1,
                                          ),
    "nonlinear_solver" => OptionsDict("nonlinear_max_iterations" => 10,
                                      "rtol" => 1.0e3,
                                      "atol" => 1.0e3,
                                      "linear_restart" => 5,
                                      "preconditioner_update_interval" => 100,
                                     ),
    "ion_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0),
    "electron_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0),
   )

adaptive_timestep_input = deepcopy(fixed_timestep_input)
adaptive_timestep_input["output"]["run_name"] = "electron_debug_io_test_adaptive_timestep"
adaptive_timestep_input["timestepping"]["type"] = "KennedyCarpenterARK324"
adaptive_timestep_input["timestepping"]["maximum_dt"] = 1.0e-5

# The following settings don't run for long enough to give a very good test, but at least
# make sure that the explicit electron solver can take a single timestep.
implicit_ppar_explicit_pseudotimestep_input = deepcopy(fixed_timestep_input)
implicit_ppar_explicit_pseudotimestep_input["output"]["run_name"] = "electron_debug_io_test_implicit_ppar_explicit_pseudotimestep"
implicit_ppar_explicit_pseudotimestep_input["timestepping"]["nstep"] = 1
implicit_ppar_explicit_pseudotimestep_input["timestepping"]["kinetic_electron_solver"] = "implicit_p_explicit_pseudotimestep"
implicit_ppar_explicit_pseudotimestep_input["electron_timestepping"]["nstep"] = 5000000
implicit_ppar_explicit_pseudotimestep_input["electron_timestepping"]["dt"] = 1.0e-7
implicit_ppar_explicit_pseudotimestep_input["electron_timestepping"]["maximum_dt"] = 1.0e-5
implicit_ppar_explicit_pseudotimestep_input["electron_timestepping"]["nwrite"] = 10000
implicit_ppar_explicit_pseudotimestep_input["electron_timestepping"]["nwrite_dfns"] = 100000
implicit_ppar_explicit_pseudotimestep_input["electron_timestepping"]["type"] = "Fekete4(3)"
implicit_ppar_explicit_pseudotimestep_input["electron_timestepping"]["rtol"] = 1.0e6
implicit_ppar_explicit_pseudotimestep_input["electron_timestepping"]["atol"] = 1.0e6
implicit_ppar_explicit_pseudotimestep_input["electron_timestepping"]["minimum_dt"] = 1.0e-9

test_inputs = [fixed_timestep_input, adaptive_timestep_input,
               implicit_ppar_explicit_pseudotimestep_input]

function runtests()
    for input ∈ test_inputs
        run_moment_kinetics(input)
    end
    return nothing
end

end # DebugElectronIOTest


using .DebugElectronIOTest

DebugElectronIOTest.runtests()
