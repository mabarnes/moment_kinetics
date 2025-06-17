#!julia

# Script to convert 'old' input files to latest format, updating any option names, etc.
# that have changed.
#
# Relies on loading TOML and re-writing file, so will lose all comments and may re-arrange
# the file. You may want to keep your original input file and just copy over the new
# sections.
#
# The original input file is not deleted, but is renamed with a '.unmodified' suffix.

using moment_kinetics.type_definitions: OptionsDict, OrderedDict
using moment_kinetics.utils: recursive_merge
using TOML

# Overload printing for type-alias OptionsDict to make it easier to copy-paste printed
# output of update_input_dict().
Base.show(io::IO, ::Type{OptionsDict}) = print(io, "OptionsDict")

# Define the map of old name to new section and name.
# Note, a couple of options have some special handling that is defined within the
# update_input_file() function.
const top_level_update_map = OptionsDict(
    "n_ion_species" => ("composition", "n_ion_species"),
    "n_neutral_species" => ("composition", "n_neutral_species"),
    "electron_physics" => ("composition", "electron_physics"),
    "T_e" => ("composition", "T_e"),
    "T_wall" => ("composition", "T_wall"),
    "phi_wall" => ("composition", "phi_wall"),
    "use_test_neutral_wall_pdf" => ("composition", "use_test_neutral_wall_pdf"),
    "recycling_fraction" => ("composition", "recycling_fraction"),
    "gyrokinetic_ions" => ("composition", "gyrokinetic_ions"),

    "run_name" => ("output", "run_name"),
    "base_directory" => ("output", "base_directory"),

    "evolve_moments_density" => ("evolve_moments", "density"),
    "evolve_moments_parallel_flow" => ("evolve_moments", "parallel_flow"),
    "evolve_moments_parallel_pressure" => ("evolve_moments", "parallel_pressure"),
    "evolve_moments_conservation" => ("evolve_moments", "moments_conservation"),

    "initial_density1" => ("ion_species_1", "initial_density"),
    "initial_temperature1" => ("ion_species_1", "initial_temperature"),
    "z_IC_option1" => ("z_IC_ion_species_1", "initialization_option"),
    "z_IC_width1" => ("z_IC_ion_species_1", "width"),
    "z_IC_wavenumber1" => ("z_IC_ion_species_1", "wavenumber"),
    "z_IC_density_amplitude1" => ("z_IC_ion_species_1", "density_amplitude"),
    "z_IC_density_phase1" => ("z_IC_ion_species_1", "density_phase"),
    "z_IC_upar_amplitude1" => ("z_IC_ion_species_1", "upar_amplitude"),
    "z_IC_upar_phase1" => ("z_IC_ion_species_1", "upar_phase"),
    "z_IC_temperature_amplitude1" => ("z_IC_ion_species_1", "temperature_amplitude"),
    "z_IC_temperature_phase1" => ("z_IC_ion_species_1", "temperature_phase"),
    "r_IC_option1" => ("r_IC_ion_species_1", "initialization_option"),
    "r_IC_width1" => ("r_IC_ion_species_1", "width"),
    "r_IC_wavenumber1" => ("r_IC_ion_species_1", "wavenumber"),
    "r_IC_density_amplitude1" => ("r_IC_ion_species_1", "density_amplitude"),
    "r_IC_density_phase1" => ("r_IC_ion_species_1", "density_phase"),
    "r_IC_upar_amplitude1" => ("r_IC_ion_species_1", "upar_amplitude"),
    "r_IC_upar_phase1" => ("r_IC_ion_species_1", "upar_phase"),
    "r_IC_temperature_amplitude1" => ("r_IC_ion_species_1", "temperature_amplitude"),
    "r_IC_temperature_phase1" => ("r_IC_ion_species_1", "temperature_phase"),
    "vpa_IC_option1" => ("vpa_IC_ion_species_1", "initialization_option"),
    "vpa_IC_width1" => ("vpa_IC_ion_species_1", "width"),
    "vpa_IC_wavenumber1" => ("vpa_IC_ion_species_1", "wavenumber"),
    "vpa_IC_density_amplitude1" => ("vpa_IC_ion_species_1", "density_amplitude"),
    "vpa_IC_density_phase1" => ("vpa_IC_ion_species_1", "density_phase"),
    "vpa_IC_upar_amplitude1" => ("vpa_IC_ion_species_1", "upar_amplitude"),
    "vpa_IC_upar_phase1" => ("vpa_IC_ion_species_1", "upar_phase"),
    "vpa_IC_temperature_amplitude1" => ("vpa_IC_ion_species_1", "temperature_amplitude"),
    "vpa_IC_temperature_phase1" => ("vpa_IC_ion_species_1", "temperature_phase"),
    "vpa_IC_v01" => ("vpa_IC_ion_species_1", "v0"),
    "vpa_IC_vth01" => ("vpa_IC_ion_species_1", "vth0"),
    "vpa_IC_vpa01" => ("vpa_IC_ion_species_1", "vpa0"),
    "vpa_IC_vperp01" => ("vpa_IC_ion_species_1", "vperp0"),

    "initial_density2" => ("neutral_species_1", "initial_density"),
    "initial_temperature2" => ("neutral_species_1", "initial_temperature"),
    "z_IC_option2" => ("z_IC_neutral_species_1", "initialization_option"),
    "z_IC_width2" => ("z_IC_neutral_species_1", "width"),
    "z_IC_wavenumber2" => ("z_IC_neutral_species_1", "wavenumber"),
    "z_IC_density_amplitude2" => ("z_IC_neutral_species_1", "density_amplitude"),
    "z_IC_density_phase2" => ("z_IC_neutral_species_1", "density_phase"),
    "z_IC_upar_amplitude2" => ("z_IC_neutral_species_1", "upar_amplitude"),
    "z_IC_upar_phase2" => ("z_IC_neutral_species_1", "upar_phase"),
    "z_IC_temperature_amplitude2" => ("z_IC_neutral_species_1", "temperature_amplitude"),
    "z_IC_temperature_phase2" => ("z_IC_neutral_species_1", "temperature_phase"),
    "r_IC_option2" => ("r_IC_neutral_species_1", "initialization_option"),
    "r_IC_width2" => ("r_IC_neutral_species_1", "width"),
    "r_IC_wavenumber2" => ("r_IC_neutral_species_1", "wavenumber"),
    "r_IC_density_amplitude2" => ("r_IC_neutral_species_1", "density_amplitude"),
    "r_IC_density_phase2" => ("r_IC_neutral_species_1", "density_phase"),
    "r_IC_upar_amplitude2" => ("r_IC_neutral_species_1", "upar_amplitude"),
    "r_IC_upar_phase2" => ("r_IC_neutral_species_1", "upar_phase"),
    "r_IC_temperature_amplitude2" => ("r_IC_neutral_species_1", "temperature_amplitude"),
    "r_IC_temperature_phase2" => ("r_IC_neutral_species_1", "temperature_phase"),
    "vpa_IC_option2" => ("vz_IC_neutral_species_1", "initialization_option"),
    "vpa_IC_width2" => ("vz_IC_neutral_species_1", "width"),
    "vpa_IC_wavenumber2" => ("vz_IC_neutral_species_1", "wavenumber"),
    "vpa_IC_density_amplitude2" => ("vz_IC_neutral_species_1", "density_amplitude"),
    "vpa_IC_density_phase2" => ("vz_IC_neutral_species_1", "density_phase"),
    "vpa_IC_upar_amplitude2" => ("vz_IC_neutral_species_1", "upar_amplitude"),
    "vpa_IC_upar_phase2" => ("vz_IC_neutral_species_1", "upar_phase"),
    "vpa_IC_temperature_amplitude2" => ("vz_IC_neutral_species_1", "temperature_amplitude"),
    "vpa_IC_temperature_phase2" => ("vz_IC_neutral_species_1", "temperature_phase"),
    "vpa_IC_v02" => ("vz_IC_neutral_species_1", "v0"),
    "vpa_IC_vth02" => ("vz_IC_neutral_species_1", "vth0"),
    "vpa_IC_vpa02" => ("vz_IC_neutral_species_1", "vpa0"),
    "vpa_IC_vperp02" => ("vz_IC_neutral_species_1", "vperp0"),

    "charge_exchange_frequency" => ("reactions", "charge_exchange_frequency"),
    "electron_charge_exchange_frequency" => ("reactions", "electron_charge_exchange_frequency"),
    "ionization_frequency" => ("reactions", "ionization_frequency"),
    "electron_ionization_frequency" => ("reactions", "electron_ionization_frequency"),
    "ionization_energy" => ("reactions", "ionization_energy"),

    "nu_ei" => ("electron_fluid_colisions", "nu_ei"),

    "r_ngrid" => ("r", "ngrid"),
    "r_nelement" => ("r", "nelement"),
    "r_nelement_local" => ("r", "nelement_local"),
    "r_L" => ("r", "L"),
    "r_discretization" => ("r", "discretization"),
    "r_finite_difference_option" => ("r", "finite_difference_option"),
    "r_bc" => ("r", "bc"),
    "r_element_spacing_option" => ("r", "element_spacing_option"),

    "z_ngrid" => ("z", "ngrid"),
    "z_nelement" => ("z", "nelement"),
    "z_nelement_local" => ("z", "nelement_local"),
    "z_L" => ("z", "L"),
    "z_discretization" => ("z", "discretization"),
    "z_finite_difference_option" => ("z", "finite_difference_option"),
    "z_bc" => ("z", "bc"),
    "z_element_spacing_option" => ("z", "element_spacing_option"),

    "vpa_ngrid" => ("vpa", "ngrid"),
    "vpa_nelement" => ("vpa", "nelement"),
    "vpa_nelement_local" => ("vpa", "nelement_local"),
    "vpa_L" => ("vpa", "L"),
    "vpa_discretization" => ("vpa", "discretization"),
    "vpa_finite_difference_option" => ("z", "finite_difference_option"),
    "vpa_bc" => ("vpa", "bc"),
    "vpa_element_spacing_option" => ("vpa", "element_spacing_option"),

    "vperp_ngrid" => ("vperp", "ngrid"),
    "vperp_nelement" => ("vperp", "nelement"),
    "vperp_nelement_local" => ("vperp", "nelement_local"),
    "vperp_L" => ("vperp", "L"),
    "vperp_discretization" => ("vperp", "discretization"),
    "vperp_finite_difference_option" => ("vperp", "finite_difference_option"),
    "vperp_bc" => ("vperp", "bc"),
    "vperp_element_spacing_option" => ("vperp", "element_spacing_option"),

    "gyrophase_ngrid" => ("gyrophase", "ngrid"),
    "gyrophase_nelement" => ("gyrophase", "nelement"),
    "gyrophase_nelement_local" => ("gyrophase", "nelement_local"),
    "gyrophase_L" => ("gyrophase", "L"),
    "gyrophase_discretization" => ("gyrophase", "discretization"),
    "gyrophase_finite_difference_option" => ("gyrophase", "finite_difference_option"),
    "gyrophase_bc" => ("gyrophase", "bc"),
    "gyrophase_element_spacing_option" => ("gyrophase", "element_spacing_option"),

    "vzeta_ngrid" => ("vzeta", "ngrid"),
    "vzeta_nelement" => ("vzeta", "nelement"),
    "vzeta_nelement_local" => ("vzeta", "nelement_local"),
    "vzeta_L" => ("vzeta", "L"),
    "vzeta_discretization" => ("vzeta", "discretization"),
    "vzeta_finite_difference_option" => ("vzeta", "finite_difference_option"),
    "vzeta_bc" => ("vzeta", "bc"),
    "vzeta_element_spacing_option" => ("vzeta", "element_spacing_option"),

    "vz_ngrid" => ("vz", "ngrid"),
    "vz_nelement" => ("vz", "nelement"),
    "vz_nelement_local" => ("vz", "nelement_local"),
    "vz_L" => ("vz", "L"),
    "vz_discretization" => ("vz", "discretization"),
    "vz_finite_difference_option" => ("vz", "finite_difference_option"),
    "vz_bc" => ("vz", "bc"),
    "vz_element_spacing_option" => ("vz", "element_spacing_option"),

    "vr_ngrid" => ("vr", "ngrid"),
    "vr_nelement" => ("vr", "nelement"),
    "vr_nelement_local" => ("vr", "nelement_local"),
    "vr_L" => ("vr", "L"),
    "vr_discretization" => ("vr", "discretization"),
    "vr_finite_difference_option" => ("vr", "finite_difference_option"),
    "vr_bc" => ("vr", "bc"),
    "vr_element_spacing_option" => ("vr", "element_spacing_option"),

    "force_Er_zero_at_wall" => ("em_fields", "force_Er_zero_at_wall"),

    "dt" => ("timestepping", "dt"),
    "nstep" => ("timestepping", "nstep"),
    "nwrite" => ("timestepping", "nwrite"),
    "nwrite_dfns" => ("timestepping", "nwrite_dfns"),
    "use_semi_lagrange" => nothing,
    "n_rk_stages" => nothing,
    "split_operators" => nothing,

    "rhostar" => ("geometry", "rhostar"),
    "Bzed" => ("geometry", "Bzed"),
    "Bmag" => ("geometry", "Bmag"),
    "Er_constant" => ("geometry", "Er_constant"),

    "use_manufactured_solns_for_init" => ("manufactured_solns", "use_for_init"),
    "use_manufactured_solns_for_advance" => ("manufactured_solns", "use_for_advance"),
    "epsilon_offset" => ("manufactured_solns", "epsilon_offset"),
    "use_vpabar_in_mms_dfni" => ("manufactured_solns", "use_vpabar_in_mms_dfni"),
    "alpha_switch" => ("manufactured_solns", "alpha_switch"),
   )

# If the "new option" is a String, it is the name of the option within the same section
# that should replace the "old option".
# If the "new option" is an OptionsDict, it gives a map from the original option values to
# new option names and values. If the new value is `nothing` it is replaced by the old
# value.
# If the "new option" is a Function, the option is not renamed, instead the function is
# applied to the value.
const sections_update_map = OptionsDict(
    "evolve_moments" => OptionsDict("parallel_pressure" => "pressure"),
    "timestepping" => OptionsDict("implicit_electron_advance" => OrderedDict{Any,Any}(true => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_steady_state"),),
                                                                                      "lu" => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_steady_state", "kinetic_electron_preconditioner" => "lu"),),
                                                                                      "adi" => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_steady_state", "kinetic_electron_preconditioner" => "adi"),),
                                                                                      "static_condensation" => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_steady_state", "kinetic_electron_preconditioner" => "static_condensation"),),
                                                                                     ),
                                  "implicit_electron_time_evolving" => OrderedDict{Any,Any}(true => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_time_evolving"),),
                                                                                            "lu" => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_time_evolving", "kinetic_electron_preconditioner" => "lu"),),
                                                                                            "adi" => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_time_evolving", "kinetic_electron_preconditioner" => "adi"),),
                                                                                            "static_condensation" => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_time_evolving", "kinetic_electron_preconditioner" => "static_condensation"),),
                                                                                           ),
                                  "implicit_electron_ppar" => OrderedDict{Any,Any}(true => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_p_implicit_pseudotimestep"),),
                                                                                   "lu" => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_p_implicit_pseudotimestep", "kinetic_electron_preconditioner" => "lu"),),
                                                                                   "adi" => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_p_implicit_pseudotimestep", "kinetic_electron_preconditioner" => "adi"),),
                                                                                   "static_condensation" => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_p_implicit_pseudotimestep", "kinetic_electron_preconditioner" => "static_condensation"),),
                                                                                  ),
                                  "kinetic_electron_solver" => OrderedDict{Any,Any}("implicit_ppar_implicit_pseudotimestep" => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_p_implicit_pseudotimestep")),
                                                                                    "implicit_ppar_explicit_pseudotimestep" => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_p_explicit_pseudotimestep")),
                                                                                   ),
                                  "implicit_ion_advance" => Dict(true => Dict("timestepping" => Dict("kinetic_ion_solver" => "full_implicit_ion_advance"))),
                                  "implicit_vpa_advection" => Dict(true => Dict("timestepping" => Dict("kinetic_ion_solver" => "implicit_ion_vpa_advection"))),                                         
                                 ),
   )

PR322_p = (x) -> 2 .* x
PR322_T_1V = (x) -> x ./ 3 # Inputs before PR322 were T_∥ values, but after are T values (for 1D1V T=T_∥/3)
PR322_temperature_PI_1V = (x) -> 3 .* x # Inputs before PR322 were T_∥ values, but after are T values (for 1D1V T=T_∥/3)
PR322_v = (x) -> sqrt(2) .* x
PR322_t = (x) -> x ./ sqrt(2)
PR322_omega = (x) -> any(x .≥ 0.0) ? sqrt(2) .* x : x
PR322_nuii = (x) -> any(x .≥ 0.0) ? 4 .* x : x
PR322_I = (x) -> any(x .≥ 0.0) ? 2 .* x : x
PR322_v_diffusion_coefficient = (x) -> 2^1.5 .* x
PR322_w_diffusion_coefficient = (x) -> 3 * sqrt(2) .* x
PR322_w_evolve_ppar = (x) -> sqrt(3) .* x
const PR322_definitions_update_map_2V = OptionsDict(
    # "upar_amplitude" might not always need correcting - e.g. the amplitude is a relative
    # amplitude that should be left unchanged if using initialization_option="sinusoid" -
    # but usually either upar_amplitude=0 or initialization_option="gaussian" (in which
    # case this option represents an absolute value of parallel flow)
    "z_IC_ion_species_1" => OptionsDict("upar_amplitude" => PR322_v),
    "z_IC_neutral_species_1" => OptionsDict("upar_amplitude" => PR322_v),
    "vpa_IC_ion_species_1" => OptionsDict("v0" => PR322_v,
                                          "vth0" => PR322_v,
                                          "vpa0" => PR322_v,
                                          "vperp0" => PR322_v,
                                         ),
    "vz_IC_neutral_species_1" => OptionsDict("v0" => PR322_v,
                                             "vth0" => PR322_v,
                                             "vpa0" => PR322_v,
                                             "vperp0" => PR322_v,
                                            ),
    "reactions" => OptionsDict("charge_exchange_frequency" => PR322_omega,
                               "electron_charge_exchange_frequency" => PR322_omega,
                               "ionization_frequency" => PR322_omega,
                               "electron_ionization_frequency" => PR322_omega,
                               "ionization_energy" => PR322_p,
                              ),
    "electron_fluid_collisions" => OptionsDict("nu_ei" => PR322_omega,),
    "krook_collisions" => OptionsDict("nuii0" => PR322_omega,
                                      "nuee0" => PR322_omega,
                                      "nuei0" => PR322_omega,
                                     ),
    "fokker_planck_collisions" => OptionsDict("nuii" => PR322_nuii,
                                             ),
    "maxwell_diffusion_collisions" => OptionsDict("D_ii" => PR322_v_diffusion_coefficient,
                                                  "D_nn" => PR322_v_diffusion_coefficient,
                                                 ),
    "ion_numerical_dissipation" => OptionsDict("vpa_boundary_buffer_damping_rate" => PR322_omega,
                                               "vpa_boundary_buffer_diffusion_coefficient" => PR322_v_diffusion_coefficient,
                                               "vpa_dissipation_coefficient" => PR322_v_diffusion_coefficient,
                                               "vperp_dissipation_coefficient" => PR322_v_diffusion_coefficient,
                                               "z_dissipation_coefficient" => PR322_omega,
                                               "r_dissipation_coefficient" => PR322_omega,
                                               "moment_dissipation_coefficient" => PR322_omega,
                                              ),
    "electron_numerical_dissipation" => OptionsDict("vpa_boundary_buffer_damping_rate" => PR322_omega,
                                                    "vpa_boundary_buffer_diffusion_coefficient" => PR322_v_diffusion_coefficient,
                                                    "vpa_dissipation_coefficient" => PR322_v_diffusion_coefficient,
                                                    "vperp_dissipation_coefficient" => PR322_v_diffusion_coefficient,
                                                    "z_dissipation_coefficient" => PR322_omega,
                                                    "r_dissipation_coefficient" => PR322_omega,
                                                    "moment_dissipation_coefficient" => PR322_omega,
                                                   ),
    "neutral_numerical_dissipation" => OptionsDict("vz_dissipation_coefficient" => PR322_v_diffusion_coefficient,
                                                   "z_dissipation_coefficient" => PR322_omega,
                                                   "r_dissipation_coefficient" => PR322_omega,
                                                   "moment_dissipation_coefficient" => PR322_omega,
                                                  ),
    "timestepping" => OptionsDict("dt" => PR322_t,
                                  "minimum_dt" => PR322_t,
                                  "maximum_dt" => PR322_t,
                                  "constraint_forcing_rate" => PR322_omega,
                                  "converged_residual_value" => PR322_omega,
                                 ),
    "electron_timestepping" => OptionsDict("dt" => PR322_t,
                                           "minimum_dt" => PR322_t,
                                           "maximum_dt" => PR322_t,
                                           "constraint_forcing_rate" => PR322_omega,
                                           "converged_residual_value" => PR322_omega,
                                          ),
    "vpa" => OptionsDict("element_spacing_option" => OptionsDict("coarse_tails" => OptionsDict("vpa" => OptionsDict("element_spacing_option" => "coarse_tails$(5.0*sqrt(2))"))),
                         "L" => PR322_v,),
    "vperp" => OptionsDict("element_spacing_option" => OptionsDict("coarse_tails" => OptionsDict("vperp" => OptionsDict("element_spacing_option" => "coarse_tails$(5.0*sqrt(2))"))),
                           "L" => PR322_v,),
    "vz" => OptionsDict("element_spacing_option" => OptionsDict("coarse_tails" => OptionsDict("vz" => OptionsDict("element_spacing_option" => "coarse_tails$(5.0*sqrt(2))"))),
                        "L" => PR322_v,),
    "vr" => OptionsDict("element_spacing_option" => OptionsDict("coarse_tails" => OptionsDict("vr" => OptionsDict("element_spacing_option" => "coarse_tails$(5.0*sqrt(2))"))),
                        "L" => PR322_v,),
    "vzeta" => OptionsDict("element_spacing_option" => OptionsDict("coarse_tails" => OptionsDict("vzeta" => OptionsDict("element_spacing_option" => "coarse_tails$(5.0*sqrt(2))"))),
                           "L" => PR322_v,),
    "ion_source_1" => OptionsDict("source_strength" => PR322_omega,
                                  "source_v0" => PR322_v,
                                  "source_vpa0" => PR322_v,
                                  "source_vperp0" => PR322_v,
                                  "sink_vth" => PR322_v,
                                  "PI_density_controller_P" => PR322_omega,
                                  "PI_density_controller_I" => PR322_I,
                                  "PI_temperature_controller_P" => PR322_omega,
                                  "PI_temperature_controller_I" => PR322_I,
                                 ),
    "ion_source_2" => OptionsDict("source_strength" => PR322_omega,
                                  "source_v0" => PR322_v,
                                  "source_vpa0" => PR322_v,
                                  "source_vperp0" => PR322_v,
                                  "sink_vth" => PR322_v,
                                  "PI_density_controller_P" => PR322_omega,
                                  "PI_density_controller_I" => PR322_I,
                                  "PI_temperature_controller_P" => PR322_omega,
                                  "PI_temperature_controller_I" => PR322_I,
                                 ),
    "ion_source_3" => OptionsDict("source_strength" => PR322_omega,
                                  "source_v0" => PR322_v,
                                  "source_vpa0" => PR322_v,
                                  "source_vperp0" => PR322_v,
                                  "sink_vth" => PR322_v,
                                  "PI_density_controller_P" => PR322_omega,
                                  "PI_density_controller_I" => PR322_I,
                                  "PI_temperature_controller_P" => PR322_omega,
                                  "PI_temperature_controller_I" => PR322_I,
                                 ),
    "electron_source_1" => OptionsDict("source_strength" => PR322_omega,
                                      ),
    "neutral_source_1" => OptionsDict("source_strength" => PR322_omega,
                                      "source_v0" => PR322_v,
                                      "source_vpa0" => PR322_v,
                                      "source_vperp0" => PR322_v,
                                      "sink_vth" => PR322_v,
                                      "PI_density_controller_P" => PR322_omega,
                                      "PI_density_controller_I" => PR322_I,
                                      "PI_temperature_controller_P" => PR322_omega,
                                      "PI_temperature_controller_I" => PR322_I,
                                     ),
   )
const PR322_definitions_update_map_1V = recursive_merge(
    PR322_definitions_update_map_2V,
    OptionsDict("composition" => OptionsDict(# Note T_e and T_wall should not get updated.
                                            ),
                "ion_species_1" => OptionsDict("initial_temperature" => PR322_T_1V,),
                "neutral_species_1" => OptionsDict("initial_temperature" => PR322_T_1V,),
                "reactions" => OptionsDict("ionization_energy" => PR322_T_1V,),
                "fokker_planck_collisions" => OptionsDict("sd_temp" => PR322_T_1V,),
                "ion_source_1" => OptionsDict("PI_temperature_target_amplitude" => PR322_T_1V,)
               )
   )
const PR322_definitions_update_map_1V_evolve_ppar = recursive_merge(
    PR322_definitions_update_map_1V,
    OptionsDict("vpa" => OptionsDict("element_spacing_option" => OptionsDict("coarse_tails" => OptionsDict("vpa" => OptionsDict("element_spacing_option" => "coarse_tails$(5.0*sqrt(3))"))),
                                     "L" => PR322_w_evolve_ppar,
                                    ),
                "vz" => OptionsDict("element_spacing_option" => OptionsDict("coarse_tails" => OptionsDict("vz" => OptionsDict("element_spacing_option" => "coarse_tails$(5.0*sqrt(3))"))),
                                    "L" => PR322_w_evolve_ppar,
                                   ),
                "ion_numerical_dissipation" => OptionsDict("vpa_boundary_buffer_diffusion_coefficient" => PR322_w_diffusion_coefficient,
                                                           "vpa_dissipation_coefficient" => PR322_w_diffusion_coefficient,
                                                          ),
                "electron_numerical_dissipation" => OptionsDict("vpa_boundary_buffer_diffusion_coefficient" => PR322_w_diffusion_coefficient,
                                                                "vpa_dissipation_coefficient" => PR322_w_diffusion_coefficient,
                                                               ),
                "neutral_numerical_dissipation" => OptionsDict("vz_dissipation_coefficient" => PR322_w_diffusion_coefficient,),
               )
   )


function update_input_dict(original_input::DictType;
                           update_definitions_322=false) where DictType <: AbstractDict
    original_input = deepcopy(original_input)
    updated_input = DictType()

    # Get the existing sections first
    for k ∈ collect(keys(original_input))
        v = original_input[k]
        if isa(v, AbstractDict)
            updated_input[k] = v
            pop!(original_input, k)
        end
    end

    # Put top-level values into the correct sections.
    # Note at this point the original sections have been removed from `original_input`.
    for (k,v) ∈ original_input
        if k == "constant_ionization_rate"
            if v
                println("constant_ionization_rate is no longer supported.")
                println("It can be replaced using the ion source term (with the value that was `ionization_frequency` set as the `source_strength`), with a section like:")
                println("[ion_source_1]")
                println("z_profile = \"constant\"")
                println("source_strength = 1.0")
                println("souce_T = 0.25")
                error("constant_ionization_rate is no longer supported")
            else
                println("constant_ionization_rate is no longer supported. It was set to `false`, so just dropping it")
            end
        elseif k == "krook_collisions_option"
            section = get(updated_input, "krook_collisions", DictType())
            section["use_krook"] = true
            section["frequency_option"] = v
        elseif k == "nuii"
            section = get(updated_input, "fokker_planck_collisions", DictType())
            section["use_fokker_planck"] = true
            section[k] = v
        elseif k == "combine_outer"
            # Option for parameter scans, leave in top level
            if "" ∉ keys(updated_input)
                updated_input[""] = OptionsDict()
            end
            updated_input[""]["combine_outer"] = original_input["combine_outer"]
        else
            if top_level_update_map[k] === nothing
                # Just drop this option, assuming it was only ever set to a default value
            else
                new_section_name, new_key = top_level_update_map[k]
                updated_input[new_section_name] = get(updated_input, new_section_name, DictType())
                updated_input[new_section_name][new_key] = v
            end
        end
    end

    combined_update_map = sections_update_map
    if update_definitions_322
        is_2V = ("vperp" ∈ keys(updated_input)
                 && (("ngrid" ∈ keys(updated_input["vperp"]) && updated_input["vperp"]["ngrid"] > 1)
                     || ("nelement" ∈ keys(updated_input["vperp"]) && updated_input["vperp"]["nelement"] > 1)
                    )
                )
        evolve_moments_section = get(updated_input, "evolve_moments", OptionsDict())
        if "pressure" ∈ keys(evolve_moments_section)
            error("Updating for changes in PR #322, but \"pressure\" is present in "
                  * "[evolve_moments] section, which means input file must have been "
                  * "created after PR #322 was merged.")
        end
        evolve_p = get(evolve_moments_section, "parallel_pressure", false) || get(evolve_moments_section, "pressure", false)
        if is_2V
            combined_update_map = recursive_merge(combined_update_map, PR322_definitions_update_map_2V)
        else
            if evolve_p
                combined_update_map = recursive_merge(combined_update_map, PR322_definitions_update_map_1V_evolve_ppar)
            else
                combined_update_map = recursive_merge(combined_update_map, PR322_definitions_update_map_1V)
            end
        end
    end

    # Fix updated options in the sections
    existing_sections = keys(updated_input)
    # Some special updates that it is inconvenient to do using combined_update_map
    for section_name ∈ existing_sections
        if update_definitions_322 && !is_2V &&
                (startswith(section_name, "ion_source")
                 || startswith(section_name, "electron_source")
                 || startswith(section_name, "neutral_source"))
            section = updated_input[section_name]
            if ("source_strength" ∈ keys(section)
                && "source_type" ∈ keys(section)
                && section["source_type"] ∈ ("energy", "temperature_midpoint_control")
               )
                # Update for 1V Tpar->T here. The tref correction will be applied below.
                if "PI_temperature_controller_P" ∈ keys(section)
                    section["PI_temperature_controller_P"] = PR322_temperature_PI_1V(section["PI_temperature_controller_P"])
                end
                if "PI_temperature_controller_I" ∈ keys(section)
                    section["PI_temperature_controller_I"] = PR322_temperature_PI_1V(section["PI_temperature_controller_I"])
                end
            end
        end
    end
    for (section_name, section_update_map) ∈ combined_update_map
        if section_name ∉ existing_sections
            continue
        end
        section = updated_input[section_name]
        existing_keys = keys(section)
        for (old_key, new_option_setting) ∈ section_update_map
            if old_key ∉ existing_keys
                continue
            end
            old_value = section[old_key]
            if new_option_setting isa AbstractDict
                if old_value ∉ keys(new_option_setting)
                    continue
                end
                pop!(section, old_key)
                for (new_section_name, new_section_map) ∈ new_option_setting[old_value]
                    new_section = get(updated_input, new_section_name, DictType())
                    for (k,v) ∈ new_section_map
                        if k ∈ keys(new_section)
                            error("Trying to add new key \"$k\" to $new_section_name, but \"$k\" already exists")
                        end
                        if v === nothing
                            new_section[k] = old_value
                        else
                            new_section[k] = v
                        end
                    end
                end
            elseif new_option_setting isa Function
                section[old_key] = new_option_setting(old_value)
            else
                pop!(section, old_key)
                section[new_option_setting] = old_value
            end
        end
    end

    return updated_input
end

function update_input_file(filename; update_definitions_322=false)
    file_text = read(filename, String)

    original_input = TOML.parse(file_text)

    updated_input = update_input_dict(original_input;
                                      update_definitions_322=update_definitions_322)

    updated_file_text = ""
    section = ""
    updated_sections = collect(keys(updated_input))
    top_level = true
    have_printed_missing_comments_header = false
    if "" ∈ updated_sections
        updated_section = pop!(updated_input, "")
        skip_this_section = false
    else
        updated_section = OptionsDict()
        skip_this_section = true
    end
    updated_section_keys = collect(keys(updated_section))
    function print_missing_comment_message(key_value, comment)
        if !have_printed_missing_comments_header
            println("The comments from the following lines have not been transferred to "
                    * "the updated input files. Please copy them manually if you want to "
                    * "keep them.")
            have_printed_missing_comments_header = true
        end
        println("$key_value#$comment")
        return nothing
    end

    # Update sections that are already present in original input file.
    for line ∈ split(file_text, "\n")
        comment_split = split(line, "#"; limit=2)
        if length(comment_split) == 1
            # No comment present
            key_value = comment_split[1]
            comment = ""
        else
            key_value, comment = comment_split
        end

        section_regex_match = match(r"^\w*\[.*\]", key_value)
        if section_regex_match !== nothing
            # This line is a section heading

            top_level = false

            # Add any more options that have not already been handled from
            # `updated_section`.
            if length(updated_section) > 0
                updated_file_text = rstrip(updated_file_text)
                updated_file_text *= "\n"

                for (key, value) ∈ updated_section
                    if value isa String
                        value = "\"$value\""
                    end
                    updated_file_text *= "$key = $value\n"
                end

                updated_file_text *= "\n"
            end

            section_title = section_regex_match.match[2:end-1]
            if section_title ∉ updated_sections
                skip_this_section = true
                if comment != ""
                    print_missing_comment_message(key_value, comment)
                end
            else
                updated_section = pop!(updated_input, section_title)
                updated_section_keys = collect(keys(updated_section))
                skip_this_section = false
                updated_file_text *= "$key_value"
                if comment != ""
                    updated_file_text *= "#$comment"
                end
                updated_file_text *= "\n"
            end
        elseif top_level && comment != ""
            updated_file_text *= "$key_value#$comment\n"
        elseif skip_this_section && comment != ""
            print_missing_comment_message(key_value, comment)
        elseif skip_this_section
            # Nothing to do
        elseif all(c->isspace(c), key_value)
            # No actual setting on this line, so just re-print
            updated_file_text *= "$key_value"
            if comment != ""
                updated_file_text *= "#$comment"
            end
            updated_file_text *= "\n"
        else
            key, _ = split(key_value, "=")
            key = strip(key)

            if key ∈ updated_section_keys
                value = pop!(updated_section, key)
                if value isa String
                    value = "\"$value\""
                end
                updated_file_text *= "$key = $value"
                if comment != ""
                    updated_file_text *= " #$comment"
                end
                updated_file_text *= "\n"
            else
                if comment != ""
                    print_missing_comment_message(key_value, comment)
                end
            end
        end
    end

    # Add any new sections that were not present in originial input file.
    for (updated_section_name, updated_section) ∈ updated_input
        updated_file_text *= "\n[$updated_section_name]\n"
        for (key, value) ∈ updated_section
            if value isa String
                value = "\"$value\""
            end
            updated_file_text *= "$key = $value\n"
        end
    end

    # Make sure there is just one newline at the end of the file
    updated_file_text = rstrip(updated_file_text, '\n')
    updated_file_text *= "\n"

    mv(filename, "$filename.unmodified")

    # Write the updated file. We have moved the original file, so this does not need to
    # overwrite. Check if the file already exists first to ensure we never accidentally
    # delete a file, even though this should never happen.
    if isfile(filename)
        error("$filename already exists")
    end
    open(filename; write=true) do io
        print(io, updated_file_text)
    end

    return nothing
end

using moment_kinetics.command_line_options.ArgParse
if abspath(PROGRAM_FILE) == @__FILE__
    s = ArgParseSettings()
    @add_arg_table! s begin
        "inputfiles"
            help = "Name of TOML input file to update."
            arg_type = String
            nargs = '+'
            default = nothing
        "--update-definitions-322"
            help = "Update definitions and dimensionless variables according to the changes in PR #322 (April 2025)"
            action = :store_true
    end
    args = parse_args(s)

    for filename ∈ args["inputfiles"]
        update_input_file(filename; update_definitions_322=args["update-definitions-322"])
    end
end
