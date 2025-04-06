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
                                  "implicit_electron_ppar" => OrderedDict{Any,Any}(true => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_ppar_implicit_pseudotimestep"),),
                                                                                   "lu" => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_ppar_implicit_pseudotimestep", "kinetic_electron_preconditioner" => "lu"),),
                                                                                   "adi" => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_ppar_implicit_pseudotimestep", "kinetic_electron_preconditioner" => "adi"),),
                                                                                   "static_condensation" => OptionsDict("timestepping" => OptionsDict("kinetic_electron_solver" => "implicit_ppar_implicit_pseudotimestep", "kinetic_electron_preconditioner" => "static_condensation"),),
                                                                                  ),
                                 ),
   )

PR322_T = (x) -> 2*x
PR322_T_1V = (x) -> 2*x/3 # Inputs before PR322 were T_∥ values, but after are T values (for 1D1V T=T_∥/3)
PR322_v = (x) -> sqrt(2)*x
PR322_t = (x) -> x/sqrt(2)
PR322_omega = (x) -> x ≥ 0.0 ? sqrt(2)*x : x
PR322_v_diffusion_coefficient = (x) -> 2^1.5*x
const PR322_definitions_update_map_2V = OptionsDict(
    "composition" => OptionsDict("T_e" => PR322_T,
                                 "T_wall" => PR322_T,
                                ),
    "ion_species_1" => OptionsDict("initial_temperature" => PR322_T,),
    "z_IC_ion_species_1" => OptionsDict("temperature_amplitude" => PR322_T,
                                        "upar_amplitude" => PR322_v,
                                       ),
    "r_IC_ion_species_1" => OptionsDict("temperature_amplitude" => PR322_T,
                                        "upar_amplitude" => PR322_v,
                                       ),
    "vpa_IC_ion_species_1" => OptionsDict("temperature_amplitude" => PR322_T,
                                          "upar_amplitude" => PR322_v,
                                          "v0" => PR322_v,
                                          "vth0" => PR322_v,
                                          "vpa0" => PR322_v,
                                          "vperp0" => PR322_v,
                                         ),
    "neutral_species_1" => OptionsDict("initial_temperature" => PR322_T,),
    "z_IC_neutral_species_1" => OptionsDict("temperature_amplitude" => PR322_T,
                                            "upar_amplitude" => PR322_v,
                                           ),
    "r_IC_neutral_species_1" => OptionsDict("temperature_amplitude" => PR322_T,
                                            "upar_amplitude" => PR322_v,
                                           ),
    "vz_IC_neutral_species_1" => OptionsDict("temperature_amplitude" => PR322_T,
                                             "upar_amplitude" => PR322_v,
                                             "v0" => PR322_v,
                                             "vth0" => PR322_v,
                                             "vpa0" => PR322_v,
                                             "vperp0" => PR322_v,
                                            ),
    "reactions" => OptionsDict("charge_exchange_frequency" => PR322_omega,
                               "electron_charge_exchange_frequency" => PR322_omega,
                               "ionization_frequency" => PR322_omega,
                               "electron_ionization_frequency" => PR322_omega,
                               "ionization_energy" => PR322_T,
                              ),
    "electron_fluid_collisions" => OptionsDict("nu_ei" => PR322_omega,),
    "krook_collisions" => OptionsDict("nuii0" => PR322_omega,
                                      "nuee0" => PR322_omega,
                                      "nuei0" => PR322_omega,
                                     ),
    "fokker_planck_collisions" => OptionsDict("nuii0" => PR322_omega,
                                              "sd_temp" => PR322_T,
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
    "vpa" => OptionsDict("L" => PR322_v,),
    "vperp" => OptionsDict("L" => PR322_v,),
    "vz" => OptionsDict("L" => PR322_v,),
    "vr" => OptionsDict("L" => PR322_v,),
    "vzeta" => OptionsDict("L" => PR322_v,),
    "ion_source_1" => OptionsDict("source_strength" => PR322_omega,
                                  "source_T" => PR322_T, # This should be PR322_T even for 1V case, not PR322_T_1V, because we still implement the source T_∥=source_T for 1V in the updated code.
                                  "source_v0" => PR322_v,
                                  "source_vpa0" => PR322_v,
                                  "source_vperp0" => PR322_v,
                                  "sink_vth" => PR322_v,
                                  "PI_density_controller_P" => PR322_omega,
                                  "PI_density_controller_I" => PR322_omega,
                                  "PI_temperature_controller_P" => PR322_omega,
                                  "PI_temperature_controller_I" => PR322_omega,
                                  "PI_temperature_target_amplitude" => PR322_T,
                                 ),
    "ion_source_2" => OptionsDict("source_strength" => PR322_omega,
                                  "source_T" => PR322_T, # This should be PR322_T even for 1V case, not PR322_T_1V, because we still implement the source T_∥=source_T for 1V in the updated code.
                                  "source_v0" => PR322_v,
                                  "source_vpa0" => PR322_v,
                                  "source_vperp0" => PR322_v,
                                  "sink_vth" => PR322_v,
                                  "PI_density_controller_P" => PR322_omega,
                                  "PI_density_controller_I" => PR322_omega,
                                  "PI_temperature_controller_P" => PR322_omega,
                                  "PI_temperature_controller_I" => PR322_omega,
                                  "PI_temperature_target_amplitude" => PR322_T,
                                 ),
    "ion_source_3" => OptionsDict("source_strength" => PR322_omega,
                                  "source_T" => PR322_T, # This should be PR322_T even for 1V case, not PR322_T_1V, because we still implement the source T_∥=source_T for 1V in the updated code.
                                  "source_v0" => PR322_v,
                                  "source_vpa0" => PR322_v,
                                  "source_vperp0" => PR322_v,
                                  "sink_vth" => PR322_v,
                                  "PI_density_controller_P" => PR322_omega,
                                  "PI_density_controller_I" => PR322_omega,
                                  "PI_temperature_controller_P" => PR322_omega,
                                  "PI_temperature_controller_I" => PR322_omega,
                                  "PI_temperature_target_amplitude" => PR322_T,
                                 ),
    "electron_source_1" => OptionsDict("source_strength" => PR322_omega,
                                       "source_T" => PR322_T, # This should be PR322_T even for 1V case, not PR322_T_1V, because we still implement the source T_∥=source_T for 1V in the updated code.
                                      ),
    "neutral_source_1" => OptionsDict("source_strength" => PR322_omega,
                                      "source_T" => PR322_T, # This should be PR322_T even for 1V case, not PR322_T_1V, because we still implement the source T_∥=source_T for 1V in the updated code.
                                      "source_v0" => PR322_v,
                                      "source_vpa0" => PR322_v,
                                      "source_vperp0" => PR322_v,
                                      "sink_vth" => PR322_v,
                                      "PI_density_controller_P" => PR322_omega,
                                      "PI_density_controller_I" => PR322_omega,
                                      "PI_temperature_controller_P" => PR322_omega,
                                      "PI_temperature_controller_I" => PR322_omega,
                                      "PI_temperature_target_amplitude" => PR322_T,
                                     ),
   )
const PR322_definitions_update_map_1V = recursive_merge(
    PR322_definitions_update_map_2V,
    OptionsDict("composition" => OptionsDict("T_e" => PR322_T_1V,
                                             "T_wall" => PR322_T_1V,
                                            ),
                "ion_species_1" => OptionsDict("initial_temperature" => PR322_T_1V,),
                "z_IC_ion_species_1" => OptionsDict("temperature_amplitude" => PR322_T_1V,),
                "r_IC_ion_species_1" => OptionsDict("temperature_amplitude" => PR322_T_1V,),
                "vpa_IC_ion_species_1" => OptionsDict("temperature_amplitude" => PR322_T_1V,),
                "neutral_species_1" => OptionsDict("initial_temperature" => PR322_T_1V,),
                "z_IC_neutral_species_1" => OptionsDict("temperature_amplitude" => PR322_T_1V),
                "r_IC_neutral_species_1" => OptionsDict("temperature_amplitude" => PR322_T_1V,),
                "vz_IC_neutral_species_1" => OptionsDict("temperature_amplitude" => PR322_T_1V,),
                "reactions" => OptionsDict("ionization_energy" => PR322_T_1V,),
                "fokker_planck_collisions" => OptionsDict("sd_temp" => PR322_T_1V,),
                "ion_source_1" => OptionsDict("PI_temperature_target_amplitude" => PR322_T_1V,)
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
                println("It can be replaced using the ion source term (with the value that was `ionization_frequency` set as the `source_amplitude`), with a section like:")
                println("[ion_source]")
                println("z_profile = \"constant\"")
                println("source_amplitude = 1.0")
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
        else
            new_section_name, new_key = top_level_update_map[k]
            updated_input[new_section_name] = get(updated_input, new_section_name, DictType())
            updated_input[new_section_name][new_key] = v
        end
    end

    combined_update_map = sections_update_map
    if update_definitions_322
        is_2V = ("vperp" ∈ keys(updated_input)
                 && (("ngrid" ∈ keys(updated_input["vperp"]) && updated_input["vperp"]["ngrid"] > 1)
                     || ("nelement" ∈ keys(updated_input["vperp"]) && updated_input["vperp"]["nelement"] > 1)
                    )
                )
        if is_2V
            combined_update_map = merge(combined_update_map, PR322_definitions_update_map_2V)
        else
            combined_update_map = merge(combined_update_map, PR322_definitions_update_map_1V)
        end
    end

    # Fix updated options in the sections
    existing_sections = keys(updated_input)
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
                section[new_option_setting] = old_value
            end
        end
    end

    return updated_input
end

function update_input_file(filename)
    file_text = read(filename, String)
    mv(filename, "$filename.unmodified")

    comments = String[]
    for line ∈ split(file_text, "\n")
        if occursin("#", line)
            push!(comments, line)
        end
    end
    if length(comments) > 0
        println("Found comments in file. These are not copied to output. If you want to keep them, copy by hand!")
        println("Lines with comments were:")
        for line in comments
            println(line)
        end
    end

    original_input = TOML.parse(file_text)

    updated_input = update_input_dict(original_input)

    # Write the updated file. We have moved the original file, so this does not need to
    # overwrite. Pass `truncate=false` to ensure we never accidentally delete a file, even
    # though this should never happen anyway.
    if isfile(filename)
        error("$filename already exists")
    end
    open(filename; write=true) do io
        TOML.print(io, updated_input)
    end

    return nothing
end

using moment_kinetics.command_line_options.ArgParse
if abspath(PROGRAM_FILE) == @__FILE__
    s = ArgParseSettings()
    @add_arg_table! s begin
        "inputfile"
            help = "Name of TOML input file to update."
            arg_type = String
            default = nothing
        "--update-definitions-322"
            help = "Update definitions and dimensionless variables according to the changes in PR #322 (April 2025)"
            action = :store_true
    end
    args = parse_args(s)

    update_input_file(args.inputfile; update_definitions_322=args.update_definitions_322)
end
