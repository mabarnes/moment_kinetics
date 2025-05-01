#!julia

# Script to convert 'old' input files to latest format, updating any option names, etc.
# that have changed.
#
# Relies on loading TOML and re-writing file, so will lose all comments and may re-arrange
# the file. You may want to keep your original input file and just copy over the new
# sections.
#
# The original input file is not deleted, but is renamed with a '.unmodified' suffix.

using TOML

# Define the map of old name to new section and name.
# Note, a couple of options have some special handling that is defined within the
# update_input_file() function.
const top_level_update_map = Dict(
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

# If the "new option name" is a Dict, it gives a map from the original
# option values to new option names and values. If the new value is `nothing` it is
# replaced by the old value.
const sections_update_map = Dict(
    "timestepping" => Dict("implicit_electron_advance" => Dict(true => Dict("timestepping" => Dict("kinetic_electron_solver" => "implicit_steady_state"),),
                                                               "lu" => Dict("timestepping" => Dict("kinetic_electron_solver" => "implicit_steady_state", "kinetic_electron_preconditioner" => "lu"),),
                                                               "adi" => Dict("timestepping" => Dict("kinetic_electron_solver" => "implicit_steady_state", "kinetic_electron_preconditioner" => "adi"),),
                                                               "static_condensation" => Dict("timestepping" => Dict("kinetic_electron_solver" => "implicit_steady_state", "kinetic_electron_preconditioner" => "static_condensation"),),
                                                              ),
                           "implicit_electron_time_evolving" => Dict(true => Dict("timestepping" => Dict("kinetic_electron_solver" => "implicit_time_evolving"),),
                                                                     "lu" => Dict("timestepping" => Dict("kinetic_electron_solver" => "implicit_time_evolving", "kinetic_electron_preconditioner" => "lu"),),
                                                                     "adi" => Dict("timestepping" => Dict("kinetic_electron_solver" => "implicit_time_evolving", "kinetic_electron_preconditioner" => "adi"),),
                                                                     "static_condensation" => Dict("timestepping" => Dict("kinetic_electron_solver" => "implicit_time_evolving", "kinetic_electron_preconditioner" => "static_condensation"),),
                                                              ),
                           "implicit_electron_ppar" => Dict(true => Dict("timestepping" => Dict("kinetic_electron_solver" => "implicit_ppar_implicit_pseudotimestep"),),
                                                            "lu" => Dict("timestepping" => Dict("kinetic_electron_solver" => "implicit_ppar_implicit_pseudotimestep", "kinetic_electron_preconditioner" => "lu"),),
                                                            "adi" => Dict("timestepping" => Dict("kinetic_electron_solver" => "implicit_ppar_implicit_pseudotimestep", "kinetic_electron_preconditioner" => "adi"),),
                                                            "static_condensation" => Dict("timestepping" => Dict("kinetic_electron_solver" => "implicit_ppar_implicit_pseudotimestep", "kinetic_electron_preconditioner" => "static_condensation"),),
                                                              ),
                           "implicit_ion_advance" => Dict(true => Dict("timestepping" => Dict("kinetic_ion_solver" => "full_implicit_ion_advance"))),
                           "implicit_vpa_advection" => Dict(true => Dict("timestepping" => Dict("kinetic_ion_solver" => "implicit_ion_vpa_advection"))),
                          ),
   )

function update_input_dict(original_input)
    updated_input = Dict{String,Any}()

    # Get the existing sections first
    for (k,v) ∈ original_input
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
            section = get(updated_input, "krook_collisions", Dict{String,Any}())
            section["use_krook"] = true
            section["frequency_option"] = v
        elseif k == "nuii"
            section = get(updated_input, "fokker_planck_collisions", Dict{String,Any}())
            section["use_fokker_planck"] = true
            section[k] = v
        else
            new_section_name, new_key = top_level_update_map[k]
            updated_input[new_section_name] = get(updated_input, new_section_name, Dict{String,Any}())
            updated_input[new_section_name][new_key] = v
        end
    end

    # Fix updated options in the sections
    existing_sections = keys(updated_input)
    for (section_name, section_update_map) ∈ sections_update_map
        if section_name ∉ existing_sections
            continue
        end
        section = updated_input[section_name]
        existing_keys = keys(section)
        for (old_key, new_key_or_map) ∈ section_update_map
            if old_key ∉ existing_keys
                continue
            end
            old_value = pop!(section, old_key)
            if new_key_or_map isa AbstractDict
                if old_value ∉ keys(new_key_or_map)
                    continue
                end
                for (new_section_name, new_section_map) ∈ new_key_or_map[old_value]
                    new_section = get(updated_input, new_section_name, Dict{String,Any}())
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
            else
println("check $new_key_or_map, ", section[old_key])
                section[new_key_or_map] = old_value
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
    open(filename; write=true, truncate=false) do io
        TOML.print(io, updated_input)
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    update_input_file(ARGS[1])
end
