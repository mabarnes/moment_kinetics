"""
Module for handling i/o for species specific input parameters
which are hosted in the composition and species structs for passing to functions
"""
module species_input

export get_species_input

using ..type_definitions: mk_float, mk_int
using ..input_structs: set_defaults_and_check_section!
using ..input_structs: species_composition, ion_species_parameters, neutral_species_parameters
using ..input_structs: spatial_initial_condition_input, velocity_initial_condition_input
using ..input_structs: boltzmann_electron_response, boltzmann_electron_response_with_simple_sheath

function get_species_input(toml_input)

    # read general composition parameters
    composition_section = set_defaults_and_check_section!(toml_input, "composition",
        # n_ion_species is the number of evolved ion species
        n_ion_species = 1,
        # n_neutral_species is the number of evolved neutral species
        n_neutral_species = 1,
        # * if electron_physics=boltzmann_electron_response, then the electron density is
        #   fixed to be N_e*(eϕ/T_e)
        # * if electron_physics=boltzmann_electron_response_with_simple_sheath, then the
        #   electron density is fixed to be N_e*(eϕ/T_e) and N_e is calculated w.r.t a
        #   reference value using J_||e + J_||i = 0 at z = 0
        electron_physics = boltzmann_electron_response,
        # initial Tₑ = 1
        T_e = 1.0,
        # wall temperature T_wall = Tw/Te
        T_wall = 1.0,
        # phi_wall at z = -L/2
        phi_wall = 0.0,
        # ratio of the neutral particle mass to the ion mass
        mn_over_mi = 1.0,
        # ratio of the electron particle mass to the ion mass
        me_over_mi = 1.0/1836.0,
        # if false use true Knudsen cosine for neutral wall bc
        use_test_neutral_wall_pdf = false,
        # The ion flux reaching the wall that is recycled as neutrals is reduced by
        # `recycling_fraction` to account for ions absorbed by the wall.
        recycling_fraction = 1.0,
        # gyrokinetic_ions = True -> use gyroaveraged fields at fixed guiding centre and moments of the pdf computed at fixed r
        # gyrokinetic_ions = False -> use drift kinetic approximation
        gyrokinetic_ions = false)

    nspec_ion = composition_section["n_ion_species"]
    nspec_neutral = composition_section["n_neutral_species"]
    nspec_tot = nspec_ion + nspec_neutral
    
    # read individual species parameters
    ion_spec_params_list = Array{ion_species_parameters,1}(undef,nspec_ion)
    neutral_spec_params_list = Array{neutral_species_parameters,1}(undef,nspec_neutral)
    for is in 1:nspec_ion
        spec_section = set_defaults_and_check_section!(toml_input, "ion_species_$is",
            # [ion_species_1], [ion_species_2], etc 
            # mass of ion species
            mass = 1.0,
            # charge number
            zeds = 1.0,
            # initial density
            initial_density = 1.0,
            # initial temperature
            initial_temperature = 1.0)
        
        z_IC_section = set_defaults_and_check_section!(toml_input, "z_IC_ion_species_$is",
            # [ion_z_IC_species_1], [ion_z_IC_species_2], etc 
            initialization_option = "gaussian",
            width = 0.125,
            wavenumber = 1,
            density_amplitude = 0.001,
            density_phase = 0.0,
            upar_amplitude = 0.0,
            upar_phase = 0.0,
            temperature_amplitude = 0.0,
            temperature_phase = 0.0,
            monomial_degree = 2)
        z_IC_input = Dict(Symbol(k)=>v for (k,v) in z_IC_section)
        z_IC = spatial_initial_condition_input(; z_IC_input...)
        
        r_IC_section = set_defaults_and_check_section!(toml_input, "r_IC_ion_species_$is",
            # [ion_r_IC_species_1], [ion_r_IC_species_2], etc 
            initialization_option = "gaussian",
            width = 0.125,
            wavenumber = 1,
            density_amplitude = 0.001,
            density_phase = 0.0,
            upar_amplitude = 0.0,
            upar_phase = 0.0,
            temperature_amplitude = 0.0,
            temperature_phase = 0.0,
            monomial_degree = 2)
        r_IC_input= Dict(Symbol(k)=>v for (k,v) in r_IC_section)
        r_IC = spatial_initial_condition_input(; r_IC_input...)
        
        vpa_IC_section = set_defaults_and_check_section!(toml_input, "vpa_IC_ion_species_"*string(is),
            # [ion_vpa_IC_species_1], [ion_vpa_IC_species_2], etc 
            initialization_option = "gaussian",
            width = 1.0,
            wavenumber = 1,
            density_amplitude = 1.0,
            density_phase = 0.0,
            upar_amplitude = 0.0,
            upar_phase = 0.0,
            temperature_amplitude = 0.0,
            temperature_phase = 0.0,
            monomial_degree = 2,
            # need to read resolutions before setting defaults here
            v0 = 1.0,
            vth0 = 1.0,
            vpa0 = 1.0,
            vperp0 = 1.0)
        vpa_IC_input= Dict(Symbol(k)=>v for (k,v) in vpa_IC_section)
        vpa_IC = velocity_initial_condition_input(; vpa_IC_input...)
        
        IC_input = Dict("z_IC" => z_IC, "r_IC" => r_IC, "vpa_IC" => vpa_IC)
        type_input = Dict("type" => "ion")
        spec_section = merge(spec_section, IC_input, type_input)
        spec_input = Dict(Symbol(k)=>v for (k,v) in spec_section)
        ion_spec_params_list[is] = ion_species_parameters(; spec_input...)
    end
    for isn in 1:nspec_neutral
        spec_section = set_defaults_and_check_section!(toml_input, "neutral_species_"*string(isn),
            # [neutral_species_1], [neutral_species_2], etc
            # mass of neutral species
            mass = 1.0,
            # initial density
            initial_density = 1.0,
            # initial temperature
            initial_temperature = 1.0)
        
        z_IC_section = set_defaults_and_check_section!(toml_input, "z_IC_neutral_species_"*string(isn),
            # [neutral_z_IC_species_1], [neutral_z_IC_species_2], etc 
            initialization_option = "gaussian",
            width = 0.125,
            wavenumber = 1,
            density_amplitude = 0.001,
            density_phase = 0.0,
            upar_amplitude = 0.0,
            upar_phase = 0.0,
            temperature_amplitude = 0.0,
            temperature_phase = 0.0,
            monomial_degree = 2)
        z_IC_input = Dict(Symbol(k)=>v for (k,v) in z_IC_section)
        z_IC = spatial_initial_condition_input(; z_IC_input...)
        
        r_IC_section = set_defaults_and_check_section!(toml_input, "r_IC_neutral_species_"*string(isn),
            # [neutral_r_IC_species_1], [neutral_r_IC_species_2], etc 
            initialization_option = "gaussian",
            width = 0.125,
            wavenumber = 1,
            density_amplitude = 0.001,
            density_phase = 0.0,
            upar_amplitude = 0.0,
            upar_phase = 0.0,
            temperature_amplitude = 0.0,
            temperature_phase = 0.0,
            monomial_degree = 2)
        r_IC_input= Dict(Symbol(k)=>v for (k,v) in r_IC_section)
        r_IC = spatial_initial_condition_input(; r_IC_input...)
        
        vpa_IC_section = set_defaults_and_check_section!(toml_input, "vz_IC_neutral_species_$isn",
            # [neutral_vpa_IC_species_1], [neutral_vpa_IC_species_2], etc 
            initialization_option = "gaussian",
            width = 1.0,
            wavenumber = 1,
            density_amplitude = 1.0,
            density_phase = 0.0,
            upar_amplitude = 0.0,
            upar_phase = 0.0,
            temperature_amplitude = 0.0,
            temperature_phase = 0.0,
            monomial_degree = 2,
            # need to read resolutions before setting defaults here
            v0 = 1.0,
            vth0 = 1.0,
            vpa0 = 1.0,
            vperp0 = 1.0)
        vpa_IC_input= Dict(Symbol(k)=>v for (k,v) in vpa_IC_section)
        vpa_IC = velocity_initial_condition_input(; vpa_IC_input...)
        
        IC_input = Dict("z_IC" => z_IC, "r_IC" => r_IC, "vpa_IC" => vpa_IC)
        type_input = Dict("type" => "neutral")
        spec_section = merge(spec_section, IC_input, type_input)
        spec_input = Dict(Symbol(k)=>v for (k,v) in spec_section)
        neutral_spec_params_list[isn] = neutral_species_parameters(; spec_input...)
    end
    # construct composition dict
    #println(composition_section)
    #println(ion_spec_params_list)
    #println(neutral_spec_params_list)
    #println(composition_section["n_ion_species"])
    species_dict = Dict("n_species" => nspec_tot, "ion" => ion_spec_params_list, "neutral" => neutral_spec_params_list)
    #println(species_dict)
    composition_section = merge(composition_section,species_dict)
    #println(composition_section)
    input = Dict(Symbol(k)=>v for (k,v) in composition_section)
    #println(input)
    # construct composition struct
    composition = species_composition(; input...)
    #println(composition)
    #println("")
    #for is in 1:composition.n_ion_species
        #println("ion_species_"*string(is))
        #println("mass: ",composition.ion[1].mass)
        #println("Zs: ",composition.ion[1].zeds)
        #println("")
    #end

    #for isn in 1:composition.n_neutral_species
    #    println("neutral_species_"*string(isn))
    #    println("mass: ",composition.neutral[1].mass)
    #    println("")
    #end
    
    ## checks and errors
    if !(0.0 <= composition.recycling_fraction <= 1.0)
        error("recycling_fraction must be between 0 and 1. Got $recycling_fraction.")
    end
        
    return composition
end

end
