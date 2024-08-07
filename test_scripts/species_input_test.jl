export species_input_test

import moment_kinetics
using moment_kinetics.input_structs: set_defaults_and_check_section!
using moment_kinetics.type_definitions: mk_float, mk_int

Base.@kwdef struct ion_spec_params
    # mass
    mass::mk_float
    # charge number
    zeds::mk_float
end

Base.@kwdef struct neutral_spec_params
    # mass
    mass::mk_float
end

"""
"""
Base.@kwdef struct spec_composition
    # n_ion_species is the number of evolved ion species
    n_ion_species::mk_int
    # n_neutral_species is the number of evolved neutral species
    n_neutral_species::mk_int
    ion_species::Vector{ion_spec_params}
    neutral_species::Vector{neutral_spec_params}
end

const test_toml_base = Dict( "composition" => Dict("n_ion_species" => 2,"n_neutral_species" => 2),
                        "ion_species_1" => Dict("mass" => 2.0, "zeds" => 4.0),
                        "ion_species_2" => Dict("mass" => 3.0, "zeds" => 5.0),
                        "neutral_species_1" => Dict("mass" => 2.5),
                        "neutral_species_2" => Dict("mass" => 3.5))
                              

function species_input_test(toml_input=test_toml_base)
    comp_input_section = set_defaults_and_check_section!(toml_input, "composition",
        n_ion_species = 1,
        n_neutral_species = 1)
    nspec_ion = comp_input_section["n_ion_species"]
    nspec_neutral = comp_input_section["n_neutral_species"]
    # read species parameters
    ion_spec_params_list = Array{ion_spec_params,1}(undef,nspec_ion)
    neutral_spec_params_list = Array{neutral_spec_params,1}(undef,nspec_neutral)
    for is in 1:nspec_ion
        spec_input_section = set_defaults_and_check_section!(toml_input, "ion_species_"*string(is),
            mass = 1.0,
            zeds = 1.0)
        spec_input = Dict(Symbol(k)=>v for (k,v) in spec_input_section)
        println(spec_input)
        ion_spec_params_list[is] = ion_spec_params(; spec_input...)
    end
    for isn in 1:nspec_neutral
        spec_input_section = set_defaults_and_check_section!(toml_input, "neutral_species_"*string(isn),
            mass = 1.0)
        spec_input = Dict(Symbol(k)=>v for (k,v) in spec_input_section)
        println(spec_input)
        neutral_spec_params_list[isn] = neutral_spec_params(; spec_input...)
    end
    # construct composition dict
    println(comp_input_section)
    println(ion_spec_params_list)
    println(neutral_spec_params_list)
    println(comp_input_section["n_ion_species"])
    newdict = Dict("ion_species" => ion_spec_params_list, "neutral_species" => neutral_spec_params_list)
    println(newdict)
    comp_input_section = merge(comp_input_section,newdict)
    println(comp_input_section)
    input = Dict(Symbol(k)=>v for (k,v) in comp_input_section)
    println(input)
    # construct composition struct
    composition = spec_composition(; input...)
    println(composition)
    println("")
    for is in 1:composition.n_ion_species
        println("ion_species_"*string(is))
        println("mass: ",composition.ion_species[1].mass)
        println("Zs: ",composition.ion_species[1].zeds)
        println("")
    end

    for isn in 1:composition.n_neutral_species
        println("neutral_species_"*string(isn))
        println("mass: ",composition.neutral_species[1].mass)
        println("")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(".")

    species_input_test()
end