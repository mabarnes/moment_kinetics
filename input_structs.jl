module input_structs

export time_input
export advection_input, advection_input_mutable
export grid_input, grid_input_mutable
export initial_condition_input, initial_condition_input_mutable
export species_parameters, species_parameters_mutable
export species_composition

using type_definitions: mk_float, mk_int

struct time_input
    nstep::mk_int
    dt::mk_float
    nwrite::mk_int
    use_semi_lagrange::Bool
    n_rk_stages::mk_int
end
mutable struct advection_input_mutable
    # advection speed option
    option::String
    # constant advection speed to use with the "constant" advection option
    constant_speed::mk_float
    # for option = "oscillating", advection speed is of form
    # speed = constant_speed*(1 + oscillation_amplitude*sinpi(frequency*t))
    frequency::mk_float
    oscillation_amplitude::mk_float
end
struct advection_input
    # advection speed option
    option::String
    # constant advection speed to use with the "constant" advection option
    constant_speed::mk_float
    # for option = "oscillating", advection speed is of form
    # speed = constant_speed*(1 + oscillation_amplitude*sinpi(frequency*t))
    frequency::mk_float
    oscillation_amplitude::mk_float
end
mutable struct grid_input_mutable
    # name of the variable associated with this coordinate
    name::String
    # number of grid points per element
    ngrid::mk_int
    # number of elements
    nelement::mk_int
    # box length
    L::mk_float
    # discretization option
    discretization::String
    # finite difference option (only used if discretization is "finite_difference")
    fd_option::String
    # boundary option
    bc::String
    # mutable struct containing advection speed options
    advection::advection_input_mutable
end
struct grid_input
    # name of the variable associated with this coordinate
    name::String
    # number of grid points per element
    ngrid::mk_int
    # number of elements
    nelement::mk_int
    # box length
    L::mk_float
    # discretization option
    discretization::String
    # finite difference option (only used if discretization is "finite_difference")
    fd_option::String
    # boundary option
    bc::String
    # struct containing advection speed options
    advection::advection_input
end
mutable struct initial_condition_input_mutable
    # initialization inputs for one coordinate of a separable distribution function
    initialization_option::String
    # inputs for "gaussian" initial condition
    width::mk_float
    # inputs for "sinusoid" initial condition
    wavenumber::mk_int
    amplitude::mk_float
    # inputs for "monomial" initial condition
    monomial_degree::mk_int
end
struct initial_condition_input
    # initialization inputs for one coordinate of a separable distribution function
    initialization_option::String
    # inputs for "gaussian" initial condition
    width::mk_float
    # inputs for "sinusoid" initial condition
    wavenumber::mk_int
    amplitude::mk_float
    # inputs for "monomial" initial condition
    monomial_degree::mk_int
end
mutable struct species_parameters_mutable
    # type is the type of species; options are 'ion' or 'neutral'
    type::String
    # array containing the initial line-averaged temperature for this species
    initial_temperature::mk_float
    # array containing the initial line-averaged density for this species
    initial_density::mk_float
    # struct containing the initial condition info in z for this species
    z_IC::initial_condition_input_mutable
    # struct containing the initial condition info in vpa for this species
    vpa_IC::initial_condition_input_mutable
end
struct species_parameters
    # type is the type of species; options are 'ion' or 'neutral'
    type::String
    # array containing the initial line-averaged temperature for this species
    initial_temperature::mk_float
    # array containing the initial line-averaged density for this species
    initial_density::mk_float
    # struct containing the initial condition info in z for this species
    z_IC::initial_condition_input
    # struct containing the initial condition info in vpa for this species
    vpa_IC::initial_condition_input
end
struct species_composition
    # n_species = total number of evolved species (including ions, neutrals and electrons)
    n_species::mk_int
    # n_ion_species is the number of evolved ion species
    n_ion_species::mk_int
    # n_neutral_species is the number of evolved neutral species
    n_neutral_species::mk_int
    # if boltzmann_electron_response = true, the electron density is fixed to be Nₑ*(eϕ/T_e)
    boltzmann_electron_response::Bool
end

end
