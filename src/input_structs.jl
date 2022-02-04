"""
"""
module input_structs

export evolve_moments_options
export time_input
export advection_input, advection_input_mutable
export grid_input, grid_input_mutable
export initial_condition_input, initial_condition_input_mutable
export species_parameters, species_parameters_mutable
export species_composition
export drive_input, drive_input_mutable
export collisions_input
export pp_input

using ..type_definitions: mk_float, mk_int

"""
"""
mutable struct evolve_moments_options
    density::Bool
    parallel_flow::Bool
    parallel_pressure::Bool
    conservation::Bool
    #advective_form::Bool
end

"""
"""
struct time_input
    nstep::mk_int
    dt::mk_float
    nwrite::mk_int
    use_semi_lagrange::Bool
    n_rk_stages::mk_int
    split_operators::Bool
end

"""
"""
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

"""
"""
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

"""
"""
@enum electron_physics_type boltzmann_electron_response boltzmann_electron_response_with_simple_sheath
export electron_physics_type, boltzmann_electron_response, boltzmann_electron_response_with_simple_sheath

"""
"""
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

"""
"""
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

"""
"""
mutable struct initial_condition_input_mutable
    # initialization inputs for one coordinate of a separable distribution function
    initialization_option::String
    # inputs for "gaussian" initial condition
    width::mk_float
    # inputs for "sinusoid" initial condition
    wavenumber::mk_int
    density_amplitude::mk_float
    density_phase::mk_float
    upar_amplitude::mk_float
    upar_phase::mk_float
    temperature_amplitude::mk_float
    temperature_phase::mk_float
    # inputs for "monomial" initial condition
    monomial_degree::mk_int
end

"""
"""
struct initial_condition_input
    # initialization inputs for one coordinate of a separable distribution function
    initialization_option::String
    # inputs for "gaussian" initial condition
    width::mk_float
    # inputs for "sinusoid" initial condition
    wavenumber::mk_int
    density_amplitude::mk_float
    density_phase::mk_float
    upar_amplitude::mk_float
    upar_phase::mk_float
    temperature_amplitude::mk_float
    temperature_phase::mk_float
    # inputs for "monomial" initial condition
    monomial_degree::mk_int
end

"""
"""
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

"""
"""
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

"""
"""
mutable struct species_composition
    # n_species = total number of evolved species (including ions, neutrals and electrons)
    n_species::mk_int
    # n_ion_species is the number of evolved ion species
    n_ion_species::mk_int
    # n_neutral_species is the number of evolved neutral species
    n_neutral_species::mk_int
    # * if electron_physics=boltzmann_electron_response, the electron density is fixed
    #   to be Nₑ*(eϕ/T_e)
    # * if electron_physics=boltzmann_electron_response_with_simple_sheath, the electron
    #   density is fixed to be Nₑ*(eϕ/T_e) and N_e is calculated using a current
    #   condition at the wall
    electron_physics::electron_physics_type
    # Species indices that represent ions
    ion_species_range::UnitRange{mk_int}
    # Species indices that represent neutrals
    neutral_species_range::UnitRange{mk_int}
    # electron temperature used for Boltzmann response
    T_e::mk_float
    # wall temperature used if 'wall' BC selected for z coordinate; normalised by electron temperature
    T_wall::mk_float
    # wall potential used if electron_physics=boltzmann_electron_response_with_simple_sheath 
    phi_wall::mk_float
    # ratio of the neutral particle mass to the ion mass
    mn_over_mi::mk_float
    # ratio of the electron particle mass to the ion mass
    me_over_mi::mk_float
    # scratch buffer whose size is n_species
    scratch::Vector{mk_float}
end

"""
"""
mutable struct drive_input_mutable
    # if drive.phi = true, include external electrostatic potential
    force_phi::Bool
    # if external field included, it is of the form
    # phi(z,t=0)*amplitude*sinpi(t*frequency)
    amplitude::mk_float
    frequency::mk_float
end

"""
"""
struct drive_input
    # if drive.phi = true, include external electrostatic potential
    force_phi::Bool
    # if external field included, it is of the form
    # phi(z,t=0)*amplitude*sinpi(t*frequency)
    amplitude::mk_float
    frequency::mk_float
end

"""
"""
mutable struct collisions_input
    # charge exchange collision frequency
    charge_exchange::mk_float
    # ionization collision frequency
    ionization::mk_float
    # if constant_ionization_rate = true, use an ionization term that is constant in z
    constant_ionization_rate::Bool
end

"""
"""
struct pp_input
    # if calculate_frequencies = true, calculate and print the frequency and growth/decay
    # rate of phi, using values at iz = iz0
    calculate_frequencies::Bool
    # if plot_phi0_vs_t = true, create plot of phi(z0) vs time
    plot_phi0_vs_t::Bool
    # if plot_phi_vs_z_t = true, create plot of phi vs z and time
    plot_phi_vs_z_t::Bool
    # if animate_phi_vs_z = true, create animation of phi vs z at different time slices
    animate_phi_vs_z::Bool
    # if plot_dens0_vs_t = true, create plots of species density(z0) vs time
    plot_dens0_vs_t::Bool
    # if plot_upar0_vs_t = true, create plots of species upar(z0) vs time
    plot_upar0_vs_t::Bool
    # if plot_ppar0_vs_t = true, create plots of species ppar(z0) vs time
    plot_ppar0_vs_t::Bool
    # if plot_qpar0_vs_t = true, create plots of species qpar(z0) vs time
    plot_qpar0_vs_t::Bool
    # if plot_dens_vs_z_t = true, create plot of species density vs z and time
    plot_dens_vs_z_t::Bool
    # if plot_upar_vs_z_t = true, create plot of species parallel flow vs z and time
    plot_upar_vs_z_t::Bool
    # if plot_ppar_vs_z_t = true, create plot of species parallel pressure vs z and time
    plot_ppar_vs_z_t::Bool
    # if plot_qpar_vs_z_t = true, create plot of species parallel heat flux vs z and time
    plot_qpar_vs_z_t::Bool
    # if animate_dens_vs_z = true, create animation of species density vs z at different time slices
    animate_dens_vs_z::Bool
    # if animate_upar_vs_z = true, create animation of species parallel flow vs z at different time slices
    animate_upar_vs_z::Bool
    # if animate_f_vs_z_vpa = true, create animation of f(z,vpa) at different time slices
    animate_f_vs_vpa_z::Bool
    # if animate_f_vs_z_vpa0 = true, create animation of f(z,vpa0) at different time slices
    animate_f_vs_vpa0_z::Bool
    # if animate_f_vs_z0_vpa = true, create animation of f(z0,vpa) at different time slices
    animate_f_vs_vpa_z0::Bool
    # if animate_deltaf_vs_z_vpa = true, create animation of δf(z,vpa) at different time slices
    animate_deltaf_vs_vpa_z::Bool
    # if animate_deltaf_vs_z_vpa0 = true, create animation of δf(z,vpa0) at different time slices
    animate_deltaf_vs_vpa0_z::Bool
    # if animate_deltaf_vs_z0_vpa = true, create animation of δf(z0,vpa) at different time slices
    animate_deltaf_vs_vpa_z0::Bool
    # animations will use one in every nwrite_movie data slices
    nwrite_movie::mk_int
    # itime_min is the minimum time index at which to start animations
    itime_min::mk_int
    # itime_max is the final time index at which to end animations
    # if itime_max < 0, the value used will be the total number of time slices
    itime_max::mk_int
    # ivpa0 is the ivpa index used when plotting data at a single vpa location
    ivpa0::mk_int
    # iz0 is the iz index used when plotting data at a single z location
    iz0::mk_int
    # ir0 is the ir index used when plotting data at a single r location
    ir0::mk_int
end

end
