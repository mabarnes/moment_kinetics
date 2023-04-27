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
export io_input
export pp_input
export geometry_input

using ..type_definitions: mk_float, mk_int

using MPI

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
    nwrite_moments::mk_int
    nwrite_dfns::mk_int
    n_rk_stages::mk_int
    split_operators::Bool
    runtime_plots::Bool
    use_manufactured_solns_for_advance::Bool
    use_manufactured_solns_for_init::Bool
end

"""
"""
mutable struct advance_info
    vpa_advection::Bool
    z_advection::Bool
    r_advection::Bool
    neutral_z_advection::Bool
    neutral_r_advection::Bool
    neutral_vz_advection::Bool
    cx_collisions::Bool
    cx_collisions_1V::Bool
    ionization_collisions::Bool
    ionization_collisions_1V::Bool
    ionization_source::Bool
    numerical_dissipation::Bool
    source_terms::Bool
    continuity::Bool
    force_balance::Bool
    energy::Bool
    neutral_source_terms::Bool
    neutral_continuity::Bool
    neutral_force_balance::Bool
    neutral_energy::Bool
    rk_coefs::Array{mk_float,2}
    manufactured_solns_test::Bool
    r_diffusion::Bool #flag to control how r bc is imposed when r diffusion terms are present
    vpa_diffusion::Bool #flag to control how vpa bc is imposed when vpa diffusion terms are present
    vz_diffusion::Bool #flag to control how vz bc is imposed when vz diffusion terms are present
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
    # number of elements in global grid across ranks 
    nelement_global::mk_int
    # number of elements in local grid on this rank 
    nelement_local::mk_int
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
    # number of elements globally
    nelement_global::mk_int
    # number of elements locally
    nelement_local::mk_int
    # number of ranks involved in the calculation
    nrank::mk_int
    # rank of this process
    irank::mk_int
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
    # MPI communicator
    comm::MPI.Comm
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
    # struct containing the initial condition info in r for this species
    r_IC::initial_condition_input_mutable
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
    # struct containing the initial condition info in r for this species
    r_IC::initial_condition_input
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
    # if false -- wall bc uses true Knudsen cosine to specify neutral pdf leaving the wall
    # if true -- use a simpler pdf that is easier to integrate
    use_test_neutral_wall_pdf::Bool
    # electron temperature used for Boltzmann response
    T_e::mk_float
    # wall temperature used if 'wall' BC selected for z coordinate; normalised by electron temperature
    T_wall::mk_float
    # wall potential used if electron_physics=boltzmann_electron_response_with_simple_sheath
    phi_wall::mk_float
    # constant for testing nonzero Er
    Er_constant::mk_float
    # constant controlling divergence at wall boundaries in MMS test
	epsilon_offset::mk_float
    # logical controlling whether or not dfni(vpabar,z,r) or dfni(vpa,z,r) in MMS test
    use_vpabar_in_mms_dfni::Bool
    # associated float controlling form of assumed potential in MMS test
    alpha_switch::mk_float    
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
    # if true, forces Er = 0.0 at wall plates 
    force_Er_zero_at_wall::Bool 
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
mutable struct geometry_input
    # Bz/Bref
    Bzed::mk_float
    # Btot/Bref
    Bmag::mk_float
    # bz -- unit vector component in z direction
    bzed::mk_float
    # bz -- unit vector component in zeta direction
    bzeta::mk_float
    # Bzeta/Bref
    Bzeta::mk_float
    # rhostar ion (ref)
    rhostar::mk_float #used to premultiply ExB drift terms
end

@enum binary_format_type hdf5 netcdf
export hdf5, netcdf

"""
Settings and input for setting up file I/O
"""
Base.@kwdef struct io_input
    output_dir::String
    run_name::String
    ascii_output::Bool
    binary_format::binary_format_type
    parallel_io::Bool
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
    # if plot_vth0_vs_t = true, create plots of species vth(z0) vs time
    plot_vth0_vs_t::Bool
    # if plot_qpar0_vs_t = true, create plots of species qpar(z0) vs time
    plot_qpar0_vs_t::Bool
    # if plot_dens_vs_z_t = true, create plot of species density vs z and time
    plot_dens_vs_z_t::Bool
    # if plot_upar_vs_z_t = true, create plot of species parallel flow vs z and time
    plot_upar_vs_z_t::Bool
    # if plot_ppar_vs_z_t = true, create plot of species parallel pressure vs z and time
    plot_ppar_vs_z_t::Bool
    # if plot_Tpar_vs_z_t = true, create plot of species parallel temperature vs z and time
    plot_Tpar_vs_z_t::Bool
    # if plot_qpar_vs_z_t = true, create plot of species parallel heat flux vs z and time
    plot_qpar_vs_z_t::Bool
    # if animate_dens_vs_z = true, create animation of species density vs z at different time slices
    animate_dens_vs_z::Bool
    # if animate_upar_vs_z = true, create animation of species parallel flow vs z at different time slices
    animate_upar_vs_z::Bool
    # if animate_ppar_vs_z = true, create animation of species parallel pressure vs z at different time slices
    animate_ppar_vs_z::Bool
    # if animate_Tpar_vs_z = true, create animation of species parallel temperature vs z at different time slices
    animate_Tpar_vs_z::Bool
    # if animate_vth_vs_z = true, create animation of species thermal speed vs z at different time slices
    animate_vth_vs_z::Bool
    # if animate_qpar_vs_z = true, create animation of species parallel heat flux vs z at different time slices
    animate_qpar_vs_z::Bool
    # if plot_f_unnormalized_vs_vpa_z = true, create plot of f_unnorm(v_parallel_unnorm,z)
    # at it=itime_max
    plot_f_unnormalized_vs_vpa_z::Bool
    # if animate_f_vs_vpa_z = true, create animation of f(z,vpa) at different time slices
    animate_f_vs_vpa_z::Bool
    # if animate_f_unnormalized = true, create animation of
    # f_unnorm(v_parallel_unnorm,z) at different time slices
    animate_f_unnormalized::Bool
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
    # if animate_f_vs_vpa_r = true, create animation of f(vpa,r) at different time slices
    animate_f_vs_vpa_r::Bool
    # if animate_f_vs_vperp_z = true, create animation of f(vperp,z) at different time slices
    animate_f_vs_vperp_z::Bool
    # if animate_f_vs_vperp_r = true, create animation of f(vperp,r) at different time slices
    animate_f_vs_vperp_r::Bool
    # if animate_f_vs_vperp_vpa = true, create animation of f(vperp,vpa) at different time slices
    animate_f_vs_vperp_vpa::Bool
    # if animate_f_vs_r_z = true, create animation of f(r,z) at different time slices
    animate_f_vs_r_z::Bool
    # if animate_f_vs_vz_z = true, create animation of f(vz,z) at different time slices
    animate_f_vs_vz_z::Bool
    # if animate_f_vs_vr_r = true, create animation of f(vr,r) at different time slices
    animate_f_vs_vr_r::Bool
    # if animate_Er_vs_r_z = true, create animation of Er(r,z) at different time slices
    animate_Er_vs_r_z::Bool
    # if animate_Ez_vs_r_z = true, create animation of Ez(r,z) at different time slices
    animate_Ez_vs_r_z::Bool
    # if animate_phi_vs_r_z = true, create animation of phi(r,z) at different time slices
    animate_phi_vs_r_z::Bool
    # if plot_phi_vs_r0_z = true, plot last timestep phi[z,ir0]
    plot_phi_vs_r0_z::Bool
    # if plot_Ez_vs_r0_z = true, plot last timestep Ez[z,ir0]
    plot_Ez_vs_r0_z::Bool
    # if plot_wall_Ez_vs_r = true, plot last timestep Ez[z_wall,r]
    plot_wall_Ez_vs_r::Bool
    # if plot_Er_vs_r0_z  = true, plot last timestep Er[z,ir0]
    plot_Er_vs_r0_z::Bool
    # if plot_wall_Er_vs_r = true, plot last timestep Er[z_wall,r]
    plot_wall_Er_vs_r::Bool
	# if plot_density_vs_r0_z = true  plot last timestep density[z,ir0]
	plot_density_vs_r0_z::Bool
	# if plot_wall_density_vs_r = true  plot last timestep density[z_wall,r]
	plot_wall_density_vs_r::Bool
    # if plot_density_vs_r_z = true plot density vs r z at last timestep
	plot_density_vs_r_z::Bool
	# if animate_density_vs_r_z = true animate density vs r z
	animate_density_vs_r_z::Bool
	# if plot_parallel_flow_vs_r0_z = true  plot last timestep parallel_flow[z,ir0]
	plot_parallel_flow_vs_r0_z::Bool
	# if plot_wall_parallel_flow_vs_r = true  plot last timestep parallel_flow[z_wall,r]
	plot_wall_parallel_flow_vs_r::Bool
    # if plot_parallel_flow_vs_r_z = true plot parallel_flow vs r z at last timestep
	plot_parallel_flow_vs_r_z::Bool
	# if animate_parallel_flow_vs_r_z = true animate parallel_flow vs r z
	animate_parallel_flow_vs_r_z::Bool
	# if plot_parallel_pressure_vs_r0_z = true  plot last timestep parallel_pressure[z,ir0]
	plot_parallel_pressure_vs_r0_z::Bool
	# if plot_wall_parallel_pressure_vs_r = true  plot last timestep parallel_pressure[z_wall,r]
	plot_wall_parallel_pressure_vs_r::Bool
    # if plot_parallel_pressure_vs_r_z = true plot parallel_pressure vs r z at last timestep 
	plot_parallel_pressure_vs_r_z::Bool
    # if animate_parallel_pressure_vs_r_z = true animate parallel_pressure vs r z
	animate_parallel_pressure_vs_r_z::Bool
    # if plot_parallel_temperature_vs_r0_z = true  plot last timestep parallel_temperature[z,ir0]
    plot_parallel_temperature_vs_r0_z::Bool
    # if plot_wall_parallel_temperature_vs_r = true  plot last timestep parallel_temperature[z_wall,r]
    plot_wall_parallel_temperature_vs_r::Bool
    # if plot_parallel_temperature_vs_r_z = true plot parallel_temperature vs r z at last timestep
    plot_parallel_temperature_vs_r_z::Bool
    # if animate_parallel_temperature_vs_r_z = true animate parallel_temperature vs r z
    animate_parallel_temperature_vs_r_z::Bool
    # if plot_wall_pdf = true then plot the ion distribution (vpa,vperp,z,r) in the element nearest the wall at the last timestep 
    plot_wall_pdf::Bool
    # animations of moments will use one in every nwrite_movie data slices
    nwrite_movie::mk_int
    # itime_min is the minimum time index at which to start animations of the moments
    itime_min::mk_int
    # itime_max is the final time index at which to end animations of the moments
    # if itime_max < 0, the value used will be the total number of time slices
    itime_max::mk_int
    # animations of pdfs will use one in every nwrite_movie data slices
    nwrite_movie_pdfs::mk_int
    # itime_min_pdfs is the minimum time index at which to start animations of the pdfs
    itime_min_pdfs::mk_int
    # itime_max_pdfs is the final time index at which to end animations of the pdfs
    # if itime_max < 0, the value used will be the total number of time slices
    itime_max_pdfs::mk_int
    # ivpa0 is the ivpa index used when plotting data at a single vpa location
    ivpa0::mk_int
    # ivperp0 is the ivperp index used when plotting data at a single vperp location
    ivperp0::mk_int
    # iz0 is the iz index used when plotting data at a single z location
    iz0::mk_int
    # ir0 is the ir index used when plotting data at a single r location
    ir0::mk_int
    # ivz0 is the ivz index used when plotting data at a single vz location
    ivz0::mk_int
    # ivr0 is the ivr index used when plotting data at a single vr location
    ivr0::mk_int
    # ivzeta0 is the ivzeta index used when plotting data at a single vzeta location
    ivzeta0::mk_int
end

end
