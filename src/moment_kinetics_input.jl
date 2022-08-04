"""
"""
module moment_kinetics_input

export mk_input
export performance_test
#export advective_form

using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float
using ..communication
using ..file_io: input_option_error, open_output_file
using ..finite_differences: fd_check_option
using ..input_structs
using ..numerical_dissipation: setup_numerical_dissipation

@enum RunType single performance_test scan
const run_type = single

import Base: get
"""
Utility mothod for converting a string to an Enum when getting from a Dict, based on the
type of the default value
"""
function get(d::Dict, key, default::Enum)
    valstring = get(d, key, nothing)
    if valstring == nothing
        return default
    # instances(typeof(default)) gets the possible values of the Enum. Then convert to
    # Symbol, then to String.
    elseif valstring ∈ String.(Symbol.(instances(typeof(default))))
        return eval(Symbol(valstring))
    else
        error("Expected a $(typeof(default)), but '$valstring' is not in "
              * "$(instances(typeof(default)))")
    end
end

"""
"""
function mk_input(scan_input=Dict())

    # n_ion_species is the number of evolved ion species
    # currently only n_ion_species = 1 is supported
    n_ion_species = 1
    # n_neutral_species is the number of evolved neutral species
    # currently only n_neutral_species = 0 is supported
    n_neutral_species = get(scan_input, "n_neutral_species", 1)
    # * if electron_physics=boltzmann_electron_response, then the electron density is
    #   fixed to be N_e*(eϕ/T_e)
    # * if electron_physics=boltzmann_electron_response_with_simple_sheath, then the
    #   electron density is fixed to be N_e*(eϕ/T_e) and N_e is calculated w.r.t a
    #   reference value using J_||e + J_||i = 0 at z = 0
    electron_physics = get(scan_input, "electron_physics", boltzmann_electron_response)
    
    z, r, vpa, species, composition, drive, evolve_moments, collisions =
        load_defaults(n_ion_species, n_neutral_species, electron_physics)

    # this is the prefix for all output files associated with this run
    run_name = get(scan_input, "run_name", "wallBC")
    # this is the directory where the simulation data will be stored
    base_directory = get(scan_input, "base_directory", "runs")
    output_dir = string(base_directory, "/", run_name)
    # if evolve_moments.density = true, evolve density via continuity eqn
    # and g = f/n via modified drift kinetic equation
    evolve_moments.density = get(scan_input, "evolve_moments_density", false)
    evolve_moments.parallel_flow = get(scan_input, "evolve_moments_parallel_flow", false)
    evolve_moments.parallel_pressure = get(scan_input, "evolve_moments_parallel_pressure", false)
    evolve_moments.conservation = get(scan_input, "evolve_moments_conservation", false)

    ####### specify any deviations from default inputs for evolved species #######
    # set initial Tₑ = 1
    composition.T_e = get(scan_input, "T_e", 1.0)
    # set wall temperature T_wall = Tw/Te
    composition.T_wall = get(scan_input, "T_wall", 1.0)
    # set initial neutral temperature Tn/Tₑ = 1
    # set initial nᵢ/Nₑ = 1.0
    # set phi_wall at z = 0
    composition.phi_wall = get(scan_input, "phi_wall", 0.0)
    
    
    species[1].z_IC.initialization_option = get(scan_input, "z_IC_option1", "gaussian")
    species[1].initial_density = get(scan_input, "initial_density1", 1.0)
    species[1].initial_temperature = get(scan_input, "initial_temperature1", 1.0)
    species[1].z_IC.density_amplitude = get(scan_input, "z_IC_density_amplitude1", 0.001)
    species[1].z_IC.density_phase = get(scan_input, "z_IC_density_phase1", 0.0)
    species[1].z_IC.upar_amplitude = get(scan_input, "z_IC_upar_amplitude1", 0.0)
    species[1].z_IC.upar_phase = get(scan_input, "z_IC_upar_phase1", 0.0)
    species[1].z_IC.temperature_amplitude = get(scan_input, "z_IC_temperature_amplitude1", 0.0)
    species[1].z_IC.temperature_phase = get(scan_input, "z_IC_temperature_phase1", 0.0)
    #species[1].z_IC.initialization_option = "bgk"
    # set initial neutral densiity = Nₑ
    species[1].vpa_IC.initialization_option = get(scan_input, "vpa_IC_option1", "gaussian")
    species[1].vpa_IC.density_amplitude = get(scan_input, "vpa_IC_density_amplitude1", 1.000)
    species[1].vpa_IC.density_phase = get(scan_input, "vpa_IC_density_phase1", 0.0)
    species[1].vpa_IC.upar_amplitude = get(scan_input, "vpa_IC_upar_amplitude1", 0.0)
    species[1].vpa_IC.upar_phase = get(scan_input, "vpa_IC_upar_phase1", 0.0)
    species[1].vpa_IC.temperature_amplitude = get(scan_input, "vpa_IC_temperature_amplitude1", 0.0)
    species[1].vpa_IC.temperature_phase = get(scan_input, "vpa_IC_temperature_phase1", 0.0)
    for (i, s) in enumerate(species[2:end])
        i = i+1
        s.z_IC.initialization_option = get(scan_input, "z_IC_option$i", species[1].z_IC.initialization_option)
        s.initial_density = get(scan_input, "initial_density$i", 0.5)
        s.initial_temperature = get(scan_input, "initial_temperature$i", species[1].initial_temperature)
        s.z_IC.width = get(scan_input, "z_IC_width$i", species[1].z_IC.width)
        s.z_IC.density_amplitude = get(scan_input, "z_IC_density_amplitude$i", species[1].z_IC.density_amplitude)
        s.z_IC.density_phase = get(scan_input, "z_IC_density_phase$i", species[1].z_IC.density_phase)
        s.z_IC.upar_amplitude = get(scan_input, "z_IC_upar_amplitude$i", species[1].z_IC.upar_amplitude)
        s.z_IC.upar_phase = get(scan_input, "z_IC_upar_phase$i", species[1].z_IC.upar_phase)
        s.z_IC.temperature_amplitude = get(scan_input, "z_IC_temperature_amplitude$i", species[1].z_IC.temperature_amplitude)
        s.z_IC.temperature_phase = get(scan_input, "z_IC_temperature_phase$i", species[1].z_IC.temperature_phase)
        s.vpa_IC.initialization_option = get(scan_input, "vpa_IC_option$i", species[1].vpa_IC.initialization_option)
        s.vpa_IC.width = get(scan_input, "vpa_IC_width$i", species[1].vpa_IC.width)
        s.vpa_IC.density_amplitude = get(scan_input, "vpa_IC_density_amplitude$i", species[1].vpa_IC.density_amplitude)
        s.vpa_IC.density_phase = get(scan_input, "vpa_IC_density_phase$i", species[1].vpa_IC.density_phase)
        s.vpa_IC.upar_amplitude = get(scan_input, "vpa_IC_upar_amplitude$i", species[1].vpa_IC.upar_amplitude)
        s.vpa_IC.upar_phase = get(scan_input, "vpa_IC_upar_phase$i", species[1].vpa_IC.upar_phase)
        s.vpa_IC.temperature_amplitude = get(scan_input, "vpa_IC_temperature_amplitude$i", species[1].vpa_IC.temperature_amplitude)
        s.vpa_IC.temperature_phase = get(scan_input, "vpa_IC_temperature_phase$i", species[1].vpa_IC.temperature_phase)
    end
    #################### end specification of species inputs #####################

    collisions.charge_exchange = get(scan_input, "charge_exchange_frequency", 2.0*sqrt(species[1].initial_temperature))
    collisions.ionization = get(scan_input, "ionization_frequency", collisions.charge_exchange)
    collisions.constant_ionization_rate = get(scan_input, "constant_ionization_rate", false)

    # parameters related to the time stepping
    nstep = get(scan_input, "nstep", 40000)
    dt = get(scan_input, "dt", 0.00025/sqrt(species[1].initial_temperature))
    nwrite = get(scan_input, "nwrite", 80)
    # use_semi_lagrange = true to use interpolation-free semi-Lagrange treatment
    # otherwise, solve problem solely using the discretization_option above
    use_semi_lagrange = get(scan_input, "use_semi_lagrange", false)
    # options are n_rk_stages = 1, 2, 3 or 4 (corresponding to forward Euler,
    # Heun's method, SSP RK3 and 4-stage SSP RK3)
    n_rk_stages = get(scan_input, "n_rk_stages", 4)
    split_operators = get(scan_input, "split_operators", false)
    runtime_plots = get(scan_input, "runtime_plots", false)

    # overwrite some default parameters related to the r grid
    # ngrid is number of grid points per element
    r.ngrid = get(scan_input, "r_ngrid", 1)
    # nelement is the number of elements
    r.nelement = get(scan_input, "r_nelement", 1)
    # determine the discretization option for the r grid
    # supported options are "chebyshev_pseudospectral" and "finite_difference"
    r.discretization = get(scan_input, "r_discretization", "finite_difference")
    r.fd_option = get(scan_input, "r_finite_difference_option", "third_order_upwind")
    # determine the boundary condition to impose in r
    # supported options are "constant", "periodic" and "wall"
    r.bc = get(scan_input, "r_bc", "periodic")

    # overwrite some default parameters related to the z grid
    # ngrid is number of grid points per element
    z.ngrid = get(scan_input, "z_ngrid", 9)
    # nelement is the number of elements
    z.nelement = get(scan_input, "z_nelement", 8)
    # determine the discretization option for the z grid
    # supported options are "chebyshev_pseudospectral" and "finite_difference"
    z.discretization = get(scan_input, "z_discretization", "chebyshev_pseudospectral")
    z.fd_option = get(scan_input, "z_finite_difference_option", "third_order_upwind")
    # determine the boundary condition to impose in z
    # supported options are "constant", "periodic" and "wall"
    z.bc = get(scan_input, "z_bc", "wall")

    # overwrite some default parameters related to the vpa grid
    # ngrid is the number of grid points per element
    vpa.ngrid = get(scan_input, "vpa_ngrid", 17)
    # nelement is the number of elements
    vpa.nelement = get(scan_input, "vpa_nelement", 10)
    # L is the box length in units of vthermal_species
    vpa.L = get(scan_input, "vpa_L", 8.0*sqrt(species[1].initial_temperature))
    # determine the boundary condition
    # only supported option at present is "zero" and "periodic"
    vpa.bc = get(scan_input, "vpa_bc", "periodic")
    # determine the discretization option for the vpa grid
    # supported options are "chebyshev_pseudospectral" and "finite_difference"
    vpa.discretization = get(scan_input, "vpa_discretization", "chebyshev_pseudospectral")
    vpa.fd_option = get(scan_input, "vpa_finite_difference_option", "third_order_upwind")

    num_diss_params = setup_numerical_dissipation(
        get(scan_input, "numerical_dissipation", Dict{String,Any}()))

    #########################################################################
    ########## end user inputs. do not modify following code! ###############
    #########################################################################

    t = time_input(nstep, dt, nwrite, use_semi_lagrange, n_rk_stages, split_operators,
                   runtime_plots)
    # replace mutable structures with immutable ones to optimize performance
    # and avoid possible misunderstandings
    z_advection_immutable = advection_input(z.advection.option, z.advection.constant_speed,
        z.advection.frequency, z.advection.oscillation_amplitude)
    z_immutable = grid_input("z", z.ngrid, z.nelement, z.L,
        z.discretization, z.fd_option, z.bc, z_advection_immutable)
    r_advection_immutable = advection_input(r.advection.option, r.advection.constant_speed,
        r.advection.frequency, r.advection.oscillation_amplitude)
    r_immutable = grid_input("r", r.ngrid, r.nelement, r.L,
        r.discretization, r.fd_option, r.bc, r_advection_immutable)
    vpa_advection_immutable = advection_input(vpa.advection.option, vpa.advection.constant_speed,
        vpa.advection.frequency, vpa.advection.oscillation_amplitude)
    vpa_immutable = grid_input("vpa", vpa.ngrid, vpa.nelement, vpa.L,
        vpa.discretization, vpa.fd_option, vpa.bc, vpa_advection_immutable)
    n_species = composition.n_species
    species_immutable = Array{species_parameters,1}(undef,n_species)
    for is ∈ 1:n_species
        if is <= n_ion_species
            species_type = "ion"
        elseif is <= n_ion_species + n_neutral_species
            species_type = "neutral"
        else
            species_type = "electron"
        end
        z_IC = initial_condition_input(species[is].z_IC.initialization_option,
            species[is].z_IC.width, species[is].z_IC.wavenumber,
            species[is].z_IC.density_amplitude, species[is].z_IC.density_phase,
            species[is].z_IC.upar_amplitude, species[is].z_IC.upar_phase,
            species[is].z_IC.temperature_amplitude, species[is].z_IC.temperature_phase,
            species[is].z_IC.monomial_degree)
        vpa_IC = initial_condition_input(species[is].vpa_IC.initialization_option,
            species[is].vpa_IC.width, species[is].vpa_IC.wavenumber,
            species[is].vpa_IC.density_amplitude, species[is].vpa_IC.density_phase,
            species[is].vpa_IC.upar_amplitude, species[is].vpa_IC.upar_phase,
            species[is].vpa_IC.temperature_amplitude,
            species[is].vpa_IC.temperature_phase, species[is].vpa_IC.monomial_degree)
        species_immutable[is] = species_parameters(species_type, species[is].initial_temperature,
            species[is].initial_density, z_IC, vpa_IC)
    end
    drive_immutable = drive_input(drive.force_phi, drive.amplitude, drive.frequency)

    # Make file to log some information about inputs into.
    # check to see if output_dir exists in the current directory
    # if not, create it
    if block_rank[] == 0
        isdir(output_dir) || mkdir(output_dir)
        io = open_output_file(string(output_dir,"/",run_name), "input")
    else
        io = devnull
    end

    # check input to catch errors/unsupported options
    check_input(io, output_dir, nstep, dt, use_semi_lagrange, r_immutable, z_immutable,
        vpa_immutable, composition, species_immutable, evolve_moments)

    # return immutable structs for z, vpa, species and composition
    all_inputs = (run_name, output_dir, evolve_moments, t, z_immutable, r_immutable, vpa_immutable,
                  composition, species_immutable, collisions, drive_immutable, num_diss_params)
    println(io, "\nAll inputs returned from mk_input():")
    println(io, all_inputs)
    close(io)

    return all_inputs
end

"""
"""
function load_defaults(n_ion_species, n_neutral_species, electron_physics)
    ############## options related to the equations being solved ###############
    evolve_density = false
    evolve_parallel_flow = false
    evolve_parallel_pressure = false
    conservation = true
    #advective_form = false
    evolve_moments = evolve_moments_options(evolve_density, evolve_parallel_flow, evolve_parallel_pressure, conservation)#advective_form)
    #################### parameters related to the z grid ######################
    # ngrid_z is number of grid points per element
    ngrid_z = 100
    # nelement_z is the number of elements
    nelement_z = 1
    # L_z is the box length
    L_z = 1.0
    # determine the boundary condition in z
    # currently supported options are "constant" and "periodic"
    boundary_option_z = "periodic"
    #boundary_option_z = "constant"
    # determine the discretization option for the z grid
    # supported options are "chebyshev_pseudospectral" and "finite_difference"
    #discretization_option_z = "chebyshev_pseudospectral"
    discretization_option_z = "finite_difference"
    # if discretization_option_z = "finite_difference", then
    # finite_difference_option_z determines the finite difference scheme to be used
    # supported options are "third_order_upwind", "second_order_upwind" and "first_order_upwind"
    #finite_difference_option_z = "first_order_upwind"
    #finite_difference_option_z = "second_order_upwind"
    finite_difference_option_z = "third_order_upwind"
    # determine the option used for the advection speed in z
    # supported options are "constant" and "oscillating",
    # in addition to the "default" option which uses dz/dt = vpa as the advection speed
    advection_option_z = "default"
    # constant advection speed in z to use with advection_option_z = "constant"
    advection_speed_z = 1.0
    # for advection_option_z = "oscillating", advection speed is of form
    # speed = advection_speed_z*(1 + oscillation_amplitude_z*sinpi(frequency_z*t))
    frequency_z = 1.0
    oscillation_amplitude_z = 1.0
    # mutable struct containing advection speed options/inputs for z
    advection_z = advection_input_mutable(advection_option_z, advection_speed_z,
        frequency_z, oscillation_amplitude_z)
    # create a mutable structure containing the input info related to the z grid
    z = grid_input_mutable("z", ngrid_z, nelement_z, L_z,
        discretization_option_z, finite_difference_option_z, boundary_option_z,
        advection_z)
    #################### parameters related to the r grid ######################
    # ngrid_r is number of grid points per element
    ngrid_r = 1
    # nelement_r is the number of elements
    nelement_r = 1
    # L_r is the box length
    L_r = 1.0
    # determine the boundary condition in r
    # currently supported options are "constant" and "periodic"
    boundary_option_r = "periodic"
    #boundary_option_r = "constant"
    # determine the discretization option for the r grid
    # supported options are "chebyshev_pseudospectral" and "finite_difference"
    #discretization_option_r = "chebyshev_pseudospectral"
    discretization_option_r = "finite_difference"
    # if discretization_option_r = "finite_difference", then
    # finite_difference_option_r determines the finite difference scheme to be used
    # supported options are "third_order_upwind", "second_order_upwind" and "first_order_upwind"
    #finite_difference_option_r = "first_order_upwind"
    #finite_difference_option_r = "second_order_upwind"
    finite_difference_option_r = "third_order_upwind"
    # determine the option used for the advection speed in r
    # supported options are "constant" and "oscillating",
    # in addition to the "default" option which uses dr/dt = vpa as the advection speed
    advection_option_r = "default" # MRH -- NEED TO CHANGE THIS ASAP!
    # constant advection speed in r to use with advection_option_r = "constant"
    advection_speed_r = 1.0
    # for advection_option_r = "oscillating", advection speed is of form
    # speed = advection_speed_r*(1 + oscillation_amplitude_r*sinpi(frequency_r*t))
    frequency_r = 1.0
    oscillation_amplitude_r = 1.0
    # mutable struct containing advection speed options/inputs for r
    advection_r = advection_input_mutable(advection_option_r, advection_speed_r,
        frequency_r, oscillation_amplitude_r)
    # create a mutable structure containing the input info related to the r grid
    r = grid_input_mutable("r", ngrid_r, nelement_r, L_r,
        discretization_option_r, finite_difference_option_r, boundary_option_r,
        advection_r)
    ############################################################################
    ################### parameters related to the vpa grid #####################
    # ngrid_vpa is the number of grid points per element
    ngrid_vpa = 300
    # nelement_vpa is the number of elements
    nelement_vpa = 1
    # L_vpa is the box length in units of vthermal_species
    L_vpa = 6.0
    # determine the boundary condition
    # currently supported options are "zero" and "periodic"
    #boundary_option_vpa = "zero"
    boundary_option_vpa = "periodic"
    # determine the discretization option for the vpa grid
    # supported options are "chebyshev_pseudospectral" and "finite_difference"
    #discretization_option_vpa = "chebyshev_pseudospectral"
    discretization_option_vpa = "finite_difference"
    # if discretization_option_vpa = "finite_difference", then
    # finite_difference_option_vpa determines the finite difference scheme to be used
    # supported options are "third_order_upwind", "second_order_upwind" and "first_order_upwind"
    #finite_difference_option_vpa = "second_order_upwind"
    finite_difference_option_vpa = "third_order_upwind"
    # determine the option used for the advection speed in vpa
    # supported options are "constant" and "oscillating",
    # in addition to the "default" option which uses dvpa/dt = q*Ez/m as the advection speed
    advection_option_vpa = "default"
    # constant advection speed in vpa to use with advection_option_vpa = "constant"
    advection_speed_vpa = 1.0
    # for advection_option_vpa = "oscillating", advection speed is of form
    # speed = advection_speed_vpa*(1 + oscillation_amplitude_vpa*sinpi(frequency_vpa*t))
    frequency_vpa = 1.0
    oscillation_amplitude_vpa = 1.0
    # mutable struct containing advection speed options/inputs for z
    advection_vpa = advection_input_mutable(advection_option_vpa, advection_speed_vpa,
        frequency_vpa, oscillation_amplitude_vpa)
    # create a mutable structure containing the input info related to the vpa grid
    vpa = grid_input_mutable("vpa", ngrid_vpa, nelement_vpa, L_vpa,
        discretization_option_vpa, finite_difference_option_vpa, boundary_option_vpa,
        advection_vpa)
    #############################################################################
    # define default values and create corresponding mutable structs holding
    # information about the composition of the species and their initial conditions
    if electron_physics ∈ (boltzmann_electron_response, boltzmann_electron_response_with_simple_sheath)
        n_species = n_ion_species + n_neutral_species
    else
        n_species = n_ion_speces + n_neutral_species + 1
    end
    # electron temperature over reference temperature
    T_e = 1.0
    # temperature at the entrance to the wall in terms of the electron temperature
    T_wall = 1.0
    # wall potential at z = 0
    phi_wall = 0.0
    # ratio of the neutral particle mass to the ion particle mass
    mn_over_mi = 1.0
    # ratio of the electron particle mass to the ion particle mass
    me_over_mi = 1.0/1836.0
    composition = species_composition(n_species, n_ion_species, n_neutral_species,
        electron_physics, 1:n_ion_species, n_ion_species+1:n_species, T_e, T_wall,
        phi_wall, mn_over_mi, me_over_mi, allocate_float(n_species))
    species = Array{species_parameters_mutable,1}(undef,n_species)
    # initial temperature for each species defaults to Tₑ
    initial_temperature = 1.0
    # initial density for each species defaults to Nₑ
    initial_density = 1.0
    # initialization inputs for z part of distribution function
    # supported options are "gaussian", "sinusoid" and "monomial"
    z_initialization_option = "sinusoid"
    # inputs for "gaussian" initial condition
    # width of the Gaussian in z
    z_width = 0.125
    # inputs for "sinusoid" initial condition
    # z_wavenumber should be an integer
    z_wavenumber = 1
    z_density_amplitude = 0.1
    z_density_phase = 0.0
    z_upar_amplitude = 0.0
    z_upar_phase = 0.0
    z_temperature_amplitude = 0.0
    z_temperature_phase = 0.0
    # inputs for "monomial" initial condition
    z_monomial_degree = 2
    z_initial_conditions = initial_condition_input_mutable(z_initialization_option,
        z_width, z_wavenumber, z_density_amplitude, z_density_phase, z_upar_amplitude,
        z_upar_phase, z_temperature_amplitude, z_temperature_phase, z_monomial_degree)
    # initialization inputs for vpa part of distribution function
    # supported options are "gaussian", "sinusoid" and "monomial"
    # inputs for 'gaussian' initial condition
    vpa_initialization_option = "gaussian"
    # if initializing a Maxwellian, vpa_width = 1.0 for each species
    # any temperature-dependence will be self-consistently treated using initial_temperature
    vpa_width = 1.0
    # inputs for "sinusoid" initial condition
    vpa_wavenumber = 1
    vpa_density_amplitude = 1.0
    vpa_density_phase = 0.0
    vpa_upar_amplitude = 0.0
    vpa_upar_phase = 0.0
    vpa_temperature_amplitude = 0.0
    vpa_temperature_phase = 0.0
    # inputs for "monomial" initial condition
    vpa_monomial_degree = 2
    vpa_initial_conditions = initial_condition_input_mutable(vpa_initialization_option,
        vpa_width, vpa_wavenumber, vpa_density_amplitude, vpa_density_phase,
        vpa_upar_amplitude, vpa_upar_phase, vpa_temperature_amplitude,
        vpa_temperature_phase, vpa_monomial_degree)

    # fill in entries in species struct corresponding to ion species
    for is ∈ 1:n_ion_species
        species[is] = species_parameters_mutable("ion", initial_temperature, initial_density,
            deepcopy(z_initial_conditions), deepcopy(vpa_initial_conditions))
    end
    # if there are neutrals, fill in corresponding entries in species struct
    if n_neutral_species > 0
        for is ∈ 1:n_neutral_species
            species[n_ion_species + is] = species_parameters_mutable("neutral", initial_temperature,
                initial_density, deepcopy(z_initial_conditions), deepcopy(vpa_initial_conditions))
        end
    end
    # if drive_phi = true, include external electrostatic potential of form
    # phi(z,t=0)*drive_amplitude*sinpi(time*drive_frequency)
    drive_phi = false
    drive_amplitude = 1.0
    drive_frequency = 1.0
    drive = drive_input_mutable(drive_phi, drive_amplitude, drive_frequency)
    # charge exchange collision frequency
    charge_exchange = 0.0
    # ionization collision frequency
    ionization = 0.0
    constant_ionization_rate = false
    collisions = collisions_input(charge_exchange, ionization, constant_ionization_rate)

    return z, r, vpa, species, composition, drive, evolve_moments, collisions
end

"""
check various input options to ensure they are all valid/consistent
"""
function check_input(io, output_dir, nstep, dt, use_semi_lagrange, r, z, vpa,
    composition, species, evolve_moments)
    # copy the input file to the output directory to be saved
    if block_rank[] == 0
        cp(joinpath(@__DIR__, "moment_kinetics_input.jl"), joinpath(output_dir, "moment_kinetics_input.jl"), force=true)
    end
    # open ascii file in which informtaion about input choices will be written
    check_input_time_advance(nstep, dt, use_semi_lagrange, io)
    check_coordinate_input(r, "r", io)
    check_coordinate_input(z, "z", io)
    check_coordinate_input(vpa, "vpa", io)
    check_input_initialization(composition, species, io)
    # if the parallel flow is evolved separately, then the density must also be evolved separately
    if evolve_moments.parallel_flow && !evolve_moments.density
        print(io,">evolve_moments.parallel_flow = true, but evolve_moments.density = false.")
        println(io, "this is not a supported option.  forcing evolve_moments.density = true.")
        evolve_moments.density = true
    end
end

"""
"""
function check_input_time_advance(nstep, dt, use_semi_lagrange, io)
    println(io,"##### time advance #####")
    println(io)
    # use_semi_lagrange = true to use interpolation-free semi-Lagrange treatment
    # otherwise, solve problem solely using the discretization_option above
    if use_semi_lagrange
        print(io,">use_semi_lagrange set to true.  ")
        println(io,"using interpolation-free semi-Lagrange for advection terms.")
    end
    println(io,">running for ", nstep, " time steps, with step size ", dt, ".")
end

"""
Check input for a coordinate
"""
function check_coordinate_input(coord, coord_name, io)
    if coord.ngrid * coord.nelement == 1
        # Coordinate is not being used for this run
        return
    end

    println(io)
    println(io,"######## $coord_name-grid ########")
    println(io)
    # discretization_option determines discretization in coord
    # supported options are chebyshev_pseudospectral and finite_difference
    if coord.discretization == "chebyshev_pseudospectral"
        print(io,">$coord_name.discretization = 'chebyshev_pseudospectral'.  ")
        println(io,"using a Chebyshev pseudospectral method in $coord_name.")
    elseif coord.discretization == "finite_difference"
        println(io,">$coord_name.discretization = 'finite_difference', ",
            "and $coord_name.fd_option = ", coord.fd_option,
            "  using finite differences on an equally spaced grid in $coord_name.")
        fd_check_option(coord.fd_option, coord.ngrid)
    else
        input_option_error("$coord_name.discretization", coord.discretization)
    end
    # boundary_option determines coord boundary condition
    # supported options are "constant" and "periodic"
    if coord.bc == "constant"
        println(io,">$coord_name.bc = 'constant'.  enforcing constant incoming BC in $coord_name.")
    elseif coord.bc == "zero"
        println(io,">$coord_name.bc = 'zero'.  enforcing zero incoming BC in $coord_name.")
    elseif coord.bc == "both_zero"
        println(io,">$coord_name.bc = 'both_zero'.  enforcing zero BC in $coord_name.")
    elseif coord.bc == "periodic"
        println(io,">$coord_name.bc = 'periodic'.  enforcing periodicity in $coord_name.")
    elseif coord_name == "z" && coord.bc == "wall"
        println(io,">$coord_name.bc = 'wall'.  enforcing wall BC in $coord_name.")
    else
        input_option_error("$coord_name.bc", coord.bc)
    end
    println(io,">using ", coord.ngrid, " grid points per $coord_name element on ", coord.nelement,
        " elements across the $coord_name domain [", -0.5*coord.L, ",", 0.5*coord.L, "].")
end

"""
"""
function check_input_initialization(composition, species, io)
    println(io)
    println(io,"####### initialization #######")
    println(io)
    # xx_initialization_option determines the initial condition for coordinate xx
    # currently supported options are "gaussian" and "monomial"
    n_ion_species = composition.n_ion_species
    n_neutral_species = composition.n_neutral_species
    for is ∈ 1:composition.n_species
        if is <= n_ion_species
            print(io,">initial distribution function for ion species ", is)
        elseif is <= n_ion_species + n_neutral_species
            print(io,">initial distribution function for neutral species ", is-n_ion_species)
        else
            print(io,">initial distribution function for the electrons")
        end
        println(io," is of the form f(z,vpa,t=0)=F(z)*G(vpa).")
        if species[is].z_IC.initialization_option == "gaussian"
            print(io,">z intialization_option = 'gaussian'.")
            println(io,"  setting F(z) = initial_density + exp(-(z/z_width)^2).")
        elseif species[is].z_IC.initialization_option == "monomial"
            print(io,">z_intialization_option = 'monomial'.")
            println(io,"  setting F(z) = (z + L_z/2)^", species[is].z_IC.monomial_degree, ".")
        elseif species[is].z_IC.initialization_option == "sinusoid"
            print(io,">z_initialization_option = 'sinusoid'.")
            println(io,"  setting F(z) = initial_density + z_amplitude*sinpi(z_wavenumber*z/L_z).")
        elseif species[is].z_IC.initialization_option == "bgk"
            print(io,">z_initialization_option = 'bgk'.")
            println(io,"  setting F(z,vpa) = F(vpa^2 + phi), with phi_max = 0.")
        else
            input_option_error("z_initialization_option", species[is].z_IC.initialization_option)
        end
        if species[is].vpa_IC.initialization_option == "gaussian"
            print(io,">vpa_intialization_option = 'gaussian'.")
            println(io,"  setting G(vpa) = exp(-(vpa/vpa_width)^2).")
        elseif species[is].vpa_IC.initialization_option == "monomial"
            print(io,">vpa_intialization_option = 'monomial'.")
            println(io,"  setting G(vpa) = (vpa + L_vpa/2)^", species[is].vpa_IC._monomial_degree, ".")
        elseif species[is].vpa_IC.initialization_option == "sinusoid"
            print(io,">vpa_initialization_option = 'sinusoid'.")
            println(io,"  setting G(vpa) = vpa_amplitude*sinpi(vpa_wavenumber*vpa/L_vpa).")
        elseif species[is].vpa_IC.initialization_option == "bgk"
            print(io,">vpa_initialization_option = 'bgk'.")
            println(io,"  setting F(z,vpa) = F(vpa^2 + phi), with phi_max = 0.")
        elseif species[is].vpa_IC.initialization_option == "vpagaussian"
            print(io,">vpa_initialization_option = 'vpagaussian'.")
            println(io,"  setting G(vpa) = vpa^2*exp(-(vpa/vpa_width)^2).")
        else
            input_option_error("vpa_initialization_option", species[is].vpa_IC.initialization_option)
        end
        println(io)
    end
end

end
