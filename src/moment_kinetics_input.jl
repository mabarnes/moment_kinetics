@enum single performance_test scan
const run_type = single

function mk_input(scan_input=Dict())

    # n_ion_species is the number of evolved ion species
    # currently only n_ion_species = 1 is supported
    n_ion_species = 1
    # n_neutral_species is the number of evolved neutral species
    # currently only n_neutral_species = 0 is supported
    n_neutral_species = 1
    # if boltzmann_electron_response = true, then the electron
    # density is fixed to be N_e*(eϕ/T_e)
    # currently this is the only supported option
    boltzmann_electron_response = true

    z, vpa, species, composition, drive, evolve_moments =
        load_defaults(n_ion_species, n_neutral_species, boltzmann_electron_response)

    # this is the prefix for all output files associated with this run
    run_name = get(scan_input, :run_name, "ppar")
    # this is the directory where the simulation data will be stored
    output_dir = string("runs/",run_name)
    # if evolve_moments.density = true, evolve density via continuity eqn
    # and g = f/n via modified drift kinetic equation
    evolve_moments.density = true
    evolve_moments.parallel_flow = true
    evolve_moments.parallel_pressure = true
    evolve_moments.conservation = true
#    evolve_moments.advective_form = false

    #z.advection.option = "constant"
    #z.advection.constant_speed = 1.0

    #vpa.advection.option = "constant"
    #vpa.advection.constant_speed = 0.0

    ####### specify any deviations from default inputs for evolved species #######
    # set initial Tₑ = 1
    composition.T_e = get(scan_input, :T_e, 1.0)
    # set initial neutral temperature Tn/Tₑ = 1
    # set initial nᵢ/Nₑ = 1.0
    species[1].initial_density = get(scan_input, (:initial_density, 1), 0.5)
    species[1].initial_temperature = get(scan_input, (:initial_temperature, 1), 1.0)
    species[1].z_IC.amplitude = get(scan_input, (:z_IC_amplitude, 1), 0.001)
    #species[1].z_IC.initialization_option = "bgk"
    # set initial neutral densiity = Nₑ
    if composition.n_species > 1
        species[2].initial_density = get(scan_input, (:initial_density, 2), 0.5)
        species[2].initial_temperature = get(scan_input, (:initial_temperature, 2), species[1].initial_temperature)
        species[2].z_IC.amplitude = get(scan_input, (:z_IC_amplitude, 2), species[1].z_IC.amplitude)
    end
    #################### end specification of species inputs #####################

    charge_exchange_frequency = get(scan_input, :charge_exchange_frequency, 2.0*sqrt(species[1].initial_temperature))

    # parameters related to the time stepping
    nstep = get(scan_input, :nstep, 5000)
    dt = get(scan_input, :dt, 0.001/sqrt(species[1].initial_temperature))
    nwrite = get(scan_input, :nwrite, 10)
    # use_semi_lagrange = true to use interpolation-free semi-Lagrange treatment
    # otherwise, solve problem solely using the discretization_option above
    use_semi_lagrange = false
    # options are n_rk_stages = 1, 2, 3 or 4 (corresponding to forward Euler,
    # Heun's method, SSP RK3 and 4-stage SSP RK3)
    n_rk_stages = 4
    split_operators = false

    # overwrite some default parameters related to the z grid
    # ngrid is number of grid points per element
    z.ngrid = 9
    # nelement is the number of elements
    z.nelement = 2
    #z.ngrid = 400
    #z.nelement = 1
    # determine the discretization option for the z grid
    # supported options are "chebyshev_pseudospectral" and "finite_difference"
    z.discretization = "chebyshev_pseudospectral"
    #z.discretization = "finite_difference"

    # overwrite some default parameters related to the vpa grid
    # ngrid is the number of grid points per element
    vpa.ngrid = 17
    # nelement is the number of elements
    vpa.nelement = 10
    #vpa.ngrid = 400
    #vpa.nelement = 1
    # L is the box length in units of vthermal_species
    vpa.L = 8.0*sqrt(species[1].initial_temperature)
    # determine the boundary condition
    # only supported option at present is "zero" and "periodic"
    vpa.bc = "periodic"
    #vpa.bc = "zero"
    # determine the discretization option for the vpa grid
    # supported options are "chebyshev_pseudospectral" and "finite_difference"
    vpa.discretization = "chebyshev_pseudospectral"
    #vpa.discretization = "finite_difference"

    #########################################################################
    ########## end user inputs. do not modify following code! ###############
    #########################################################################

    t = time_input(nstep, dt, nwrite, use_semi_lagrange, n_rk_stages, split_operators)
    # replace mutable structures with immutable ones to optimize performance
    # and avoid possible misunderstandings
    z_advection_immutable = advection_input(z.advection.option, z.advection.constant_speed,
        z.advection.frequency, z.advection.oscillation_amplitude)
    z_immutable = grid_input("z", z.ngrid, z.nelement, z.L,
        z.discretization, z.fd_option, z.bc, z_advection_immutable)
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
            species[is].z_IC.amplitude, species[is].z_IC.monomial_degree)
        vpa_IC = initial_condition_input(species[is].vpa_IC.initialization_option,
            species[is].vpa_IC.width, species[is].vpa_IC.wavenumber,
            species[is].vpa_IC.amplitude, species[is].vpa_IC.monomial_degree)
        species_immutable[is] = species_parameters(species_type, species[is].initial_temperature,
            species[is].initial_density, z_IC, vpa_IC)
    end
    drive_immutable = drive_input(drive.force_phi, drive.amplitude, drive.frequency)

    # check input to catch errors/unsupported options
    check_input(run_name, output_dir, nstep, dt, use_semi_lagrange,
        z_immutable, vpa_immutable, composition, species_immutable, evolve_moments)

    # return immutable structs for z, vpa, species and composition
    return run_name, output_dir, evolve_moments, t, z_immutable, vpa_immutable,
        composition, species_immutable, charge_exchange_frequency, drive_immutable
end

function load_defaults(n_ion_species, n_neutral_species, boltzmann_electron_response)
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
    if boltzmann_electron_response
        n_species = n_ion_species + n_neutral_species
    else
        n_species = n_ion_speces + n_neutral_species + 1
    end
    composition = species_composition(n_species, n_ion_species, n_neutral_species,
        boltzmann_electron_response, 1.0)
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
    z_amplitude = 0.1
    # inputs for "monomial" initial condition
    z_monomial_degree = 2
    z_initial_conditions = initial_condition_input_mutable(z_initialization_option,
        z_width, z_wavenumber, z_amplitude, z_monomial_degree)
    # initialization inputs for vpa part of distribution function
    # supported options are "gaussian", "sinusoid" and "monomial"
    # inputs for 'gaussian' initial condition
    vpa_initialization_option = "gaussian"
    # if initializing a Maxwellian, vpa_width = 1.0 for each species
    # any temperature-dependence will be self-consistently treated using initial_temperature
    vpa_width = 1.0
    # inputs for "sinusoid" initial condition
    vpa_wavenumber = 1
    vpa_amplitude = 1.0
    # inputs for "monomial" initial condition
    vpa_monomial_degree = 2
    vpa_initial_conditions = initial_condition_input_mutable(vpa_initialization_option,
        vpa_width, vpa_wavenumber, vpa_amplitude, vpa_monomial_degree)

    # fill in entries in species struct corresponding to ion species
    for is ∈ 1:n_ion_species
        species[is] = species_parameters_mutable("ion", initial_temperature, initial_density,
            z_initial_conditions, vpa_initial_conditions)
    end
    # if there are neutrals, fill in corresponding entries in species struct
    if n_neutral_species > 0
        for is ∈ 1:n_neutral_species
            species[n_ion_species + is] = species_parameters_mutable("neutral", initial_temperature,
                initial_density, z_initial_conditions, vpa_initial_conditions)
        end
    end
    # if drive_phi = true, include external electrostatic potential of form
    # phi(z,t=0)*drive_amplitude*sinpi(time*drive_frequency)
    drive_phi = false
    drive_amplitude = 1.0
    drive_frequency = 1.0
    drive = drive_input_mutable(drive_phi, drive_amplitude, drive_frequency)
    return z, vpa, species, composition, drive, evolve_moments
end

# check various input options to ensure they are all valid/consistent
function check_input(run_name, output_dir, nstep, dt, use_semi_lagrange, z, vpa,
    composition, species, evolve_moments)
    # check to see if output_dir exists in the current directory
    # if not, create it
    isdir(output_dir) || mkdir(output_dir)
    # copy the input file to the output directory to be saved
    cp("src/moment_kinetics_input.jl", string(output_dir,"/moment_kinetics_input.jl"), force=true)
    # open ascii file in which informtaion about input choices will be written
    io = open_output_file(string(output_dir,"/",run_name), "input")
    check_input_time_advance(nstep, dt, use_semi_lagrange, io)
    check_input_z(z, io)
    check_input_vpa(vpa, io)
    check_input_initialization(composition, species, io)
    # if the parallel flow is evolved separately, then the density must also be evolved separately
    if evolve_moments.parallel_flow && !evolve_moments.density
        print(io,">evolve_moments.parallel_flow = true, but evolve_moments.density = false.")
        println(io, "this is not a supported option.  forcing evolve_moments.density = true.")
        evolve_moments.density = true
    end
    close(io)
end
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
function check_input_z(z, io)
    println(io)
    println(io,"######## z-grid ########")
    println(io)
    # discretization_option determines discretization in z
    # supported options are chebyshev_pseudospectral and finite_difference
    if z.discretization == "chebyshev_pseudospectral"
        print(io,">z.discretization = 'chebyshev_pseudospectral'.  ")
        println(io,"using a Chebyshev pseudospectral method in z.")
    elseif z.discretization == "finite_difference"
        print(io,">z.discretization = 'finite_difference', ",
            "and z.fd_option = ")
        if z.fd_option == "third_order_upwind"
            print(io,"'third_order_upwind'.")
        elseif z.fd_option == "second_order_upwind"
            print(io,"'second_order_upwind'.")
        elseif z.fd_option == "first_order_upwind"
            print(io,"'first_order_upwind'.")
        else
            input_option_error("z.fd_option", z.fd_option)
        end
        println(io,"  using finite differences on an equally spaced grid in z.")
    else
        input_option_error("z.discretization", z.discretization)
    end
    # boundary_option determines z boundary condition
    # supported options are "constant" and "periodic"
    if z.bc == "constant"
        println(io,">z.bc = 'constant'.  enforcing constant incoming BC in z.")
    elseif z.bc == "periodic"
        println(io,">z.bc = 'periodic'.  enforcing periodicity in z.")
    else
        input_option_error("z.bc", z.bc)
    end
    println(io,">using ", z.ngrid, " grid points per z element on ", z.nelement,
        " elements across the z domain [", -0.5*z.L, ",", 0.5*z.L, "].")
end
function check_input_vpa(vpa, io)
    println(io)
    println(io,"######## vpa-grid ########")
    println(io)
    # discretization_option determines discretization in vpa
    # supported options are chebyshev_pseudospectral and finite_difference
    if vpa.discretization == "chebyshev_pseudospectral"
        print(io,">vpa.discretization = 'chebyshev_pseudospectral'.  ")
        println(io,"using a Chebyshev pseudospectral method in vpa.")
    elseif vpa.discretization == "finite_difference"
        print(io,">vpa.discretization = 'finite_difference', and ",
            "vpa.fd_option = ")
        if vpa.fd_option == "third_order_upwind"
            print(io,"'third_order_upwind'.")
            if (vpa.ngrid < 4)
                println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                println("ERROR: vpa.ngrid < 4 incompatible with 3rd order upwind differences.  Aborting.")
                println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                exit(1)
            end
        elseif vpa.fd_option == "second_order_upwind"
            print(io,"'second_order_upwind'.")
        elseif vpa.fd_option == "first_order_upwind"
            print(io,"'first_order_upwind'.")
        else
            input_option_error("vpa.fd_option", vpa.fd_option)
        end
        println(io,"  using finite differences on an equally spaced grid in vpa.")
    else
        input_option_error("vpa.discretization", vpa.discretization)
    end
    # boundary_option determines vpa boundary condition
    # supported options are "zero" and "periodic"
    if vpa.bc == "zero"
        println(io,">vpa.bc = 'zero'.  enforcing zero incoming BC in vpa.")
    elseif vpa.bc == "periodic"
        println(io,">vpa.bc = 'periodic'.  enforcing periodicity in vpa.")
    else
        input_option_error("vpa.bc", vpa.bc)
    end
    println(io,">using ", vpa.ngrid, " grid points per vpa element on ", vpa.nelement,
        " elements across the vpa domain [", -0.5*vpa.L, ",", 0.5*vpa.L, "].")
end
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
        else
            input_option_error("vpa_initialization_option", species[is].vpa_IC.initialization_option)
        end
        println(io)
    end
end
