module moment_kinetics_input

export mk_input
export advection_speed, advection_speed_option_z, advection_speed_option_vpa
export z_adv_oscillation_amplitude, z_adv_frequency
export performance_test

using type_definitions: mk_float, mk_int
using array_allocation: allocate_float
using file_io: input_option_error, open_output_file
using input_structs: time_input
using input_structs: grid_input, grid_input_mutable
using input_structs: initial_condition_input, initial_condition_input_mutable
using input_structs: species_parameters, species_parameters_mutable
using input_structs: species_composition

const advection_speed_option_z = "default"
const advection_speed_option_vpa = "default"
# advection speed
const advection_speed = -1.0
# for advection_speed_option = "oscillating", advection speed is of form
# speed = advection_speed*(1 + z_adv_oscillation_amplitude*sinpi(z_adv_frequency*t))
const z_adv_frequency = 1.0
const z_adv_oscillation_amplitude = 1.0

const performance_test = false

function mk_input()

    # n_ion_species is the number of evolved ion species
    # currently only n_ion_species = 1 is supported
    n_ion_species = 1
    # n_neutral_species is the number of evolved neutral species
    # currently only n_neutral_species = 0 is supported
    n_neutral_species = 0
    # if boltzmann_electron_response = true, then the electron
    # density is fixed to be N_e*(eϕ/T_e)
    # currently this is the only supported option
    boltzmann_electron_response = true

    z, vpa, species, composition =
        load_defaults(n_ion_species, n_neutral_species, boltzmann_electron_response)

    # this is the prefix for all output files associated with this run
    run_name = "example"
    # this is the directory where the simulation data will be stored
    output_dir = run_name

    # parameters related to the time stepping
    nstep = 3000
    dt = 0.001
    nwrite = 10
    # use_semi_lagrange = true to use interpolation-free semi-Lagrange treatment
    # otherwise, solve problem solely using the discretization_option above
    use_semi_lagrange = false

    # overwrite some default parameters related to the z grid
    # ngrid is number of grid points per element
    z.ngrid = 50
    # nelement is the number of elements
    z.nelement = 1
    # determine the discretization option for the z grid
    # supported options are "chebyshev_pseudospectral" and "finite_difference"
    z.discretization = "finite_difference"

    # overwrite some default parameters related to the vpa grid
    # ngrid is the number of grid points per element
    vpa.ngrid = 300
    # nelement is the number of elements
    vpa.nelement = 1
    # L is the box length in units of vthermal_species
    vpa.L = 6.0
    # determine the boundary condition
    # only supported option at present is "zero" and "periodic"
    vpa.bc = "periodic"
    # determine the discretization option for the vpa grid
    # supported options are "chebyshev_pseudospectral" and "finite_difference"
    vpa.discretization = "finite_difference"

    ####### specify any deviations from default inputs for evolved species #######
    # set initial Tᵢ/Tₑ = 1
    species[1].initial_temperature = 1.0
    #species[2].initial_temperature = 1.0
    # set initial nᵢ/Nₑ = 1.0
    species[1].initial_density = 1.0
    # set initial neutral densiity = Nₑ
    #species[2].initial_density = 0.5
    #################### end specification of species inputs #####################
#=
    const advection_speed_option_z = "default"
    const advection_speed_option_vpa = "default"
    # advection speed
    const advection_speed = -1.0
    # for advection_speed_option = "oscillating", advection speed is of form
    # speed = advection_speed*(1 + z_adv_oscillation_amplitude*sinpi(z_adv_frequency*t))
    const z_adv_frequency = 1.0
    const z_adv_oscillation_amplitude = 1.0

    # performance_test = true returns timings and memory usage
    performance_test = false
=#
    #########################################################################
    ########## end user inputs. do not modify following code! ###############
    #########################################################################

    t = time_input(nstep, dt, nwrite, use_semi_lagrange)
    # replace mutable structures with immutable ones to optimize performance
    # and avoid possible misunderstandings
    z_immutable = grid_input("z", z.ngrid, z.nelement, z.L,
        z.discretization, z.fd_option, z.bc)
    vpa_immutable = grid_input("vpa", vpa.ngrid, vpa.nelement, vpa.L,
        vpa.discretization, vpa.fd_option, vpa.bc)
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

    # check input to catch errors/unsupported options
    check_input(run_name, output_dir, nstep, dt, use_semi_lagrange,
        z_immutable, vpa_immutable, composition, species_immutable)

    # return immutable structs for z, vpa, species and composition
    return run_name, output_dir, t, z_immutable, vpa_immutable, composition, species_immutable
end

function load_defaults(n_ion_species, n_neutral_species, boltzmann_electron_response)
    # parameters related to the z grid
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
    # create a mutable structure containing the input info related to the z grid
    z = grid_input_mutable("z", ngrid_z, nelement_z, L_z,
        discretization_option_z, finite_difference_option_z, boundary_option_z)
    # parameters related to the vpa grid
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
    # create a mutable structure containing the input info related to the vpa grid
    vpa = grid_input_mutable("vpa", ngrid_vpa, nelement_vpa, L_vpa,
        discretization_option_vpa, finite_difference_option_vpa, boundary_option_vpa)
    # define default values and create corresponding mutable structs holding
    # information about the composition of the species and their initial conditions
    if boltzmann_electron_response
        n_species = n_ion_species + n_neutral_species
    else
        n_species = n_ion_speces + n_neutral_species + 1
    end
    composition = species_composition(n_species, n_ion_species, n_neutral_species,
        boltzmann_electron_response)
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
    return z, vpa, species, composition
end

# check various input options to ensure they are all valid/consistent
function check_input(run_name, output_dir, nstep, dt, use_semi_lagrange, z, vpa,
    composition, species)
    # check to see if output_dir exists in the current directory
    # if not, create it
    isdir(output_dir) || mkdir(output_dir)
    # copy the input file to the output directory to be saved
    cp("moment_kinetics_input.jl", string(output_dir,"/moment_kinetics_input.jl"), force=true)
    # open ascii file in which informtaion about input choices will be written
    io = open_output_file(string(output_dir,"/",run_name), "input")
    check_input_time_advance(nstep, dt, use_semi_lagrange, io)
    check_input_z(z, io)
    check_input_vpa(vpa, io)
    check_input_initialization(composition, species, io)
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
        else
            input_option_error("vpa_initialization_option", species[is].vpa_IC.initialization_option)
        end
        println(io)
    end
end

end
