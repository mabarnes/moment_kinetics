module moment_kinetics_input

import file_io: input_option_error, open_output_file

export run_name
export z_input
export vpa_input
export nstep, dt, nwrite, use_semi_lagrange
export advection_speed, advection_speed_option
export initialization_option, zwidth, monomial_degree
export boltzmann_electron_response
export check_input
export performance_test

struct grid_input
    # name of the variable associated with this coordinate
    name::String
    # number of grid points per element
    ngrid::Int64
    # number of elements
    nelement::Int64
    # box length
    L::Float64
    # discretization option
    discretization::String
    # boundary option
    bc::String
end

# this is the prefix for all output files associated with this run
const run_name = "example"

# parameters related to the time stepping
const nstep = 500
const dt = 0.001
const nwrite = 500
# use_semi_lagrange = true to use interpolation-free semi-Lagrange treatment
# otherwise, solve problem solely using the discretization_option above
const use_semi_lagrange = true

# parameters related to the z grid
# ngrid_z is number of grid points per element
const ngrid_z = 33
# nelement_z is the number of elements
const nelement_z = 3
# L_z is the box length
const L_z = 2.
# determine the boundary condition
# currently supported option is "zero"
const boundary_option_z = "zero"
# determine the discretization option for the z grid
# supported options are "chebyshev_pseudospectral" and "finite_difference"
const discretization_option_z = "chebyshev_pseudospectral"
#const discretization_option_z= "finite_difference"

# parameters related to the vpa grid
# ngrid_vpa is the number of grid points per element
const ngrid_vpa = 33
# nelement_vpa is the number of elements
const nelement_vpa = 3
# L_vpa is the box length in units of vthermal_species
const L_vpa = 6.
# determine the boundary condition
# only supported option at present is "zero"
const boundary_option_vpa = "zero"
# determine the discretization option for the vpa grid
# supported options are "chebyshev_pseudospectral" and "finite_difference"
const discretization_option_vpa = "chebyshev_pseudospectral"
#const discretization_option_vpa = "finite_difference"

# advection speed
const advection_speed = 1.0
const advection_speed_option = "constant"

# determines which function is being advected
const initialization_option = "gaussian"
const zwidth = 0.1
const vpawidth = 0.5
const monomial_degree = 2

# if boltzmann_electron_response = true, then the electron
# density is fixed to be n₀(eϕ/T)
const boltzmann_electron_response = true

# performance_test = true returns timings and memory usage
const performance_test = false

z_input = grid_input("z", ngrid_z, nelement_z, L_z,
    discretization_option_z, boundary_option_z)
vpa_input = grid_input("vpa", ngrid_vpa, nelement_vpa, L_vpa,
    discretization_option_vpa, boundary_option_vpa)

# check various input options to ensure they are all valid/consistent
function check_input()
    io = open_output_file(run_name, "input")
    check_input_time_advance(io)
    check_input_z(io)
    check_input_vpa(io)
    check_input_initialization(io)
    close(io)
end
function check_input_time_advance(io)
    println(io,"##### time advance #####")
    # use_semi_lagrange = true to use interpolation-free semi-Lagrange treatment
    # otherwise, solve problem solely using the discretization_option above
    if use_semi_lagrange
        print(io,">use_semi_lagrange set to true.  ")
        println(io,"using interpolation-free semi-Lagrange for advection terms.")
    end
    println(io,">running for ", nstep, " time steps, with step size ", dt, ".")
end
function check_input_z(io)
    println(io,"######## z-grid ########")
    # discretization_option determines discretization in z
    # supported options are chebyshev_pseudospectral and finite_difference
    if discretization_option_z == "chebyshev_pseudospectral"
        print(io,">discretization_option_z = 'chebyshev_pseudospectral'.  ")
        println(io,"using a Chebyshev pseudospectral method in z.")
    elseif discretization_option_z == "finite_difference"
        print(io,">discretization_option_z = 'finite_difference'.  ")
        println(io,"using finite differences on an equally space grid in z.")
    else
        input_option_error("discretization_option_z", discretization_option_z)
    end
    # boundary_option determines z boundary condition
    # supported options are "zero" and "periodic"
    if boundary_option_z == "zero"
        println(io,">boundary_option_z = 'zero'.  enforcing zero incoming BC in z.")
    elseif boundary_option == "periodic"
        println(io,">boundary_option_z = 'periodic'.  enforcing periodicity in z.")
    else
        input_option_error("boundary_option_z", boundary_option_z)
    end
    println(io,">using ", ngrid_z, " grid points per z element on ", nelement_z,
        " elements across the z domain [", -0.5*L_z, ",", 0.5*L_z, "].")
end
function check_input_vpa(io)
    println(io,"######## vpa-grid ########")
    # discretization_option determines discretization in vpa
    # supported options are chebyshev_pseudospectral and finite_difference
    if discretization_option_vpa == "chebyshev_pseudospectral"
        print(io,">discretization_option_vpa = 'chebyshev_pseudospectral'.  ")
        println(io,"using a Chebyshev pseudospectral method in vpa.")
    elseif discretization_option_vpa == "finite_difference"
        print(io,">discretization_option_vpa = 'finite_difference'.  ")
        println(io,"using finite differences on an equally space grid in vpa.")
    else
        input_option_error("discretization_option_vpa", discretization_option_vpa)
    end
    # boundary_option determines z boundary condition
    # supported options are "zero" and "periodic"
    if boundary_option_vpa == "zero"
        println(io,">boundary_option_vpa = 'zero'.  enforcing zero incoming BC in vpa.")
    else
        input_option_error("boundary_option_vpa", boundary_option_vpa)
    end
    println(io,">using ", ngrid_vpa, " grid points per vpa element on ", nelement_vpa,
        " elements across the vpa domain [", -0.5*L_vpa, ",", 0.5*L_vpa, "].")
end
function check_input_initialization(io)
    println(io,"####### initialization #######")
    # initialization_option determines the initial condition
    # supported options are "gaussian" and "linear"
    if initialization_option == "gaussian"
        println(io,">intialization_option = 'gaussian'.  setting f(z,t=0) = exp(-(z/zwidth)^2).")
    elseif initialization_option == "monomial"
        print(io,">intialization_option = 'monomial'.")
        println(io,"  setting f(z,t=0) = (z + L_z/2)^", monomial_degree, ".")
    else
        input_option_error("initialization_option", initialization_option)
    end
end

end
