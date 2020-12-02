module moment_kinetics_input

export run_name, output_dir
export z_input
export vpa_input
export nstep, dt, nwrite, use_semi_lagrange
export advection_speed, advection_speed_option
export initialization_option, zwidth, monomial_degree
export boltzmann_electron_response
export check_input
export performance_test

using type_definitions: mk_float, mk_int
using file_io: input_option_error, open_output_file

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
end

# this is the prefix for all output files associated with this run
const run_name = "fd_noSL_1D_n200_v0p5_dt0p01_test"
# this is the directory where the simulation data will be stored
const output_dir = run_name

# parameters related to the time stepping
const nstep = 4000
const dt = 0.005
const nwrite = 100
# use_semi_lagrange = true to use interpolation-free semi-Lagrange treatment
# otherwise, solve problem solely using the discretization_option above
const use_semi_lagrange = false

# parameters related to the z grid
# ngrid_z is number of grid points per element
const ngrid_z = 200
# nelement_z is the number of elements
const nelement_z = 1
# L_z is the box length
const L_z = 2.
# determine the boundary condition
# currently supported options are "constant" and "periodic"
const boundary_option_z = "periodic"
#const boundary_option_z = "constant"
# determine the discretization option for the z grid
# supported options are "chebyshev_pseudospectral" and "finite_difference"
#const discretization_option_z = "chebyshev_pseudospectral"
const discretization_option_z= "finite_difference"
# if discretization_option_z = "finite_difference", then
# finite_difference_option_z determines the finite difference scheme to be used
# supported options are "second_order_upwind" and "first_order_upwind"
const finite_difference_option_z = "first_order_upwind"

# parameters related to the vpa grid
# ngrid_vpa is the number of grid points per element
const ngrid_vpa = 3
# nelement_vpa is the number of elements
const nelement_vpa = 1
# L_vpa is the box length in units of vthermal_species
const L_vpa = 2.
# determine the boundary condition
# only supported option at present is "zero"
#const boundary_option_vpa = "zero"
const boundary_option_vpa = "periodic"
# determine the discretization option for the vpa grid
# supported options are "chebyshev_pseudospectral" and "finite_difference"
const discretization_option_vpa = "chebyshev_pseudospectral"
#const discretization_option_vpa = "finite_difference"
# if discretization_option_vpa = "finite_difference", then
# finite_difference_option_vpa determines the finite difference scheme to be used
# supported options are "second_order_upwind" and "first_order_upwind"
const finite_difference_option_vpa = "second_order_upwind"

# advection speed
const advection_speed = 0.5
const advection_speed_option = "constant"

# determines which function is being advected
const initialization_option = "gaussian"
const zwidth = 0.25
const vpawidth = 1.0
const monomial_degree = 2
const density_offset = 1.0

# if boltzmann_electron_response = true, then the electron
# density is fixed to be n₀(eϕ/T)
const boltzmann_electron_response = true

# performance_test = true returns timings and memory usage
const performance_test = false

z_input = grid_input("z", ngrid_z, nelement_z, L_z,
    discretization_option_z, finite_difference_option_z, boundary_option_z)
vpa_input = grid_input("vpa", ngrid_vpa, nelement_vpa, L_vpa,
    discretization_option_vpa, finite_difference_option_vpa, boundary_option_vpa)

# check various input options to ensure they are all valid/consistent
function check_input()
    # check to see if output_dir exists in the current directory
    # if not, create it
    isdir(output_dir) || mkdir(output_dir)
    # copy the input file to the output directory to be saved
    cp("moment_kinetics_input.jl", string(output_dir,"/moment_kinetics_input.jl"), force=true)
    # open ascii file in which informtaion about input choices will be written
    io = open_output_file(string(output_dir,"/",run_name), "input")
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
        print(io,">discretization_option_z = 'finite_difference',
            and finite_difference_option_z = ")
        if finite_difference_option_z == "second_order_upwind"
            print(io,"'second_order_upwind'.")
        elseif finite_difference_option_z == "first_order_upwind"
            print(io,"'first_order_upwind'.")
        else
            input_option_error("finite_difference_option_z", finite_difference_option_z)
        end
        println(io,"using finite differences on an equally space grid in z.")
    else
        input_option_error("discretization_option_z", discretization_option_z)
    end
    # boundary_option determines z boundary condition
    # supported options are "constant" and "periodic"
    if boundary_option_z == "constant"
        println(io,">boundary_option_z = 'constant'.  enforcing constant incoming BC in z.")
    elseif boundary_option_z == "periodic"
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
        print(io,">discretization_option_vpa = 'finite_difference', and
            finite_difference_option_vpa = ")
        if finite_difference_option_vpa == "second_order_upwind"
            print(io,"'second_order_upwind'.")
        elseif finite_difference_option_vpa == "first_order_upwind"
            print(io,"'first_order_upwind'.")
        else
            input_option_error("finite_difference_option_vpa", finite_difference_option_vpa)
        end
        println(io,"using finite differences on an equally space grid in vpa.")
    else
        input_option_error("discretization_option_vpa", discretization_option_vpa)
    end
    # boundary_option determines vpa boundary condition
    # supported options are "zero" and "periodic"
    if boundary_option_vpa == "zero"
        println(io,">boundary_option_vpa = 'zero'.  enforcing zero incoming BC in vpa.")
    elseif boundary_option_vpa == "periodic"
        println(io,">boundary_option_vpa = 'periodic'.  enforcing periodicity in vpa.")
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
