# add the current directory to the path where the code looks for external modules
push!(LOAD_PATH, ".")

module tests

using input_structs: grid_input, advection_input
using coordinates: define_coordinate
using chebyshev: setup_chebyshev_pseudospectral
using derivatives: derivative!, integral

function test_fundamental_theorem_calculus_chebyshev()
    print("performing test of fundamental theorem of calculus...")
    etol = 1.0e-15
    # define inputs needed for the test
    ngrid = 5
    nelement = 2
    L = 6.0
    discretization = "chebyshev_pseudospectral"
    bc = "periodic"
    # fd_option and adv_input not actually used so given values unimportant
    fd_option = ""
    adv_input = advection_input("default", 1.0, 0.0, 0.0)
    # create the 'input' struct containing input info needed to create a coordinate
    input = grid_input("coord", ngrid, nelement, L,
        discretization, fd_option, bc, adv_input)
    # create the coordinate struct 'x'
    x = define_coordinate(input)
    # create arrays needed for Chebyshev pseudospectral treatment in x
    # and create the plans for the forward and backward fast Chebyshev transforms
    spectral = setup_chebyshev_pseudospectral(x)
    # create array for the function f(x) to be differentiated/integrated
    f = Array{Float64,1}(undef, x.n)
    # create array for the derivative df/dx
    df = Array{Float64,1}(undef, x.n)
    # initialize f
    for ix ∈ 1:x.n
        f[ix] = (cospi(2.0*x.grid[ix]/x.L)+sinpi(2.0*x.grid[ix]/x.L))*exp(-x.grid[ix]^2)
    end
    # differentiate f
    derivative!(df, f, x, spectral)
    # integrate df/dx
    intdf = integral(df, x.wgts)
    # open ascii file to which test info will be written
    io = open("tests.txt","w")
    for ix ∈ 1:x.n
        println(io, "x: ", x.grid[ix], " f: ", f[ix], "  df: ", df[ix], " intdf: ", intdf)
    end
    if intdf < etol
        println("passed.")
    else
        println("failed.  Error ", intdf, " greater than the specified error tolerance ", etol)
    end
end

test_fundamental_theorem_calculus_chebyshev()

end
