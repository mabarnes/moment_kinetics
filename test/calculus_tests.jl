module CalculusTests

include("setup.jl")

using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.calculus: derivative!, second_derivative!, integral

using MPI
using Random

function runtests()
    @testset "calculus" verbose=use_verbose begin
        println("calculus tests")
        @testset "fundamental theorem of calculus" begin
            @testset "$discretization $ngrid $nelement" for
                    (discretization, element_spacing_option, etol) ∈ (("finite_difference", "uniform", 1.0e-15), ("chebyshev_pseudospectral", "uniform", 1.0e-15), ("chebyshev_pseudospectral", "sqrt", 1.0e-2)),
                    ngrid ∈ (5,6,7,8,9,10), nelement ∈ (1, 2, 3, 4, 5)

                if discretization == "finite_difference" && (ngrid - 1) * nelement % 2 == 1
                    # When the total number of points (counting the periodically identified
                    # end points a a single point) is odd, we have to use Simpson's 3/8 rule
                    # for integration for one set of points at the beginning of the array,
                    # which breaks the symmetry that makes integration of the derivative
                    # exact, so this test would fail
                    continue
                end

                # define inputs needed for the test
                L = 6.0
                bc = "periodic"
                # fd_option and adv_input not actually used so given values unimportant
                fd_option = ""
                adv_input = advection_input("default", 1.0, 0.0, 0.0)
                # create the 'input' struct containing input info needed to create a
                # coordinate
                nelement_local = nelement
				nrank_per_block = 0 # dummy value
				irank = 0 # dummy value
				comm = MPI.COMM_NULL # dummy value 
				input = grid_input("coord", ngrid, nelement,
                    nelement_local, nrank_per_block, irank, L,
                    discretization, fd_option, bc, adv_input, comm,
                    element_spacing_option)
                # create the coordinate struct 'x'
                x, spectral = define_coordinate(input)
                # create array for the function f(x) to be differentiated/integrated
                f = Array{Float64,1}(undef, x.n)
                # create array for the derivative df/dx
                df = Array{Float64,1}(undef, x.n)
                # initialize f
                for ix ∈ 1:x.n
                    f[ix] = ( (cospi(2.0*x.grid[ix]/x.L)+sinpi(2.0*x.grid[ix]/x.L))
                              * exp(-x.grid[ix]^2) )
                end
                # differentiate f
                derivative!(df, f, x, spectral)
                # integrate df/dx
                intdf = integral(df, x.wgts)

                ## open ascii file to which test info will be written
                #io = open("tests.txt","w")
                #for ix ∈ 1:x.n
                #    println(io, "x: ", x.grid[ix], " f: ", f[ix], "  df: ", df[ix], " intdf: ", intdf)
                #end

                # Test that error intdf is less than the specified error tolerance etol
                @test abs(intdf) < etol
            end
        end

        rng = MersenneTwister(42)

        @testset "finite_difference derivatives (4 argument), periodic" verbose=false begin
            @testset "$nelement $ngrid" for nelement ∈ (1:5), ngrid ∈ (9:33)

                # define inputs needed for the test
                L = 6.0
                bc = "periodic"
                # fd_option and adv_input not actually used so given values unimportant
                fd_option = ""
                adv_input = advection_input("default", 1.0, 0.0, 0.0)
                # create the 'input' struct containing input info needed to create a
                # coordinate
                nelement_local = nelement
				nrank_per_block = 0 # dummy value
				irank = 0 # dummy value
				comm = MPI.COMM_NULL # dummy value 
				element_spacing_option = "uniform" # dummy value
                input = grid_input("coord", ngrid, nelement,
                    nelement_local, nrank_per_block, irank, L,
                    "finite_difference", fd_option, bc, adv_input, comm,
                    element_spacing_option)
                # create the coordinate struct 'x'
                x, spectral = define_coordinate(input)

                # create array for the derivative df/dx and the expected result
                df = Array{Float64,1}(undef, x.n)

                # initialize f and expected df
                offset = randn(rng)
                f = @. sinpi(2.0 * x.grid / L) + offset
                expected_df = @. 2.0 * π / L * cospi(2.0 * x.grid / L)

                # differentiate f
                derivative!(df, f, x, false)

                rtol = 1.e2 / (nelement*(ngrid-1))^4
                @test isapprox(df, expected_df, rtol=rtol, atol=1.e-15,
                               norm=maxabs_norm)
            end
        end

        @testset "finite_difference derivatives upwinding (5 argument), periodic" verbose=false begin
            @testset "$nelement $ngrid" for
                    (fd_option, order) ∈ (
                                          ("fourth_order_centered", 4),
                                          ("second_order_centered", 2),
                                          ("fourth_order_upwind", 4),
                                          ("third_order_upwind", 3),
                                          ("second_order_upwind", 2),
                                          ("first_order_upwind", 1),
                                         ),
                    nelement ∈ (1:5), ngrid ∈ (9:33)

                # define inputs needed for the test
                L = 6.0
                bc = "periodic"
                # fd_option and adv_input not actually used so given values unimportant
                fd_option = "fourth_order_centered"
                adv_input = advection_input("default", 1.0, 0.0, 0.0)
                # create the 'input' struct containing input info needed to create a
                # coordinate
                nelement_local = nelement
				nrank_per_block = 0 # dummy value
				irank = 0 # dummy value
				comm = MPI.COMM_NULL # dummy value
                element_spacing_option = "uniform" # dummy value
				input = grid_input("coord", ngrid, nelement,
                    nelement_local, nrank_per_block, irank, L,
                    "finite_difference", fd_option, bc, adv_input, comm,
                    element_spacing_option)
                # create the coordinate struct 'x'
                x, spectral = define_coordinate(input)

                # create array for the derivative df/dx and the expected result
                df = Array{Float64,1}(undef, x.n)

                # initialize f and expected df
                offset = randn(rng)
                f = @. sinpi(2.0 * x.grid / L) + offset
                expected_df = @. 2.0 * π / L * cospi(2.0 * x.grid / L)

                for advection ∈ (-1.0, 0.0, 1.0)
                    adv_fac = similar(f)
                    adv_fac .= advection

                    # differentiate f
                    derivative!(df, f, x, adv_fac, false)

                    rtol = 1.e2 / (nelement*(ngrid-1))^order
                    @test isapprox(df, expected_df, rtol=rtol, atol=1.e-15,
                                   norm=maxabs_norm)
                end
            end
        end

        @testset "finite_difference derivatives (4 argument), Neumann" verbose=false begin
            @testset "$nelement $ngrid" for bc ∈ ("constant", "zero"),
                    nelement ∈ (1:5), ngrid ∈ (9:33)

                # define inputs needed for the test
                L = 6.0
                # fd_option and adv_input not actually used so given values unimportant
                fd_option = ""
                adv_input = advection_input("default", 1.0, 0.0, 0.0)
                # create the 'input' struct containing input info needed to create a
                # coordinate
                nelement_local = nelement
				nrank_per_block = 0 # dummy value
				irank = 0 # dummy value
				comm = MPI.COMM_NULL # dummy value
                element_spacing_option = "uniform" # dummy value
				input = grid_input("coord", ngrid, nelement,
                    nelement_local, nrank_per_block, irank, L,
                    "finite_difference", fd_option, bc, adv_input, comm,
                    element_spacing_option)
                # create the coordinate struct 'x'
                x, spectral = define_coordinate(input)

                # create array for the derivative df/dx and the expected result
                df = Array{Float64,1}(undef, x.n)

                # initialize f and expected df
                offset = undef
                if bc == "zero"
                    offset = 1.0
                else
                    offset = randn(rng)
                end
                f = @. cospi(2.0 * x.grid / L) + offset
                expected_df = @. -2.0 * π / L * sinpi(2.0 * x.grid / L)

                # differentiate f
                derivative!(df, f, x, false)

                # Note: only get 1st order convergence at the boundary for an input
                # function that has zero gradient at the boundary
                rtol = 3.e0 / (nelement*(ngrid-1))
                @test isapprox(df, expected_df, rtol=rtol, atol=1.e-15,
                               norm=maxabs_norm)
            end
        end

        @testset "finite_difference derivatives upwinding (5 argument), Neumann" verbose=false begin
            @testset "$nelement $ngrid" for bc ∈ ("constant", "zero"),
                    (fd_option, rtol_prefactor) ∈ (("fourth_order_centered", 3.0),
                                                   ("second_order_centered", 3.0),
                                                   #("fourth_order_upwind", 3.0), # not defined yet
                                                   ("third_order_upwind", 3.0),
                                                   ("second_order_upwind", 3.0),
                                                   ("first_order_upwind", 5.0)
                                                  ),
                    nelement ∈ (1:5), ngrid ∈ (9:33)

                # define inputs needed for the test
                L = 6.0
                # fd_option and adv_input not actually used so given values unimportant
                adv_input = advection_input("default", 1.0, 0.0, 0.0)
                # create the 'input' struct containing input info needed to create a
                # coordinate
                nelement_local = nelement
				nrank_per_block = 0 # dummy value
				irank = 0 # dummy value
				comm = MPI.COMM_NULL # dummy value
                element_spacing_option = "uniform" # dummy value
				input = grid_input("coord", ngrid, nelement,
                    nelement_local, nrank_per_block, irank, L,
                    "finite_difference", fd_option, bc, adv_input, comm,
                    element_spacing_option)
               # create the coordinate struct 'x'
                x, spectral = define_coordinate(input)

                # create array for the derivative df/dx and the expected result
                df = Array{Float64,1}(undef, x.n)

                # initialize f and expected df
                offset = undef
                if bc == "zero"
                    offset = 1.0
                else
                    offset = randn(rng)
                end
                f = @. cospi(2.0 * x.grid / L) + offset
                expected_df = @. -2.0 * π / L * sinpi(2.0 * x.grid / L)

                for advection ∈ (-1.0, 0.0, 1.0)
                    adv_fac = similar(f)
                    adv_fac .= advection

                    # differentiate f
                    derivative!(df, f, x, adv_fac, false)

                    # Note: only get 1st order convergence at the boundary for an input
                    # function that has zero gradient at the boundary
                    rtol = rtol_prefactor / (nelement*(ngrid-1))
                    @test isapprox(df[2:end-1], expected_df[2:end-1], rtol=rtol, atol=1.e-15,
                                   norm=maxabs_norm)
                    # Some methods use 1st order one-sided derivative at boundaries
                    # (where derivative is zero for this example), so need higher atol
                    for ix ∈ (1,x.n)
                        @test isapprox(df[ix], expected_df[ix], rtol=rtol, atol=2.0*rtol,
                                       norm=maxabs_norm)
                    end
                end
            end
        end

        @testset "Chebyshev pseudospectral derivatives (4 argument), periodic" verbose=false begin
            @testset "$nelement $ngrid" for (nelement, ngrid, rtol) ∈
                    (
                     (1, 5, 8.e-1),
                     (1, 6, 2.e-1),
                     (1, 7, 1.e-1),
                     (1, 8, 1.e-2),
                     (1, 9, 5.e-3),
                     (1, 10, 3.e-3),
                     (1, 11, 1.e-4),
                     (1, 12, 5.e-6),
                     (1, 13, 3.e-6),
                     (1, 14, 8.e-8),
                     (1, 15, 4.e-8),
                     (1, 16, 8.e-10),
                     (1, 17, 4.e-10),
                     (1, 18, 4.e-12),
                     (1, 19, 2.e-12),
                     (1, 20, 2.e-13),
                     (1, 21, 2.e-13),
                     (1, 22, 2.e-13),
                     (1, 23, 2.e-13),
                     (1, 24, 2.e-13),
                     (1, 25, 2.e-13),
                     (1, 26, 2.e-13),
                     (1, 27, 2.e-13),
                     (1, 28, 2.e-13),
                     (1, 29, 2.e-13),
                     (1, 30, 2.e-13),
                     (1, 31, 2.e-13),
                     (1, 32, 2.e-13),
                     (1, 33, 2.e-13),

                     (2, 4, 2.e-1),
                     (2, 5, 4.e-2),
                     (2, 6, 2.e-2),
                     (2, 7, 4.e-4),
                     (2, 8, 2.e-4),
                     (2, 9, 4.e-6),
                     (2, 10, 2.e-6),
                     (2, 11, 2.e-8),
                     (2, 12, 1.e-8),
                     (2, 13, 1.e-10),
                     (2, 14, 5.e-11),
                     (2, 15, 4.e-13),
                     (2, 16, 2.e-13),
                     (2, 17, 2.e-13),
                     (2, 18, 2.e-13),
                     (2, 19, 2.e-13),
                     (2, 20, 2.e-13),
                     (2, 21, 2.e-13),
                     (2, 22, 2.e-13),
                     (2, 23, 2.e-13),
                     (2, 24, 2.e-13),
                     (2, 25, 2.e-13),
                     (2, 26, 2.e-13),
                     (2, 27, 2.e-13),
                     (2, 28, 2.e-13),
                     (2, 29, 2.e-13),
                     (2, 30, 4.e-13),
                     (2, 31, 4.e-13),
                     (2, 32, 4.e-13),
                     (2, 33, 4.e-13),

                     (3, 3, 4.e-1),
                     (3, 4, 1.e-1),
                     (3, 5, 1.e-2),
                     (3, 6, 2.e-3),
                     (3, 7, 1.e-4),
                     (3, 8, 1.e-5),
                     (3, 9, 6.e-7),
                     (3, 10, 5.e-8),
                     (3, 11, 2.e-9),
                     (3, 12, 1.e-10),
                     (3, 13, 5.e-12),
                     (3, 14, 3.e-13),
                     (3, 15, 2.e-13),
                     (3, 16, 2.e-13),
                     (3, 17, 2.e-13),
                     (3, 18, 2.e-13),
                     (3, 19, 2.e-13),
                     (3, 20, 2.e-13),
                     (3, 21, 2.e-13),
                     (3, 22, 2.e-13),
                     (3, 23, 2.e-13),
                     (3, 24, 2.e-13),
                     (3, 25, 2.e-13),
                     (3, 26, 2.e-13),
                     (3, 27, 2.e-13),
                     (3, 28, 2.e-13),
                     (3, 29, 4.e-13),
                     (3, 30, 4.e-13),
                     (3, 31, 4.e-13),
                     (3, 32, 4.e-13),
                     (3, 33, 4.e-13),

                     (4, 3, 3.e-1),
                     (4, 4, 4.e-2),
                     (4, 5, 4.e-3),
                     (4, 6, 4.e-4),
                     (4, 7, 4.e-5),
                     (4, 8, 1.e-6),
                     (4, 9, 8.e-8),
                     (4, 10, 4.e-9),
                     (4, 11, 1.e-10),
                     (4, 12, 4.e-12),
                     (4, 13, 2.e-13),
                     (4, 14, 2.e-13),
                     (4, 15, 2.e-13),
                     (4, 16, 2.e-13),
                     (4, 17, 2.e-13),
                     (4, 18, 2.e-13),
                     (4, 19, 2.e-13),
                     (4, 20, 2.e-13),
                     (4, 21, 2.e-13),
                     (4, 22, 2.e-13),
                     (4, 23, 2.e-13),
                     (4, 24, 4.e-13),
                     (4, 25, 4.e-13),
                     (4, 26, 4.e-13),
                     (4, 27, 4.e-13),
                     (4, 28, 4.e-13),
                     (4, 29, 4.e-13),
                     (4, 30, 4.e-13),
                     (4, 31, 4.e-13),
                     (4, 32, 4.e-13),
                     (4, 33, 4.e-13),

                     (5, 3, 2.e-1),
                     (5, 4, 2.e-2),
                     (5, 5, 2.e-3),
                     (5, 6, 1.e-4),
                     (5, 7, 1.e-5),
                     (5, 8, 2.e-7),
                     (5, 9, 2.e-8),
                     (5, 10, 3.e-10),
                     (5, 11, 2.e-11),
                     (5, 12, 3.e-13),
                     (5, 13, 2.e-13),
                     (5, 14, 2.e-13),
                     (5, 15, 2.e-13),
                     (5, 16, 2.e-13),
                     (5, 17, 4.e-13),
                     (5, 18, 4.e-13),
                     (5, 19, 4.e-13),
                     (5, 20, 4.e-13),
                     (5, 21, 4.e-13),
                     (5, 22, 8.e-13),
                     (5, 23, 8.e-13),
                     (5, 24, 8.e-13),
                     (5, 25, 8.e-13),
                     (5, 26, 8.e-13),
                     (5, 27, 8.e-13),
                     (5, 28, 8.e-13),
                     (5, 29, 8.e-13),
                     (5, 30, 8.e-13),
                     (5, 31, 8.e-13),
                     (5, 32, 8.e-13),
                     (5, 33, 8.e-13),
                    )

                # define inputs needed for the test
                L = 6.0
                bc = "periodic"
                # fd_option and adv_input not actually used so given values unimportant
                fd_option = ""
                adv_input = advection_input("default", 1.0, 0.0, 0.0)
                # create the 'input' struct containing input info needed to create a
                # coordinate
                nelement_local = nelement
				nrank_per_block = 0 # dummy value
				irank = 0 # dummy value
				comm = MPI.COMM_NULL # dummy value
                element_spacing_option = "uniform"
				input = grid_input("coord", ngrid, nelement,
                    nelement_local, nrank_per_block, irank, L,
                    "chebyshev_pseudospectral", fd_option, bc, adv_input, comm,
                    element_spacing_option)
                # create the coordinate struct 'x' and info for derivatives, etc.
                x, spectral = define_coordinate(input)

                offset = randn(rng)
                f = @. sinpi(2.0 * x.grid / L) + offset
                expected_df = @. 2.0 * π / L * cospi(2.0 * x.grid / L)

                # create array for the derivative df/dx
                df = similar(f)

                # differentiate f
                derivative!(df, f, x, spectral)

                @test isapprox(df, expected_df, rtol=rtol, atol=1.e-14,
                               norm=maxabs_norm)
            end
        end

        @testset "Chebyshev pseudospectral derivatives upwinding (5 argument), periodic" verbose=false begin
            @testset "$nelement $ngrid" for (nelement, ngrid, rtol) ∈
                    (
                     (1, 5, 8.e-1),
                     (1, 6, 2.e-1),
                     (1, 7, 1.e-1),
                     (1, 8, 1.e-2),
                     (1, 9, 5.e-3),
                     (1, 10, 3.e-3),
                     (1, 11, 1.e-4),
                     (1, 12, 5.e-6),
                     (1, 13, 3.e-6),
                     (1, 14, 8.e-8),
                     (1, 15, 4.e-8),
                     (1, 16, 8.e-10),
                     (1, 17, 4.e-10),
                     (1, 18, 4.e-12),
                     (1, 19, 2.e-12),
                     (1, 20, 2.e-13),
                     (1, 21, 2.e-13),
                     (1, 22, 2.e-13),
                     (1, 23, 2.e-13),
                     (1, 24, 2.e-13),
                     (1, 25, 2.e-13),
                     (1, 26, 2.e-13),
                     (1, 27, 2.e-13),
                     (1, 28, 2.e-13),
                     (1, 29, 2.e-13),
                     (1, 30, 2.e-13),
                     (1, 31, 2.e-13),
                     (1, 32, 2.e-13),
                     (1, 33, 2.e-13),

                     (2, 4, 2.e-1),
                     (2, 5, 4.e-2),
                     (2, 6, 2.e-2),
                     (2, 7, 4.e-4),
                     (2, 8, 2.e-4),
                     (2, 9, 4.e-6),
                     (2, 10, 2.e-6),
                     (2, 11, 2.e-8),
                     (2, 12, 1.e-8),
                     (2, 13, 1.e-10),
                     (2, 14, 5.e-11),
                     (2, 15, 4.e-13),
                     (2, 16, 2.e-13),
                     (2, 17, 2.e-13),
                     (2, 18, 2.e-13),
                     (2, 19, 2.e-13),
                     (2, 20, 2.e-13),
                     (2, 21, 2.e-13),
                     (2, 22, 2.e-13),
                     (2, 23, 2.e-13),
                     (2, 24, 2.e-13),
                     (2, 25, 2.e-13),
                     (2, 26, 2.e-13),
                     (2, 27, 2.e-13),
                     (2, 28, 2.e-13),
                     (2, 29, 2.e-13),
                     (2, 30, 4.e-13),
                     (2, 31, 4.e-13),
                     (2, 32, 4.e-13),
                     (2, 33, 4.e-13),

                     (3, 3, 4.e-1),
                     (3, 4, 1.e-1),
                     (3, 5, 1.e-2),
                     (3, 6, 2.e-3),
                     (3, 7, 1.e-4),
                     (3, 8, 1.e-5),
                     (3, 9, 6.e-7),
                     (3, 10, 5.e-8),
                     (3, 11, 2.e-9),
                     (3, 12, 1.e-10),
                     (3, 13, 5.e-12),
                     (3, 14, 3.e-13),
                     (3, 15, 2.e-13),
                     (3, 16, 2.e-13),
                     (3, 17, 2.e-13),
                     (3, 18, 2.e-13),
                     (3, 19, 2.e-13),
                     (3, 20, 2.e-13),
                     (3, 21, 2.e-13),
                     (3, 22, 2.e-13),
                     (3, 23, 2.e-13),
                     (3, 24, 4.e-13),
                     (3, 25, 4.e-13),
                     (3, 26, 4.e-13),
                     (3, 27, 4.e-13),
                     (3, 28, 4.e-13),
                     (3, 29, 4.e-13),
                     (3, 30, 4.e-13),
                     (3, 31, 4.e-13),
                     (3, 32, 4.e-13),
                     (3, 33, 4.e-13),

                     (4, 3, 3.e-1),
                     (4, 4, 4.e-2),
                     (4, 5, 4.e-3),
                     (4, 6, 4.e-4),
                     (4, 7, 4.e-5),
                     (4, 8, 1.e-6),
                     (4, 9, 8.e-8),
                     (4, 10, 4.e-9),
                     (4, 11, 1.e-10),
                     (4, 12, 4.e-12),
                     (4, 13, 2.e-13),
                     (4, 14, 2.e-13),
                     (4, 15, 2.e-13),
                     (4, 16, 2.e-13),
                     (4, 17, 2.e-13),
                     (4, 18, 4.e-13),
                     (4, 19, 4.e-13),
                     (4, 20, 4.e-13),
                     (4, 21, 4.e-13),
                     (4, 22, 4.e-13),
                     (4, 23, 4.e-13),
                     (4, 24, 4.e-13),
                     (4, 25, 4.e-13),
                     (4, 26, 4.e-13),
                     (4, 27, 4.e-13),
                     (4, 28, 4.e-13),
                     (4, 29, 4.e-13),
                     (4, 30, 8.e-13),
                     (4, 31, 8.e-13),
                     (4, 32, 8.e-13),
                     (4, 33, 8.e-13),

                     (5, 3, 2.e-1),
                     (5, 4, 2.e-2),
                     (5, 5, 2.e-3),
                     (5, 6, 1.e-4),
                     (5, 7, 1.e-5),
                     (5, 8, 4.e-7),
                     (5, 9, 2.e-8),
                     (5, 10, 3.e-10),
                     (5, 11, 2.e-11),
                     (5, 12, 3.e-13),
                     (5, 13, 2.e-13),
                     (5, 14, 2.e-13),
                     (5, 15, 2.e-13),
                     (5, 16, 2.e-13),
                     (5, 17, 4.e-13),
                     (5, 18, 4.e-13),
                     (5, 19, 4.e-13),
                     (5, 20, 4.e-13),
                     (5, 21, 4.e-13),
                     (5, 22, 8.e-13),
                     (5, 23, 8.e-13),
                     (5, 24, 8.e-13),
                     (5, 25, 8.e-13),
                     (5, 26, 8.e-13),
                     (5, 27, 8.e-13),
                     (5, 28, 8.e-13),
                     (5, 29, 8.e-13),
                     (5, 30, 8.e-13),
                     (5, 31, 8.e-13),
                     (5, 32, 8.e-13),
                     (5, 33, 8.e-13),
                    )

                # define inputs needed for the test
                L = 6.0
                bc = "periodic"
                # fd_option and adv_input not actually used so given values unimportant
                fd_option = ""
                adv_input = advection_input("default", 1.0, 0.0, 0.0)
                # create the 'input' struct containing input info needed to create a
                # coordinate
                nelement_local = nelement
				nrank_per_block = 0 # dummy value
				irank = 0 # dummy value
				comm = MPI.COMM_NULL # dummy value
                element_spacing_option = "uniform"
				input = grid_input("coord", ngrid, nelement,
                    nelement_local, nrank_per_block, irank, L,
                    "chebyshev_pseudospectral", fd_option, bc, adv_input, comm,
                    element_spacing_option)
                # create the coordinate struct 'x' and info for derivatives, etc.
                x, spectral = define_coordinate(input)

                offset = randn(rng)
                f = @. sinpi(2.0 * x.grid / L) + offset
                expected_df = @. 2.0 * π / L * cospi(2.0 * x.grid / L)

                # create array for the derivative df/dx
                df = similar(f)

                for advection ∈ (-1.0, 0.0, 1.0)
                    adv_fac = similar(f)
                    adv_fac .= advection

                    # differentiate f
                    derivative!(df, f, x, adv_fac, spectral)

                    @test isapprox(df, expected_df, rtol=rtol, atol=1.e-14,
                                   norm=maxabs_norm)
                end
            end
        end

        @testset "Chebyshev pseudospectral derivatives (4 argument), Neumann" verbose=false begin
            @testset "$nelement $ngrid" for bc ∈ ("constant", "zero"), element_spacing_option ∈ ("uniform", "sqrt"),
                    nelement ∈ (1:5), ngrid ∈ (3:33)

                # define inputs needed for the test
                L = 1.0
                bc = "constant"
                # fd_option and adv_input not actually used so given values unimportant
                fd_option = ""
                adv_input = advection_input("default", 1.0, 0.0, 0.0)
                # create the 'input' struct containing input info needed to create a
                # coordinate
                nelement_local = nelement
				nrank_per_block = 0 # dummy value
				irank = 0 # dummy value
				comm = MPI.COMM_NULL # dummy value
                input = grid_input("coord", ngrid, nelement,
                    nelement_local, nrank_per_block, irank, L,
                    "chebyshev_pseudospectral", fd_option, bc, adv_input, comm,
                    element_spacing_option)
                # create the coordinate struct 'x' and info for derivatives, etc.
                x, spectral = define_coordinate(input)

                # test polynomials up to order ngrid-1
                for n ∈ 0:ngrid-1
                    # create array for the function f(x) to be differentiated/integrated
                    f = Array{Float64,1}(undef, x.n)
                    # create array for the derivative df/dx and the expected result
                    df = similar(f)
                    expected_df = similar(f)
                    # initialize f and expected df
                    f[:] .= randn(rng)
                    expected_df .= 0.0
                    for p ∈ 1:n
                        coefficient = randn(rng)
                        @. f += coefficient * x.grid ^ p
                        @. expected_df += coefficient * p * x.grid ^ (p - 1)
                    end
                    # differentiate f
                    derivative!(df, f, x, spectral)

                    # Note the error we might expect for a p=32 polynomial is probably
                    # something like p*(round-off) for x^p (?) so error on expected_df would
                    # be p*p*(round-off), or plausibly 1024*(round-off), so tolerance of
                    # 2e-11 isn't unreasonable.
                    @test isapprox(df, expected_df, rtol=2.0e-11, atol=2.0e-12,
                                   norm=maxabs_norm)
                end
            end
        end

        @testset "Chebyshev pseudospectral derivatives upwinding (5 argument), Neumann" verbose=false begin
            @testset "$nelement $ngrid" for bc ∈ ("constant", "zero"), element_spacing_option ∈ ("uniform", "sqrt"),
                    nelement ∈ (1:5), ngrid ∈ (3:33)

                # define inputs needed for the test
                L = 1.0
                bc = "constant"
                # fd_option and adv_input not actually used so given values unimportant
                fd_option = ""
                adv_input = advection_input("default", 1.0, 0.0, 0.0)
                # create the 'input' struct containing input info needed to create a
                # coordinate
                nelement_local = nelement
				nrank_per_block = 0 # dummy value
				irank = 0 # dummy value
				comm = MPI.COMM_NULL # dummy value
                input = grid_input("coord", ngrid, nelement,
                    nelement_local, nrank_per_block, irank, L,
                    "chebyshev_pseudospectral", fd_option, bc, adv_input, comm,
                    element_spacing_option)
                # create the coordinate struct 'x' and info for derivatives, etc.
                x, spectral = define_coordinate(input)

                # test polynomials up to order ngrid-1
                for n ∈ 0:ngrid-1
                    # create array for the function f(x) to be differentiated/integrated
                    f = Array{Float64,1}(undef, x.n)
                    # create array for the derivative df/dx and the expected result
                    df = similar(f)
                    expected_df = similar(f)
                    # initialize f and expected df
                    f[:] .= randn(rng)
                    expected_df .= 0.0
                    for p ∈ 1:n
                        coefficient = randn(rng)
                        @. f += coefficient * x.grid ^ p
                        @. expected_df += coefficient * p * x.grid ^ (p - 1)
                    end

                    for advection ∈ (-1.0, 0.0, 1.0)
                        adv_fac = similar(f)
                        adv_fac .= advection

                        # differentiate f
                        derivative!(df, f, x, adv_fac, spectral)

                        # Note the error we might expect for a p=32 polynomial is probably
                        # something like p*(round-off) for x^p (?) so error on expected_df
                        # would be p*p*(round-off), or plausibly 1024*(round-off), so
                        # tolerance of 3e-11 isn't unreasonable.
                        @test isapprox(df, expected_df, rtol=3.0e-11, atol=3.0e-12,
                                       norm=maxabs_norm)
                    end
                end
            end
        end

        @testset "Chebyshev pseudospectral second derivatives (4 argument), periodic" verbose=false begin
            @testset "$nelement $ngrid" for (nelement, ngrid, rtol) ∈
                    (
                     (1, 5, 8.e-1),
                     (1, 6, 2.e-1),
                     (1, 7, 1.e-1),
                     (1, 8, 1.e-2),
                     (1, 9, 5.e-3),
                     (1, 10, 3.e-3),
                     (1, 11, 2.e-4),
                     (1, 12, 5.e-6),
                     (1, 13, 4.e-6),
                     (1, 14, 1.e-7),
                     (1, 15, 1.e-7),
                     (1, 16, 2.e-9),
                     (1, 17, 1.e-9),
                     (1, 18, 4.e-12),
                     (1, 19, 2.e-12),
                     (1, 20, 2.e-13),
                     (1, 21, 2.e-13),
                     (1, 22, 2.e-13),
                     (1, 23, 2.e-13),
                     (1, 24, 2.e-13),
                     (1, 25, 2.e-13),
                     (1, 26, 2.e-13),
                     (1, 27, 2.e-13),
                     (1, 28, 2.e-13),
                     (1, 29, 2.e-13),
                     (1, 30, 2.e-13),
                     (1, 31, 2.e-13),
                     (1, 32, 2.e-13),
                     (1, 33, 2.e-13),

                     (2, 4, 2.e-1),
                     (2, 5, 4.e-2),
                     (2, 6, 2.e-2),
                     (2, 7, 4.e-4),
                     (2, 8, 2.e-4),
                     (2, 9, 4.e-6),
                     (2, 10, 4.e-6),
                     (2, 11, 4.e-8),
                     (2, 12, 4.e-8),
                     (2, 13, 2.e-10),
                     (2, 14, 2.e-10),
                     (2, 15, 4.e-13),
                     (2, 16, 2.e-13),
                     (2, 17, 2.e-13),
                     (2, 18, 2.e-13),
                     (2, 19, 2.e-13),
                     (2, 20, 2.e-13),
                     (2, 21, 2.e-13),
                     (2, 22, 2.e-13),
                     (2, 23, 2.e-13),
                     (2, 24, 2.e-13),
                     (2, 25, 2.e-13),
                     (2, 26, 2.e-13),
                     (2, 27, 2.e-13),
                     (2, 28, 2.e-13),
                     (2, 29, 2.e-13),
                     (2, 30, 4.e-13),
                     (2, 31, 4.e-13),
                     (2, 32, 4.e-13),
                     (2, 33, 4.e-13),

                     (3, 3, 4.e-1),
                     (3, 4, 1.e-1),
                     (3, 5, 2.e-2),
                     (3, 6, 4.e-3),
                     (3, 7, 1.e-3),
                     (3, 8, 1.e-4),
                     (3, 9, 1.e-5),
                     (3, 10, 4.e-7),
                     (3, 11, 4.e-8),
                     (3, 12, 2.e-9),
                     (3, 13, 2.e-10),
                     (3, 14, 3.e-13),
                     (3, 15, 2.e-13),
                     (3, 16, 2.e-13),
                     (3, 17, 2.e-13),
                     (3, 18, 2.e-13),
                     (3, 19, 2.e-13),
                     (3, 20, 2.e-13),
                     (3, 21, 2.e-13),
                     (3, 22, 2.e-13),
                     (3, 23, 2.e-13),
                     (3, 24, 2.e-13),
                     (3, 25, 2.e-13),
                     (3, 26, 2.e-13),
                     (3, 27, 2.e-13),
                     (3, 28, 2.e-13),
                     (3, 29, 4.e-13),
                     (3, 30, 4.e-13),
                     (3, 31, 4.e-13),
                     (3, 32, 4.e-13),
                     (3, 33, 4.e-13),

                     (4, 3, 3.e-1),
                     (4, 4, 1.e-1),
                     (4, 5, 1.e-2),
                     (4, 6, 2.e-3),
                     (4, 7, 2.e-4),
                     (4, 8, 2.e-5),
                     (4, 9, 1.e-6),
                     (4, 10, 1.e-7),
                     (4, 11, 4.e-9),
                     (4, 12, 2.e-10),
                     (4, 13, 2.e-13),
                     (4, 14, 2.e-13),
                     (4, 15, 2.e-13),
                     (4, 16, 2.e-13),
                     (4, 17, 2.e-13),
                     (4, 18, 2.e-13),
                     (4, 19, 2.e-13),
                     (4, 20, 2.e-13),
                     (4, 21, 2.e-13),
                     (4, 22, 2.e-13),
                     (4, 23, 2.e-13),
                     (4, 24, 4.e-13),
                     (4, 25, 4.e-13),
                     (4, 26, 4.e-13),
                     (4, 27, 4.e-13),
                     (4, 28, 4.e-13),
                     (4, 29, 4.e-13),
                     (4, 30, 4.e-13),
                     (4, 31, 4.e-13),
                     (4, 32, 4.e-13),
                     (4, 33, 4.e-13),

                     (5, 3, 4.e-1),
                     (5, 4, 4.e-2),
                     (5, 5, 4.e-3),
                     (5, 6, 1.e-3),
                     (5, 7, 4.e-5),
                     (5, 8, 1.e-5),
                     (5, 9, 2.e-7),
                     (5, 10, 1.e-8),
                     (5, 11, 4.e-10),
                     (5, 12, 3.e-13),
                     (5, 13, 2.e-13),
                     (5, 14, 2.e-13),
                     (5, 15, 2.e-13),
                     (5, 16, 2.e-13),
                     (5, 17, 4.e-13),
                     (5, 18, 4.e-13),
                     (5, 19, 4.e-13),
                     (5, 20, 4.e-13),
                     (5, 21, 4.e-13),
                     (5, 22, 8.e-13),
                     (5, 23, 8.e-13),
                     (5, 24, 8.e-13),
                     (5, 25, 8.e-13),
                     (5, 26, 8.e-13),
                     (5, 27, 8.e-13),
                     (5, 28, 8.e-13),
                     (5, 29, 8.e-13),
                     (5, 30, 8.e-13),
                     (5, 31, 8.e-13),
                     (5, 32, 8.e-13),
                     (5, 33, 8.e-13),
                    )

                # define inputs needed for the test
                L = 6.0
                bc = "periodic"
                # fd_option and adv_input not actually used so given values unimportant
                fd_option = ""
                adv_input = advection_input("default", 1.0, 0.0, 0.0)
                # create the 'input' struct containing input info needed to create a
                # coordinate
                nelement_local = nelement
				nrank_per_block = 0 # dummy value
				irank = 0 # dummy value
				comm = MPI.COMM_NULL # dummy value
                element_spacing_option = "uniform"
				input = grid_input("coord", ngrid, nelement,
                    nelement_local, nrank_per_block, irank, L,
                    "chebyshev_pseudospectral", fd_option, bc, adv_input, comm,
                    element_spacing_option)
                # create the coordinate struct 'x' and info for derivatives, etc.
                x, spectral = define_coordinate(input)

                offset = randn(rng)
                f = @. sinpi(2.0 * x.grid / L) + offset
                expected_d2f = @. -4.0 * π^2 / L^2 * sinpi(2.0 * x.grid / L)

                # create array for the derivative d2f/dx2
                d2f = similar(f)

                # differentiate f
                x.scratch2 .= 1.0 # placeholder for Q in d / d x ( Q d f / d x)
                second_derivative!(d2f, f, x.scratch2, x, spectral)

                @test isapprox(d2f, expected_d2f, rtol=rtol, atol=1.e-10,
                               norm=maxabs_norm)
            end
        end
    end
end

end # CalculusTests


using .CalculusTests

CalculusTests.runtests()
