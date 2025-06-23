module CalculusTests

include("setup.jl")

using moment_kinetics.coordinates: define_test_coordinate
using moment_kinetics.calculus: derivative!, second_derivative!, integral
using moment_kinetics.calculus: laplacian_derivative!, indefinite_integral!

using MPI
using Random
using LinearAlgebra: mul!, ldiv!
# To update one of the long list of tolerances, uncomment the line below
# #using Printf
# and insert a print statement, e.g.,
# tolerance = 1.5*maximum(abs.(f .- expected_f))
# tolerance_string = @sprintf("%.2g",tolerance)
# println("($nelement, $ngrid, $tolerance_string),")
# and comment out the failing test that needs updating or replacing

function runtests()
    @testset "calculus" verbose=use_verbose begin
        println("calculus tests")
        @testset "fundamental theorem of calculus" begin
            @testset "$discretization $ngrid $nelement $cheb_option" for
                    (discretization, element_spacing_option, etol, cheb_option) ∈ (("finite_difference", "uniform", 1.0e-15, ""), ("chebyshev_pseudospectral", "uniform", 1.0e-15, "FFT"), ("chebyshev_pseudospectral", "uniform", 2.0e-15, "matrix"), ("chebyshev_pseudospectral", "sqrt", 1.0e-2, "FFT"), ("gausslegendre_pseudospectral", "uniform", 1.0e-14, ""), ("fourier_pseudospectral", "uniform", 2.0e-15, "")),
                    ngrid ∈ (5,6,7,8,9,10), nelement ∈ (1, 2, 3, 4, 5)

                if discretization == "finite_difference" && (ngrid - 1) * nelement % 2 == 1
                    # When the total number of points (counting the periodically identified
                    # end points a a single point) is odd, we have to use Simpson's 3/8 rule
                    # for integration for one set of points at the beginning of the array,
                    # which breaks the symmetry that makes integration of the derivative
                    # exact, so this test would fail
                    continue
                end
                if discretization == "fourier_pseudospectral" && nelement > 1
                    # fourier_pseudospectral requires a single element.
                    continue
                end

                # define inputs needed for the test
                L = 6.0
                if discretization ∈ ("finite_difference", "fourier_pseudospectral")
                    bc = "periodic"
                else
                    bc = "none"
                end
                fd_option = ""
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization=discretization,
                                                     finite_difference_option=fd_option,
                                                     cheb_option=cheb_option, bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)
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

        rng = StableRNG(42)

        @testset "finite_difference derivatives (4 argument), periodic" verbose=false begin
            @testset "$nelement $ngrid" for nelement ∈ (1:5), ngrid ∈ (9:33)

                # define inputs needed for the test
                L = 6.0
                bc = "periodic"
                fd_option = ""
                element_spacing_option = "uniform" # dummy value
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="finite_difference",
                                                     finite_difference_option=fd_option,
                                                     bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)

                # create array for the derivative df/dx and the expected result
                df = Array{Float64,1}(undef, x.n)

                # initialize f and expected df
                offset = randn(rng)
                phase = 0.42
                f = @. sinpi(2.0 * x.grid / L + phase) + offset
                expected_df = @. 2.0 * π / L * cospi(2.0 * x.grid / L + phase)

                # differentiate f
                derivative!(df, f, x, spectral)

                rtol = 1.e2 / (nelement*(ngrid-1))^4
                @test isapprox(df, expected_df, rtol=rtol, atol=1.e-15,
                               norm=maxabs_norm)
                @test df[1] == df[end]
            end
        end

        @testset "finite_difference derivatives upwinding (5 argument), periodic" verbose=false begin
            @testset "$nelement $ngrid $fd_option $is_moment" for
                    (fd_option, order) ∈ (
                                          ("fourth_order_centered", 4),
                                          ("second_order_centered", 2),
                                          ("third_order_upwind", 3),
                                          ("second_order_upwind", 2),
                                          ("first_order_upwind", 1),
                                         ),
                    nelement ∈ (1:5), ngrid ∈ (9:33), is_moment ∈ (false, true)

                # define inputs needed for the test
                L = 6.0
                bc = "periodic"
                element_spacing_option = "uniform" # dummy value
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="finite_difference",
                                                     finite_difference_option=fd_option,
                                                     bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)

                # create array for the derivative df/dx and the expected result
                df = Array{Float64,1}(undef, x.n)

                # initialize f and expected df
                offset = randn(rng)
                phase = 0.42
                f = @. sinpi(2.0 * x.grid / L + phase) + offset
                expected_df = @. 2.0 * π / L * cospi(2.0 * x.grid / L + phase)

                for advection ∈ (-1.0, 0.0, 1.0)
                    adv_fac = similar(f)
                    adv_fac .= advection

                    # differentiate f
                    derivative!(df, f, x, adv_fac, spectral, is_moment)

                    rtol = 1.e2 / (nelement*(ngrid-1))^order
                    @test isapprox(df, expected_df, rtol=rtol, atol=1.e-15,
                                   norm=maxabs_norm)
                    df[1] == df[end]
                end
            end
        end

        @testset "finite_difference derivatives (4 argument)" verbose=false begin
            @testset "$nelement $ngrid" for bc ∈ ("constant", "zero"),
                    nelement ∈ (1:5), ngrid ∈ (9:33)

                # define inputs needed for the test
                L = 6.0
                fd_option = ""
                element_spacing_option = "uniform" # dummy value
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="finite_difference",
                                                     finite_difference_option=fd_option,
                                                     bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)

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
                derivative!(df, f, x, spectral)

                # Note: only get 1st order convergence at the boundary for an input
                # function that has zero gradient at the boundary
                rtol = 3.e0 / (nelement*(ngrid-1))
                @test isapprox(df, expected_df, rtol=rtol, atol=1.e-15,
                               norm=maxabs_norm)
            end
        end

        @testset "finite_difference derivatives upwinding (5 argument)" verbose=false begin
            @testset "$nelement $ngrid $fd_option $is_moment" for bc ∈ ("constant", "zero"),
                    (fd_option, rtol_prefactor) ∈ (("fourth_order_centered", 3.0),
                                                   ("second_order_centered", 3.0),
                                                   ("third_order_upwind", 3.0),
                                                   ("second_order_upwind", 3.0),
                                                   ("first_order_upwind", 5.0)
                                                  ),
                    nelement ∈ (1:5), ngrid ∈ (9:33), is_moment ∈ (false, true)

                # define inputs needed for the test
                L = 6.0
                element_spacing_option = "uniform" # dummy value
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="finite_difference",
                                                     finite_difference_option=fd_option,
                                                     bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)

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
                    derivative!(df, f, x, adv_fac, spectral, is_moment)

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
            @testset "$nelement $ngrid $cheb_option" for (nelement, ngrid, rtol) ∈
                    (
                     (1, 5, 8.e-1),
                     (1, 6, 2.e-1),
                     (1, 7, 1.e-1),
                     (1, 8, 1.e-2),
                     (1, 9, 5.e-3),
                     (1, 10, 3.e-3),
                     (1, 11, 1.e-4),
                     (1, 12, 1.e-5),
                     (1, 13, 3.e-6),
                     (1, 14, 1.e-7),
                     (1, 15, 4.e-8),
                     (1, 16, 1.e-8),
                     (1, 17, 4.e-10),
                     (1, 18, 1.e-10),
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
                     (2, 7, 1.e-3),
                     (2, 8, 2.e-4),
                     (2, 9, 1.e-5),
                     (2, 10, 2.e-6),
                     (2, 11, 1.e-7),
                     (2, 12, 1.e-8),
                     (2, 13, 1.e-9),
                     (2, 14, 5.e-11),
                     (2, 15, 1.e-12),
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
                     (4, 32, 6.e-13),
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
                    ), cheb_option in ("FFT","matrix")

                # define inputs needed for the test
                L = 6.0
                bc = "periodic"
                element_spacing_option = "uniform"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="chebyshev_pseudospectral",
                                                     cheb_option=cheb_option, bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)

                offset = randn(rng)
                phase = 0.42
                f = @. sinpi(2.0 * x.grid / L + phase) + offset
                expected_df = @. 2.0 * π / L * cospi(2.0 * x.grid / L + phase)

                # create array for the derivative df/dx
                df = similar(f)

                # differentiate f
                derivative!(df, f, x, spectral)

                @test isapprox(df, expected_df, rtol=rtol, atol=1.e-14,
                               norm=maxabs_norm)
                @test df[1] == df[end]
            end
        end

        @testset "Chebyshev pseudospectral derivatives upwinding (5 argument), periodic" verbose=false begin
            @testset "$nelement $ngrid $cheb_option $is_moment" for (nelement, ngrid, rtol) ∈
                    (
                     (1, 5, 8.e-1),
                     (1, 6, 2.e-1),
                     (1, 7, 1.e-1),
                     (1, 8, 5.e-2),
                     (1, 9, 5.e-3),
                     (1, 10, 3.e-3),
                     (1, 11, 1.e-4),
                     (1, 12, 3.e-5),
                     (1, 13, 3.e-6),
                     (1, 14, 4.e-7),
                     (1, 15, 4.e-8),
                     (1, 16, 4.e-9),
                     (1, 17, 4.e-10),
                     (1, 18, 4.e-11),
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
                     (2, 5, 6.e-2),
                     (2, 6, 2.e-2),
                     (2, 7, 2.e-3),
                     (2, 8, 2.e-4),
                     (2, 9, 2.e-5),
                     (2, 10, 2.e-6),
                     (2, 11, 1.e-7),
                     (2, 12, 1.e-8),
                     (2, 13, 1.e-9),
                     (2, 14, 5.e-11),
                     (2, 15, 2.e-12),
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
                    ), cheb_option in ("FFT","matrix"), is_moment ∈ (false, true)

                # define inputs needed for the test
                L = 6.0
                bc = "periodic"
                element_spacing_option = "uniform"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="chebyshev_pseudospectral",
                                                     cheb_option=cheb_option, bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)

                offset = randn(rng)
                phase = 0.42
                f = @. sinpi(2.0 * x.grid / L + phase) + offset
                expected_df = @. 2.0 * π / L * cospi(2.0 * x.grid / L + phase)

                # create array for the derivative df/dx
                df = similar(f)

                for advection ∈ (-1.0, 0.0, 1.0)
                    adv_fac = similar(f)
                    adv_fac .= advection

                    # differentiate f
                    derivative!(df, f, x, adv_fac, spectral, is_moment)

                    @test isapprox(df, expected_df, rtol=rtol, atol=1.e-12,
                                   norm=maxabs_norm)
                    @test df[1] == df[end]
                end
            end
        end

        @testset "Chebyshev pseudospectral derivatives (4 argument), polynomials" verbose=false begin
            @testset "$nelement $ngrid $bc $element_spacing_option $cheb_option" for
                    bc ∈ ("constant", "zero"), element_spacing_option ∈ ("uniform", "sqrt"),
                    nelement ∈ (1:5), ngrid ∈ (3:24), cheb_option in ("FFT","matrix")

                # define inputs needed for the test
                L = 1.0
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="chebyshev_pseudospectral",
                                                     cheb_option=cheb_option, bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)
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
                    @test isapprox(df, expected_df, rtol=2.0e-11, atol=6.0e-12,
                                   norm=maxabs_norm)
                end
            end
        end

        @testset "Chebyshev pseudospectral derivatives upwinding (5 argument), polynomials" verbose=false begin
            @testset "$nelement $ngrid $bc $element_spacing_option $cheb_option $is_moment" for
                    bc ∈ ("constant", "zero"), element_spacing_option ∈ ("uniform", "sqrt"),
                    nelement ∈ (1:5), ngrid ∈ (3:33), cheb_option ∈ ("FFT","matrix"), is_moment ∈ (false, true)

                # define inputs needed for the test
                L = 1.0
                bc = "constant"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="chebyshev_pseudospectral",
                                                     cheb_option=cheb_option, bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)
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
                        derivative!(df, f, x, adv_fac, spectral, is_moment)

                        # Note the error we might expect for a p=32 polynomial is probably
                        # something like p*(round-off) for x^p (?) so error on expected_df
                        # would be p*p*(round-off), or plausibly 1024*(round-off), so
                        # tolerance of 3e-11 isn't unreasonable.
                        @test isapprox(df, expected_df, rtol=3.0e-11, atol=3.0e-11,
                                       norm=maxabs_norm)
                    end
                end
            end
        end
        
        @testset "GaussLegendre pseudospectral derivatives (4 argument), testing periodic functions" verbose=false begin
            @testset "$nelement $ngrid" for (nelement, ngrid, rtol) ∈
                    (
                     (1, 5, 8.e-1),
                     (1, 6, 2.e-1),
                     (1, 7, 1.e-1),
                     (1, 8, 1.e-2),
                     (1, 9, 5.e-3),
                     (1, 10, 3.e-3),
                     (1, 11, 5.e-4),
                     (1, 12, 1.e-5),
                     (1, 13, 3.e-6),
                     (1, 14, 3.e-7),
                     (1, 15, 4.e-8),
                     (1, 16, 4.e-9),
                     (1, 17, 8.e-10),
                     

                     (2, 4, 2.e-1),
                     (2, 5, 4.e-2),
                     (2, 6, 2.e-2),
                     (2, 7, 1.e-3),
                     (2, 8, 2.e-4),
                     (2, 9, 1.e-5),
                     (2, 10, 2.e-6),
                     (2, 11, 1.e-7),
                     (2, 12, 1.e-8),
                     (2, 13, 1.e-9),
                     (2, 14, 5.e-11),
                     (2, 15, 2.e-12),
                     (2, 16, 2.e-13),
                     (2, 17, 2.e-13),
                     
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
                     
                     (4, 3, 3.e-1),
                     (4, 4, 4.e-2),
                     (4, 5, 4.e-3),
                     (4, 6, 4.e-4),
                     (4, 7, 4.e-5),
                     (4, 8, 1.e-6),
                     (4, 9, 8.e-8),
                     (4, 10, 4.e-9),
                     (4, 11, 4.e-10),
                     (4, 12, 4.e-12),
                     (4, 13, 2.e-13),
                     (4, 14, 2.e-13),
                     (4, 15, 2.e-13),
                     (4, 16, 2.e-13),
                     (4, 17, 2.e-13),
                     
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
                    )

                # define inputs needed for the test
                L = 6.0
                bc = "periodic"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="gausslegendre_pseudospectral",
                                                     bc=bc,
                                                     collision_operator_dim=false)

                offset = randn(rng)
                phase = 0.42
                f = @. sinpi(2.0 * x.grid / L + phase) + offset
                expected_df = @. 2.0 * π / L * cospi(2.0 * x.grid / L + phase)

                # create array for the derivative df/dx
                df = similar(f)

                # differentiate f
                derivative!(df, f, x, spectral)

                @test isapprox(df, expected_df, rtol=rtol, atol=1.e-14,
                               norm=maxabs_norm)
                @test df[1] == df[end]
            end
        end

        @testset "GaussLegendre pseudospectral derivatives upwinding (5 argument), testing periodic functions" verbose=false begin
            @testset "$nelement $ngrid $is_moment" for (nelement, ngrid, rtol) ∈
                    (
                     (1, 5, 8.e-1),
                     (1, 6, 3.e-1),
                     (1, 7, 1.e-1),
                     (1, 8, 2.e-2),
                     (1, 9, 5.e-3),
                     (1, 10, 3.e-3),
                     (1, 11, 8.e-4),
                     (1, 12, 3.e-5),
                     (1, 13, 3.e-6),
                     (1, 14, 5.e-7),
                     (1, 15, 4.e-8),
                     (1, 16, 8.e-9),
                     (1, 17, 8.e-10),
                     
                     (2, 4, 2.e-1),
                     (2, 5, 8.e-2),
                     (2, 6, 2.e-2),
                     (2, 7, 2.e-3),
                     (2, 8, 2.e-4),
                     (2, 9, 2.e-5),
                     (2, 10, 2.e-6),
                     (2, 11, 2.e-7),
                     (2, 12, 1.e-8),
                     (2, 13, 1.e-9),
                     (2, 14, 5.e-11),
                     (2, 15, 5.e-12),
                     (2, 16, 2.e-13),
                     (2, 17, 2.e-13),
                     
                     (3, 3, 4.e-1),
                     (3, 4, 1.e-1),
                     (3, 5, 3.e-2),
                     (3, 6, 2.e-3),
                     (3, 7, 5.e-4),
                     (3, 8, 1.e-5),
                     (3, 9, 1.e-6),
                     (3, 10, 5.e-8),
                     (3, 11, 2.e-8),
                     (3, 12, 1.e-9),
                     (3, 13, 5.e-11),
                     (3, 14, 3.e-13),
                     (3, 15, 2.e-13),
                     (3, 16, 2.e-13),
                     (3, 17, 2.e-13),
                     
                     (4, 3, 3.e-1),
                     (4, 4, 4.e-2),
                     (4, 5, 4.e-3),
                     (4, 6, 4.e-4),
                     (4, 7, 4.e-5),
                     (4, 8, 4.e-6),
                     (4, 9, 8.e-8),
                     (4, 10, 4.e-9),
                     (4, 11, 5.e-10),
                     (4, 12, 1.e-11),
                     (4, 13, 2.e-13),
                     (4, 14, 2.e-13),
                     (4, 15, 2.e-13),
                     (4, 16, 2.e-13),
                     (4, 17, 2.e-13),
                     
                     (5, 3, 2.e-1),
                     (5, 4, 2.e-2),
                     (5, 5, 2.e-3),
                     (5, 6, 2.e-4),
                     (5, 7, 1.e-5),
                     (5, 8, 4.e-7),
                     (5, 9, 2.e-8),
                     (5, 10, 8.e-10),
                     (5, 11, 2.e-11),
                     (5, 12, 3.e-13),
                     (5, 13, 2.e-13),
                     (5, 14, 2.e-13),
                     (5, 15, 2.e-13),
                     (5, 16, 2.e-13),
                     (5, 17, 4.e-13),
                    ), is_moment ∈ (false, true)

                # define inputs needed for the test
                L = 6.0
                bc = "periodic"
                element_spacing_option = "uniform"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="gausslegendre_pseudospectral",
                                                     bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)

                offset = randn(rng)
                phase = 0.42
                f = @. sinpi(2.0 * x.grid / L + phase) + offset
                expected_df = @. 2.0 * π / L * cospi(2.0 * x.grid / L + phase)

                # create array for the derivative df/dx
                df = similar(f)

                for advection ∈ (-1.0, 0.0, 1.0)
                    adv_fac = similar(f)
                    adv_fac .= advection

                    # differentiate f
                    derivative!(df, f, x, adv_fac, spectral, is_moment)

                    @test isapprox(df, expected_df, rtol=rtol, atol=1.e-12,
                                   norm=maxabs_norm)
                    @test df[1] == df[end]
                end
            end
        end
        
        @testset "GaussLegendre pseudospectral derivatives (4 argument), testing exact polynomials" verbose=false begin
            @testset "$nelement $ngrid" for bc ∈ ("constant", "zero"), element_spacing_option ∈ ("uniform", "sqrt"),
                    nelement ∈ (1:5), ngrid ∈ (3:17)
                    
                # define inputs needed for the test
                L = 1.0
                bc = "constant"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="gausslegendre_pseudospectral",
                                                     bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)
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
                    @test isapprox(df, expected_df, rtol=2.0e-11, atol=6.0e-12,
                                   norm=maxabs_norm)
                end
            end
        end
        
        @testset "GaussLegendre pseudospectral derivatives upwinding (5 argument), testing exact polynomials" verbose=false begin
            @testset "$nelement $ngrid $is_moment" for bc ∈ ("constant", "zero"), element_spacing_option ∈ ("uniform", "sqrt"),
                    nelement ∈ (1:5), ngrid ∈ (3:17), is_moment ∈ (false, true)

                # define inputs needed for the test
                L = 1.0
                bc = "constant"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="gausslegendre_pseudospectral",
                                                     bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)
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
                        derivative!(df, f, x, adv_fac, spectral, is_moment)

                        # Note the error we might expect for a p=32 polynomial is probably
                        # something like p*(round-off) for x^p (?) so error on expected_df
                        # would be p*p*(round-off), or plausibly 1024*(round-off), so
                        # tolerance of 3e-11 isn't unreasonable.
                        @test isapprox(df, expected_df, rtol=3.0e-11, atol=3.0e-11,
                                       norm=maxabs_norm)
                    end
                end
            end
        end

        @testset "Fourier pseudospectral derivatives (4 argument), periodic" verbose=false begin
            @testset "$ngrid" for (ngrid, rtol) ∈
                    (
                     (5, 2.e-13),
                     (6, 2.e-13),
                     (7, 2.e-13),
                     (8, 2.e-13),
                     (9, 2.e-13),
                     (10, 2.e-13),
                     (11, 2.e-13),
                     (12, 2.e-13),
                     (13, 2.e-13),
                     (14, 2.e-13),
                     (15, 2.e-13),
                     (16, 2.e-13),
                     (17, 2.e-13),
                     (18, 2.e-13),
                     (19, 2.e-13),
                     (20, 2.e-13),
                     (21, 2.e-13),
                     (22, 2.e-13),
                     (23, 2.e-13),
                     (24, 2.e-13),
                     (25, 2.e-13),
                     (26, 2.e-13),
                     (27, 2.e-13),
                     (28, 2.e-13),
                     (29, 2.e-13),
                     (30, 2.e-13),
                     (31, 2.e-13),
                     (32, 2.e-13),
                     (33, 2.e-13),
                    )

                # define inputs needed for the test
                nelement = 1
                L = 6.0
                bc = "periodic"
                element_spacing_option = "uniform"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="fourier_pseudospectral",
                                                     cheb_option="", bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)

                offset = randn(rng)
                phase = 0.42
                f = @. sinpi(2.0 * x.grid / L + phase) + offset
                expected_df = @. 2.0 * π / L * cospi(2.0 * x.grid / L + phase)

                # create array for the derivative df/dx
                df = similar(f)

                # differentiate f
                derivative!(df, f, x, spectral)

                @test isapprox(df, expected_df, rtol=rtol, atol=1.e-14,
                               norm=maxabs_norm)
                @test df[1] == df[end]
            end
        end

        @testset "Fourier pseudospectral derivatives upwinding (5 argument), periodic" verbose=false begin
            @testset "$ngrid" for (ngrid, rtol) ∈
                    (
                     (5, 2.e-13),
                     (6, 2.e-13),
                     (7, 2.e-13),
                     (8, 2.e-13),
                     (9, 2.e-13),
                     (10, 2.e-13),
                     (11, 2.e-13),
                     (12, 2.e-13),
                     (13, 2.e-13),
                     (14, 2.e-13),
                     (15, 2.e-13),
                     (16, 2.e-13),
                     (17, 2.e-13),
                     (18, 2.e-13),
                     (19, 2.e-13),
                     (20, 2.e-13),
                     (21, 2.e-13),
                     (22, 2.e-13),
                     (23, 2.e-13),
                     (24, 2.e-13),
                     (25, 2.e-13),
                     (26, 2.e-13),
                     (27, 2.e-13),
                     (28, 2.e-13),
                     (29, 2.e-13),
                     (30, 2.e-13),
                     (31, 2.e-13),
                     (32, 2.e-13),
                     (33, 2.e-13),
                    ), is_moment ∈ (false, true)

                # define inputs needed for the test
                nelement = 1
                L = 6.0
                bc = "periodic"
                element_spacing_option = "uniform"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="fourier_pseudospectral",
                                                     cheb_option="", bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)

                offset = randn(rng)
                phase = 0.42
                f = @. sinpi(2.0 * x.grid / L + phase) + offset
                expected_df = @. 2.0 * π / L * cospi(2.0 * x.grid / L + phase)

                # create array for the derivative df/dx
                df = similar(f)

                for advection ∈ (-1.0, 0.0, 1.0)
                    adv_fac = similar(f)
                    adv_fac .= advection

                    # differentiate f
                    derivative!(df, f, x, adv_fac, spectral, is_moment)

                    @test isapprox(df, expected_df, rtol=rtol, atol=1.e-12,
                                   norm=maxabs_norm)
                    @test df[1] == df[end]
                end
            end
        end

        @testset "Chebyshev pseudospectral second derivatives (4 argument), periodic" verbose=false begin
            @testset "$nelement $ngrid $cheb_option" for (nelement, ngrid, rtol) ∈
                    (
                     (1, 5, 8.e-1),
                     (1, 6, 2.e-1),
                     (1, 7, 1.e-1),
                     (1, 8, 4.e-2),
                     (1, 9, 5.e-3),
                     (1, 10, 3.e-3),
                     (1, 11, 2.e-4),
                     (1, 12, 2.e-4),
                     (1, 13, 8.e-6),
                     (1, 14, 4.e-6),
                     (1, 15, 1.e-7),
                     (1, 16, 1.e-7),
                     (1, 17, 1.e-9),
                     (1, 18, 1.e-9),
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
                     (2, 5, 8.e-2),
                     (2, 6, 2.e-2),
                     (2, 7, 8.e-3),
                     (2, 8, 4.e-4),
                     (2, 9, 2.e-4),
                     (2, 10, 4.e-6),
                     (2, 11, 2.e-6),
                     (2, 12, 4.e-8),
                     (2, 13, 2.e-8),
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
                     (3, 6, 8.e-3),
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
                     (5, 5, 8.e-3),
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
                    ), cheb_option in ("FFT","matrix")

                # define inputs needed for the test
                L = 6.0
                bc = "periodic"
                element_spacing_option = "uniform"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="chebyshev_pseudospectral",
                                                     cheb_option=cheb_option, bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)

                offset = randn(rng)
                phase = 0.42
                f = @. sinpi(2.0 * x.grid / L + phase) + offset
                expected_d2f = @. -4.0 * π^2 / L^2 * sinpi(2.0 * x.grid / L + phase)

                # create array for the derivative d2f/dx2
                d2f = similar(f)

                # differentiate f
                second_derivative!(d2f, f, x, spectral)

                @test isapprox(d2f, expected_d2f, rtol=rtol, atol=1.e-10,
                               norm=maxabs_norm)
                @test d2f[1] == d2f[end]
            end
        end
        
        @testset "Chebyshev pseudospectral cylindrical laplacian derivatives (4 argument), zero" verbose=false begin
            @testset "$nelement $ngrid $cheb_option" for (nelement, ngrid, rtol) ∈
                    (
                     (4, 7, 2.e-1),
                     (4, 8, 2.e-1),
                     (4, 9, 4.e-2),
                     (4, 10, 4.e-2),
                     (4, 11, 5.e-3),
                     (4, 12, 5.e-3),
                     (4, 13, 5.e-3),
                     (4, 14, 5.e-3),
                     (4, 15, 5.e-3),
                     (4, 16, 5.e-3),
                     (4, 17, 5.e-3),
                     (4, 18, 5.e-3),
                     (4, 19, 5.e-3),
                     (4, 20, 5.e-3),
                     (4, 21, 5.e-3),
                     (4, 22, 5.e-3),
                     (4, 23, 5.e-3),
                     (4, 24, 4.e-3),
                     (4, 25, 4.e-3),
                     (4, 26, 4.e-3),
                     (4, 27, 4.e-3),
                     (4, 28, 4.e-3),
                     (4, 29, 4.e-3),
                     (4, 30, 4.e-3),
                     (4, 31, 4.e-3),
                     (4, 32, 4.e-3),
                     (4, 33, 4.e-3),

                     (5, 7, 2.e-1),
                     (5, 8, 8.e-2),
                     (5, 9, 5.e-2),
                     (5, 10, 8.e-3),
                     (5, 11, 8.e-3),
                     (5, 12, 8.e-3),
                     (5, 13, 8.e-3),
                     (5, 14, 8.e-3),
                     (5, 15, 8.e-3),
                     (5, 16, 2.e-3),
                     (5, 17, 2.e-3),
                     (5, 18, 2.e-3),
                     (5, 19, 2.e-3),
                     (5, 20, 2.e-3),
                     (5, 21, 2.e-3),
                     (5, 22, 2.e-3),
                     (5, 23, 2.e-3),
                     (5, 24, 2.e-3),
                     (5, 25, 2.e-3),
                     (5, 26, 2.e-3),
                     (5, 27, 2.e-3),
                     (5, 28, 2.e-3),
                     (5, 29, 2.e-3),
                     (5, 30, 2.e-3),
                     (5, 31, 2.e-3),
                     (5, 32, 2.e-3),
                     (5, 33, 2.e-3),
                    ), cheb_option in ("FFT","matrix")

                # define inputs needed for the test
                L = 6.0
                bc = "zero"
                element_spacing_option = "uniform"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("vperp"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="chebyshev_pseudospectral",
                                                     cheb_option=cheb_option, bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)

                f = @. exp(-x.grid^2)
                expected_d2f = @. 4.0*(x.grid^2 - 1.0)*exp(-x.grid^2)
                # create array for the derivative d2f/dx2
                d2f = similar(f)

                # differentiate f
                laplacian_derivative!(d2f, f, x, spectral)

                @test isapprox(d2f, expected_d2f, rtol=rtol, atol=1.e-10,
                               norm=maxabs_norm)
            end
        end
        
        @testset "GaussLegendre pseudospectral second derivatives (4 argument), periodic" verbose=false begin
            @testset "$nelement $ngrid" for (nelement, ngrid, rtol) ∈
                    (
                     (1, 8, 2.e-2),
                     (1, 9, 5.e-3),
                     (1, 10, 3.e-3),
                     (1, 11, 2.e-4),
                     (1, 12, 2.e-5),
                     (1, 13, 4.e-6),
                     (1, 14, 4.e-7),
                     (1, 15, 1.e-7),
                     (1, 16, 5.e-9),
                     (1, 17, 1.e-9),
                     
                     (2, 4, 2.e-1),
                     (2, 5, 5.e-2),
                     (2, 6, 2.e-2),
                     (2, 7, 2.e-3),
                     (2, 8, 2.e-4),
                     (2, 9, 2.e-5),
                     (2, 10, 4.e-6),
                     (2, 11, 2.e-7),
                     (2, 12, 4.e-8),
                     (2, 13, 8.e-10),
                     (2, 14, 2.e-10),
                     (2, 15, 4.e-13),
                     (2, 16, 2.e-13),
                     (2, 17, 2.e-13),
                     
                     (3, 5, 1.e-1),
                     (3, 6, 2.e-2),
                     (3, 7, 2.e-3),
                     (3, 8, 2.e-4),
                     (3, 9, 1.e-4),
                     (3, 10, 4.e-6),
                     (3, 11, 1.e-7),
                     (3, 12, 8.e-9),
                     (3, 13, 8.e-10),
                     (3, 14, 3.e-10),
                     (3, 15, 2.e-10),
                     (3, 16, 2.e-10),
                     (3, 17, 2.e-10),
                     
                     (4, 5, 5.e-2),
                     (4, 6, 2.e-2),
                     (4, 7, 2.e-3),
                     (4, 8, 2.e-4),
                     (4, 9, 1.e-4),
                     (4, 10, 1.e-6),
                     (4, 11, 8.e-9),
                     (4, 12, 8.e-10),
                     (4, 13, 8.e-10),
                     (4, 14, 8.e-10),
                     (4, 15, 8.e-10),
                     (4, 16, 8.e-10),
                     (4, 17, 8.e-10),
                     
                     (5, 5, 4.e-2),
                     (5, 6, 8.e-3),
                     (5, 7, 5.e-4),
                     (5, 8, 5.e-5),
                     (5, 9, 8.e-7),
                     (5, 10, 5.e-8),
                     (5, 11, 8.e-10),
                     (5, 12, 4.e-10),
                     (5, 13, 2.e-10),
                     (5, 14, 2.e-10),
                     (5, 15, 8.e-10),
                     (5, 16, 8.e-10),
                     (5, 17, 8.e-10),
                     )

                # define inputs needed for the test
                L = 6.0
                bc = "periodic"
                element_spacing_option = "uniform"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="gausslegendre_pseudospectral",
                                                     bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)

                offset = randn(rng)
                phase = 0.42
                f = @. sinpi(2.0 * x.grid / L + phase) + offset
                expected_d2f = @. -4.0 * π^2 / L^2 * sinpi(2.0 * x.grid / L + phase)

                # create array for the derivative d2f/dx2
                d2f = similar(f)

                # differentiate f
                second_derivative!(d2f, f, x, spectral)

                @test isapprox(d2f, expected_d2f, rtol=rtol, atol=1.e-10,
                               norm=maxabs_norm)
                @test d2f[1] == d2f[end]
            end
        end
        
        @testset "GaussLegendre pseudospectral cylindrical laplacian derivatives (4 argument), zero" verbose=false begin
            @testset "$nelement $ngrid" for (nelement, ngrid, rtol) ∈
                    (
                     (1, 8, 1.e-1),
                     (1, 9, 1.e-1),
                     (1, 10, 2.e-1),
                     (1, 11, 6.e-2),
                     (1, 12, 5.e-2),
                     (1, 13, 5.e-2),
                     (1, 14, 5.e-2),
                     (1, 15, 1.e-2),
                     (1, 16, 5.e-2),
                     (1, 17, 5.e-3),
                     
                     (2, 6, 8.e-2),
                     (2, 7, 8.e-2),
                     (2, 8, 5.e-2),
                     (2, 9, 5.e-2),
                     (2, 10, 5.e-2),
                     (2, 11, 5.e-3),
                     (2, 12, 5.e-3),
                     (2, 13, 5.e-4),
                     (2, 14, 5.e-4),
                     (2, 15, 5.e-4),
                     (2, 16, 5.e-4),
                     (2, 17, 5.e-4),
                     
                     (3, 6, 5.e-2),
                     (3, 7, 5.e-3),
                     (3, 8, 5.e-2),
                     (3, 9, 5.e-4),
                     (3, 10, 5.e-3),
                     (3, 11, 5.e-4),
                     (3, 12, 5.e-5),
                     (3, 13, 5.e-5),
                     (3, 14, 5.e-6),
                     (3, 15, 5.e-6),
                     (3, 16, 5.e-6),
                     (3, 17, 5.e-8),
                     
                     (4, 5, 5.e-2),
                     (4, 6, 2.e-2),
                     (4, 7, 2.e-2),
                     (4, 8, 2.e-3),
                     (4, 9, 1.e-3),
                     (4, 10, 1.e-4),
                     (4, 11, 8.e-5),
                     (4, 12, 8.e-6),
                     (4, 13, 8.e-6),
                     (4, 14, 8.e-7),
                     (4, 15, 8.e-7),
                     (4, 16, 8.e-8),
                     (4, 17, 8.e-8),
                     
                     (5, 5, 4.e-2),
                     (5, 6, 8.e-3),
                     (5, 7, 5.e-3),
                     (5, 8, 5.e-4),
                     (5, 9, 8.e-5),
                     (5, 10, 5.e-6),
                     (5, 11, 8.e-6),
                     (5, 12, 4.e-7),
                     (5, 13, 2.e-7),
                     (5, 14, 2.e-7),
                     (5, 15, 8.e-7),
                     (5, 16, 8.e-10),
                     (5, 17, 8.e-10),
                     )

                # define inputs needed for the test
                L = 6.0
                bc = "zero"
                element_spacing_option = "uniform"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("vperp"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="gausslegendre_pseudospectral",
                                                     bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)

                f = @. exp(-x.grid^2)
                expected_d2f = @. 4.0*(x.grid^2 - 1.0)*exp(-x.grid^2)
                # create array for the derivative d2f/dx2
                d2f = similar(f)

                # differentiate f
                laplacian_derivative!(d2f, f, x, spectral)

                @test isapprox(d2f, expected_d2f, rtol=rtol, atol=1.e-10,
                               norm=maxabs_norm)
            end
        end
        
        @testset "GaussLegendre pseudospectral cylindrical laplacian ODE solve, zero" verbose=false begin
            @testset "$nelement $ngrid" for (nelement, ngrid, rtol) ∈
                    (
                     (1, 8, 5.e-2),
                     (1, 9, 3.e-2),
                     (1, 10, 4.e-3),
                     (1, 11, 3.e-3),
                     (1, 12, 6.e-4),
                     (1, 13, 4.e-4),
                     (1, 14, 2.e-4),
                     (1, 15, 5.e-5),
                     (1, 16, 3.e-5),
                     (1, 17, 9.e-6),
                     
                     (2, 6, 8.e-3),
                     (2, 7, 4.e-3),
                     (2, 8, 4.e-4),
                     (2, 9, 2.e-4),
                     (2, 10, 5.e-5),
                     (2, 11, 1.e-5),
                     (2, 12, 5.e-6),
                     (2, 13, 4.e-7),
                     (2, 14, 4.e-7),
                     (2, 15, 5.e-8),
                     (2, 16, 2.e-8),
                     (2, 17, 5.e-9),
                     
                     (3, 6, 2.e-3),
                     (3, 7, 2.e-4),
                     (3, 8, 5.e-5),
                     (3, 9, 3.e-6),
                     (3, 10, 2.e-6),
                     (3, 11, 3.e-7),
                     (3, 12, 6.e-8),
                     (3, 13, 9.e-9),
                     (3, 14, 2.e-9),
                     (3, 15, 4.e-10),
                     (3, 16, 3.e-11),
                     (3, 17, 1.e-11),
                     
                     (4, 5, 1.e-3),
                     (4, 6, 8.e-5),
                     (4, 7, 3.e-5),
                     (4, 8, 3.e-6),
                     (4, 9, 6.e-7),
                     (4, 10, 8.e-8),
                     (4, 11, 2.e-8),
                     (4, 12, 3.e-9),
                     (4, 13, 3.e-10),
                     (4, 14, 5.e-11),
                     (4, 15, 3.e-12),
                     (4, 16, 2.e-12),
                     (4, 17, 1.e-12),
                     
                     (5, 5, 4.e-4),
                     (5, 6, 3.e-5),
                     (5, 7, 6.e-6),
                     (5, 8, 4.e-7),
                     (5, 9, 9.e-8),
                     (5, 10, 4.e-9),
                     (5, 11, 2.e-9),
                     (5, 12, 8.e-11),
                     (5, 13, 3.e-11),
                     (5, 14, 1.e-12),
                     (5, 15, 4.e-13),
                     (5, 16, 4.e-13),
                     (5, 17, 2.e-12),
                     )

                # define inputs needed for the test
                L = 6.0
                bc = "zero"
                element_spacing_option = "uniform"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("vperp"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="gausslegendre_pseudospectral",
                                                     bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)
                expected_f = @. exp(-x.grid^2)
                # Test solver for
                #   Laplacian_f = 1/x * d/dx(x*df/dx)
                Laplacian_f = @. 4.0*(x.grid^2 - 1.0)*exp(-x.grid^2)
                # create array for the numerical solution
                f = similar(expected_f)
                # create array for RHS vector b
                b = similar(expected_f)
                # solve for f
                mul!(b,spectral.mass_matrix,Laplacian_f)
                # Dirichlet zero BC at upper endpoint
                b[end] = 0.0
                # solve ODE
                ldiv!(f,spectral.L_matrix_lu,b)
                #err = maximum(abs.(f.-expected_f))
                #println("$nelement $ngrid $err")
                @test isapprox(f, expected_f, rtol=rtol, atol=1.e-10,
                               norm=maxabs_norm)
            end
        end
        
        @testset "GaussLegendre pseudospectral second derivative ODE solve, zero" verbose=false begin
            @testset "$nelement $ngrid" for (nelement, ngrid, rtol) ∈
                    (
                     (1, 23, 1.e-2),
                     (1, 24, 5.e-3),
                     (1, 25, 3.e-3),
                     (1, 26, 2.e-3),
                     (1, 27, 1.e-3),
                     (1, 28, 6.e-4),
                     (1, 29, 5.e-4),
                     (1, 30, 3.e-4),
                     (1, 31, 2.e-4),
                     (1, 32, 8.e-5),
                     
                     (2, 9, 3.e-2),
                     (2, 10, 2.e-2),
                     (2, 11, 4.e-3),
                     (2, 12, 1.e-3),
                     (2, 13, 4.e-4),
                     (2, 14, 2.e-4),
                     (2, 15, 9.e-5),
                     (2, 16, 3.e-5),
                     (2, 17, 7.e-6),
                     
                     (3, 9, 2.e-2),
                     (3, 10, 2.e-3),
                     (3, 11, 6.e-4),
                     (3, 12, 2.e-4),
                     (3, 13, 7.e-5),
                     (3, 14, 1.e-5),
                     (3, 15, 9.e-6),
                     (3, 16, 9.e-7),
                     (3, 17, 1.e-6),
                     
                     (4, 7, 3.e-3),
                     (4, 8, 8.e-4),
                     (4, 9, 2.e-4),
                     (4, 10, 4.e-5),
                     (4, 11, 2.e-5),
                     (4, 12, 4.e-6),
                     (4, 13, 9.e-7),
                     (4, 14, 4.e-7),
                     (4, 15, 3.e-8),
                     (4, 16, 3.e-8),
                     (4, 17, 4.e-9),
                     
                     (5, 8, 2.e-4),
                     (5, 9, 7.e-5),
                     (5, 10, 7.e-6),
                     (5, 11, 4.e-6),
                     (5, 12, 3.e-7),
                     (5, 13, 3.e-7),
                     (5, 14, 9.e-9),
                     (5, 15, 1.e-8),
                     (5, 16, 3.e-10),
                     (5, 17, 4.e-10),
                     )

                # define inputs needed for the test
                L = 12.0
                bc = "zero"
                element_spacing_option = "uniform"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("vpa"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="gausslegendre_pseudospectral",
                                                     bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)
                expected_f = @. exp(-x.grid^2)
                # Test solver for
                #   d2f = d^2f/dx^2
                d2f = @. (4.0*x.grid^2 - 2.0)*exp(-x.grid^2)
                # create array for the numerical solution
                f = similar(expected_f)
                # create array for RHS vector b
                b = similar(expected_f)
                # solve for f
                mul!(b,spectral.mass_matrix,d2f)
                # Dirichlet zero BC at lower and upper endpoint
                b[1] = 0.0
                b[end] = 0.0
                # solve ODE
                ldiv!(f,spectral.L_matrix_lu,b)
                #err = maximum(abs.(f.-expected_f))
                #maxfe = maximum(expected_f)
                #maxf = maximum(f)
                #minf = minimum(f)
                #println("$nelement $ngrid $err $maxfe $maxf $minf")
                @test isapprox(f, expected_f, rtol=rtol, atol=1.e-10,
                               norm=maxabs_norm)
            end
        end
        
        @testset "GaussLegendre pseudospectral second derivative ODE solve, periodic" verbose=false begin
            @testset "$nelement $ngrid" for (nelement, ngrid, rtol) ∈
                    (
                     (1, 10, 6.e-6),
                     (1, 11, 2.e-6),
                     (1, 12, 8.e-8),
                     (1, 13, 2.e-8),
                     (1, 14, 1.e-9),
                     (1, 15, 2.e-10),
                     (1, 16, 9.e-12),
                     (1, 17, 2.e-12),
                     
                     (2, 9, 5.e-8),
                     (2, 10, 6.e-9),
                     (2, 11, 3.e-10),
                     (2, 12, 3.e-11),
                     (2, 13, 7.e-13),
                     (2, 14, 1.e-13),
                     (2, 15, 1.e-13),
                     (2, 16, 1.e-13),
                     (2, 17, 1.e-13),
                     
                     (3, 9, 2.e-9),
                     (3, 10, 7.e-11),
                     (3, 11, 4.e-12),
                     (3, 12, 3.e-13),
                     (3, 13, 3.e-13),
                     (3, 14, 1.e-13),
                     (3, 15, 1.e-13),
                     (3, 16, 3.e-13),
                     (3, 17, 1.e-13),
                     
                     (4, 7, 5.e-8),
                     (4, 8, 3.e-9),
                     (4, 9, 8.e-11),
                     (4, 10, 4.e-12),
                     (4, 11, 3.e-13),
                     (4, 12, 1.e-13),
                     (4, 13, 3.e-13),
                     (4, 14, 1.e-13),
                     (4, 15, 1.e-13),
                     (4, 16, 3.e-13),
                     (4, 17, 3.e-13),
                     
                     (5, 8, 5.e-10),
                     (5, 9, 9.e-12),
                     (5, 10, 5.e-13),
                     (5, 11, 3.e-13),
                     (5, 12, 1.e-13),
                     (5, 13, 5.e-13),
                     (5, 14, 1.e-13),
                     (5, 15, 2.e-13),
                     (5, 16, 5.e-13),
                     (5, 17, 5.e-13),
                     )

                # define inputs needed for the test
                L = 1.0
                bc = "periodic"
                element_spacing_option = "uniform"
                phase = pi/3.0
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("vpa"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="gausslegendre_pseudospectral",
                                                     bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)
                expected_f = @. sin((2.0*pi*x.grid/x.L) + phase)
                # Test solver for
                #   d2f = d^2f/dx^2
                d2f = @. -((2.0*pi/x.L)^2)*sin((2.0*pi*x.grid/x.L)+phase)
                # create array for the numerical solution
                f = similar(expected_f)
                # create array for RHS vector b
                b = similar(expected_f)
                # solve for f
                mul!(b,spectral.mass_matrix,d2f)
                # Dirichlet zero BC at lower endpoint, periodic bc at upper endpoint
                b[1] = sin((2.0*pi*x.grid[1]/x.L) + phase) # fixes constant piece of solution
                b[end] = 0.0 # makes sure periodicity is enforced
                # solve ODE
                ldiv!(f,spectral.L_matrix_lu,b)
                #err = maximum(abs.(f.-expected_f))
                #maxfe = maximum(expected_f)
                #maxf = maximum(f)
                #minf = minimum(f)
                #println("$nelement $ngrid $err $maxfe $maxf $minf")
                @test isapprox(f, expected_f, rtol=rtol, atol=1.e-10,
                               norm=maxabs_norm)
            end
        end

        @testset "Fourier pseudospectral second derivatives (4 argument), periodic" verbose=false begin
            @testset "$ngrid" for (ngrid, rtol) ∈
                    (
                     (5, 2.e-13),
                     (6, 2.e-13),
                     (7, 2.e-13),
                     (8, 2.e-13),
                     (9, 2.e-13),
                     (10, 2.e-13),
                     (11, 2.e-13),
                     (12, 2.e-13),
                     (13, 2.e-13),
                     (14, 2.e-13),
                     (15, 2.e-13),
                     (16, 2.e-13),
                     (17, 2.e-13),
                     (18, 2.e-13),
                     (19, 2.e-13),
                     (20, 2.e-13),
                     (21, 2.e-13),
                     (22, 2.e-13),
                     (23, 2.e-13),
                     (24, 2.e-13),
                     (25, 2.e-13),
                     (26, 2.e-13),
                     (27, 2.e-13),
                     (28, 2.e-13),
                     (29, 2.e-13),
                     (30, 2.e-13),
                     (31, 2.e-13),
                     (32, 2.e-13),
                     (33, 2.e-13),
                    )

                # define inputs needed for the test
                nelement = 1
                L = 6.0
                bc = "periodic"
                element_spacing_option = "uniform"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate("coord"; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="fourier_pseudospectral",
                                                     cheb_option="", bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)

                offset = randn(rng)
                phase = 0.42
                f = @. sinpi(2.0 * x.grid / L + phase) + offset
                expected_d2f = @. -4.0 * π^2 / L^2 * sinpi(2.0 * x.grid / L + phase)

                # create array for the derivative d2f/dx2
                d2f = similar(f)

                # differentiate f
                second_derivative!(d2f, f, x, spectral)

                @test isapprox(d2f, expected_d2f, rtol=rtol, atol=1.e-10,
                               norm=maxabs_norm)
                @test d2f[1] == d2f[end]
            end
        end

        @testset "Indefinite line integration: spectral" verbose=false begin
            @testset "$nelement $ngrid" for discretization in ("gausslegendre_pseudospectral", "chebyshev_pseudospectral"), name in ("z","vperp"), (nelement, ngrid, rtol) ∈
                    (
                     (1, 8, 1.e-3),
                     (1, 9, 7.e-5),
                     (1, 10, 5.e-5),
                     (1, 11, 8.e-7),
                     (1, 12, 5.e-7),
                     (1, 13, 8.e-9),
                     (1, 14, 5.e-9),
                     (1, 15, 1.e-10),
                     (1, 16, 5.e-10),
                     (1, 17, 5.e-10),
                     
                     (2, 6, 2.e-4),
                     (2, 7, 5.e-5),
                     (2, 8, 1.e-6),
                     (2, 9, 5.e-7),
                     (2, 10, 5.e-9),
                     (2, 11, 1.e-9),
                     (2, 12, 5.e-10),
                     (2, 13, 5.e-10),
                     (2, 14, 5.e-10),
                     (2, 15, 5.e-10),
                     (2, 16, 5.e-10),
                     (2, 17, 5.e-10),
                     
                     (3, 6, 5.e-5),
                     (3, 7, 1.e-6),
                     (3, 8, 8.e-8),
                     (3, 9, 5.e-9),
                     (3, 10, 5.e-10),
                     (3, 11, 5.e-10),
                     (3, 12, 5.e-10),
                     (3, 13, 5.e-10),
                     (3, 14, 5.e-10),
                     (3, 15, 5.e-10),
                     (3, 16, 5.e-10),
                     (3, 17, 5.e-8),
                     
                     (4, 5, 6.e-5),
                     (4, 6, 5.e-6),
                     (4, 7, 1.e-7),
                     (4, 8, 5.e-9),
                     (4, 9, 1.e-9),
                     (4, 10, 1.e-10),
                     (4, 11, 8.e-10),
                     (4, 12, 8.e-10),
                     (4, 13, 8.e-10),
                     (4, 14, 8.e-10),
                     (4, 15, 8.e-10),
                     (4, 16, 8.e-8),
                     (4, 17, 8.e-8),
                     
                     (5, 5, 4.e-5),
                     (5, 6, 8.e-7),
                     (5, 7, 5.e-8),
                     (5, 8, 1.e-9),
                     (5, 9, 8.e-10),
                     (5, 10, 5.e-10),
                     (5, 11, 8.e-10),
                     (5, 12, 4.e-10),
                     (5, 13, 2.e-10),
                     (5, 14, 2.e-10),
                     (5, 15, 8.e-10),
                     (5, 16, 8.e-10),
                     (5, 17, 8.e-10),
                     )

                # define inputs needed for the test
                L = 6.0
                bc = "zero"
                element_spacing_option = "uniform"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate(name; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization=discretization,
                                                     bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)
                xllim = x.element_boundaries[1] # lower endpoint
                f = @. cos(x.grid - xllim)
                expected_pf = @. sin(x.grid - xllim)
                # create array for the indefinite integral pf
                pf = similar(f)

                # differentiate f
                indefinite_integral!(pf, f, x, spectral)
                @test isapprox(pf, expected_pf, rtol=rtol, atol=1.e-10,
                               norm=maxabs_norm)
            end
        end

        @testset "Indefinite line integration: finite differences" verbose=false begin
            @testset "$nelement $ngrid" for name in ("z","vperp"), (nelement, ngrid, rtol) ∈
                    (
                        (1, 8, 0.092),
                        (1, 9, 0.071),
                        (1, 10, 0.056),
                        (1, 11, 0.045),
                        (1, 12, 0.037),
                        (1, 13, 0.031),
                        (1, 14, 0.027),
                        (1, 15, 0.023),
                        (1, 16, 0.02),
                        (1, 17, 0.018),
                        (2, 6, 0.045),
                        (2, 7, 0.031),
                        (2, 8, 0.023),
                        (2, 9, 0.018),
                        (2, 10, 0.014),
                        (2, 11, 0.011),
                        (2, 12, 0.0093),
                        (2, 13, 0.0078),
                        (2, 14, 0.0067),
                        (2, 15, 0.0057),
                        (2, 16, 0.005),
                        (2, 17, 0.0044),
                        (3, 6, 0.02),
                        (3, 7, 0.014),
                        (3, 8, 0.01),
                        (3, 9, 0.0078),
                        (3, 10, 0.0062),
                        (3, 11, 0.005),
                        (3, 12, 0.0041),
                        (3, 13, 0.0035),
                        (3, 14, 0.003),
                        (3, 15, 0.0026),
                        (3, 16, 0.0022),
                        (3, 17, 0.002),
                        (4, 5, 0.018),
                        (4, 6, 0.011),
                        (4, 7, 0.0078),
                        (4, 8, 0.0057),
                        (4, 9, 0.0044),
                        (4, 10, 0.0035),
                        (4, 11, 0.0028),
                        (4, 12, 0.0023),
                        (4, 13, 0.002),
                        (4, 14, 0.0017),
                        (4, 15, 0.0014),
                        (4, 16, 0.0013),
                        (4, 17, 0.0011),
                        (5, 5, 0.011),
                        (5, 6, 0.0072),
                        (5, 7, 0.005),
                        (5, 8, 0.0037),
                        (5, 9, 0.0028),
                        (5, 10, 0.0022),
                        (5, 11, 0.0018),
                        (5, 12, 0.0015),
                        (5, 13, 0.0013),
                        (5, 14, 0.0011),
                        (5, 15, 0.00092),
                        (5, 16, 0.0008),
                        (5, 17, 0.0007),
                     )

                # define inputs needed for the test
                L = 6.0
                bc = "zero"
                element_spacing_option = "uniform"
                # create the coordinate struct 'x'
                # This test runs effectively in serial, so implicitly uses
                # `ignore_MPI=true` to avoid errors due to communicators not being fully
                # set up.
                x, spectral = define_test_coordinate(name; ngrid=ngrid,
                                                     nelement=nelement, L=L,
                                                     discretization="finite_difference",
                                                     bc=bc,
                                                     element_spacing_option=element_spacing_option,
                                                     collision_operator_dim=false)
                xllim = x.element_boundaries[1] # lower endpoint
                f = @. cos(x.grid - xllim)
                expected_pf = @. sin(x.grid - xllim)
                # create array for the indefinite integral pf
                pf = similar(f)

                # differentiate f
                indefinite_integral!(pf, f, x, spectral)
                @test isapprox(pf, expected_pf, rtol=rtol, atol=1.e-10,
                               norm=maxabs_norm)
            end
        end
    end
end

end # CalculusTests


using .CalculusTests

CalculusTests.runtests()
