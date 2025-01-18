module InterpolationTests

include("setup.jl")

using moment_kinetics.array_allocation: allocate_float
using moment_kinetics.coordinates: define_test_coordinate
using moment_kinetics.interpolation:
    interpolate_to_grid_1d, fill_1d_interpolation_matrix!, interpolate_to_grid_z,
    interpolate_to_grid_vpa, interpolate_symmetric!, fill_interpolate_symmetric_matrix!
using moment_kinetics.type_definitions

using MPI

# periodic test function
# returns an array whose shape is the outer product of the 2nd, 3rd, ... arguments
test_function(L, coords...) =
    MKArray([cospi(2.0*sum(x)/L)*exp(-sinpi(2.0*sum(x)/L)) for x in Iterators.product(coords...)])

test_function_first_derivative(L, coord) =
    MKArray([-2.0*π/L*sinpi(2.0*x/L)*exp(-sinpi(2.0*x/L)) - 2.0*π/L*cospi(2.0*x/L)^2*exp(-sinpi(2.0*x/L)) for x in coord])

println("interpolation tests")

# define inputs needed for the test
ngrid = 33
L = 6.0
bc = "periodic"

function runtests()
    @testset "interpolation" verbose=use_verbose begin
        @testset "$discretization, $element_spacing_option, $ntest, $nelement, $zlim" for
                (discretization, element_spacing_option, rtol) ∈
                    (("finite_difference", "uniform", 1.e-5), ("chebyshev_pseudospectral", "uniform", 1.e-8),
                    ("chebyshev_pseudospectral", "sqrt", 1.e-8),
                    ("gausslegendre_pseudospectral", "uniform", 1.e-8),
                    ("gausslegendre_pseudospectral", "sqrt", 1.e-8)),
                    ntest ∈ (3, 14), nelement ∈ (2, 8), zlim ∈ (L/2.0, L/5.0)

            # create the 'input' struct containing input info needed to create a coordinate
            nelement_local = nelement
            cheb_option = "FFT"
            input = OptionsDict("name"=>"coord", "ngrid"=>ngrid, "nelement"=>nelement,
                                "L"=>L, "discretization"=>discretization,
                                "cheb_option"=>cheb_option, "bc"=>bc,
                                "element_spacing_option"=>element_spacing_option)
            # create the coordinate struct 'z'
            # This test runs effectively in serial, so use `ignore_MPI=true` to avoid
            # errors due to communicators not being fully set up.
            z, spectral = define_test_coordinate(input; ignore_MPI=true,
                                                 collision_operator_dim=false)

            test_grid = [z for z in range(-zlim, zlim, length=ntest)]

            @testset "1d" begin
                # create array for the function f(z) to be interpolated
                f = test_function(z.L, z.grid)

                # create expected output
                test_grid = [z for z in range(-zlim, zlim, length=ntest)]
                expected = test_function(z.L, test_grid)

                @test isapprox(interpolate_to_grid_1d(test_grid, f, z, spectral),
                               expected, rtol=rtol, atol=1.e-14)

                if discretization != "finite_difference"
                    # Last element of test_grid is on the last grid point.
                    # interpolate_to_grid_1d() will interpret this as being just outside
                    # the grid and give the derivative of the extrapolation function
                    # there, which is not what we want to test, so skip that point.
                    @test isapprox(interpolate_to_grid_1d(test_grid[1:end-1], f, z, spectral, Val(1)),
                                   test_function_first_derivative(z.L, test_grid[1:end-1]),
                                   rtol=rtol, atol=1.e-14)
                end

                if discretization == "gausslegendre_pseudospectral"
                    @testset "matrix" begin
                        interp_matrix = allocate_float(length(test_grid), z.n)
                        interp_matrix .= 0.0
                        fill_1d_interpolation_matrix!(interp_matrix, test_grid, z, spectral)

                        @test isapprox(interp_matrix * f, expected, rtol=rtol, atol=1.e-14)
                    end
                end
            end

            y = [y for y in range(5.0, 10.0, length=3)]
            @testset "2d z" begin
                # create array for the function f(z) to be interpolated
                f = test_function(z.L, z.grid, y)

                # create expected output
                expected = test_function(z.L, test_grid, y)

                @test isapprox(interpolate_to_grid_z(test_grid, f, z, spectral),
                               expected, rtol=rtol, atol=1.e-14)
            end

            x = [x for x in range(11.0, 13.0, length=4)]
            @testset "3d z" begin
                # create array for the function f(z) to be interpolated
                f = test_function(z.L, y, z.grid, x)

                # create expected output
                expected = test_function(z.L, y, test_grid, x)

                @test isapprox(interpolate_to_grid_z(test_grid, f, z, spectral),
                               expected, rtol=rtol, atol=1.e-14)
            end

            vpa = z
            @testset "3d vpa" begin
                # create array for the function f(z) to be interpolated
                f = test_function(vpa.L, vpa.grid, y, x)

                # create expected output
                expected = test_function(vpa.L, test_grid, y, x)

                @test isapprox(interpolate_to_grid_vpa(test_grid, f, vpa, spectral),
                               expected, rtol=rtol, atol=1.e-14)
            end
        end

        @testset "symmetric interpolation" begin
            @testset "lower to upper $nx" for nx ∈ 4:10
                rtol = 0.2 ^ nx

                ix = collect(1:nx)
                x = @. 1.8 * (ix - 1) / (nx - 1) - 1.23
                first_positive_ind = searchsortedlast(x, 0.0) + 1
                f = cos.(x)
                dfdx = -sin.(x)

                expected = f[first_positive_ind:end]

                result = zeros(nx - first_positive_ind + 1)
                @views interpolate_symmetric!(result, x[first_positive_ind:end],
                                              f[1:first_positive_ind-1],
                                              x[1:first_positive_ind-1])

                @test isapprox(result, expected; rtol=rtol, atol=1.0e-14)

                expected_deriv = dfdx[first_positive_ind:end]

                result_deriv = zeros(nx - first_positive_ind + 1)
                @views interpolate_symmetric!(result_deriv, x[first_positive_ind:end],
                                              f[1:first_positive_ind-1],
                                              x[1:first_positive_ind-1], Val(1))

                @test isapprox(result_deriv, expected_deriv; rtol=2.0*rtol, atol=1.0e-14)

                @testset "matrix" begin
                    interp_matrix = allocate_float(nx - first_positive_ind + 1,
                                                   first_positive_ind - 1)
                    interp_matrix .= 0.0
                    fill_interpolate_symmetric_matrix!(interp_matrix,
                                                       x[first_positive_ind:end],
                                                       x[1:first_positive_ind-1])

                    @test isapprox(interp_matrix * f[1:first_positive_ind-1], expected,
                                   rtol=rtol, atol=1.0e-14)
                end
            end

            @testset "upper to lower $nx" for nx ∈ 4:10
                rtol = 0.2 ^ nx

                ix = MKArray(collect(1:nx))
                x = @. 1.8 * (ix - 1) / (nx - 1) - 0.57
                first_positive_ind = searchsortedlast(x, 0.0) + 1
                f = cos.(x)
                dfdx = -sin.(x)

                expected = f[1:first_positive_ind-1]

                result = zeros(first_positive_ind-1)
                @views interpolate_symmetric!(result, x[1:first_positive_ind-1],
                                              f[first_positive_ind:end],
                                              x[first_positive_ind:end])

                @test isapprox(result, expected; rtol=rtol, atol=1.0e-14)

                expected_deriv = dfdx[1:first_positive_ind-1]

                result_deriv = zeros(first_positive_ind-1)
                @views interpolate_symmetric!(result_deriv, x[1:first_positive_ind-1],
                                              f[first_positive_ind:end],
                                              x[first_positive_ind:end], Val(1))

                @test isapprox(result_deriv, expected_deriv; rtol=2.0*rtol, atol=1.0e-14)

                @testset "matrix" begin
                    interp_matrix = allocate_float(first_positive_ind - 1,
                                                   nx - first_positive_ind + 1)
                    interp_matrix .= 0.0
                    fill_interpolate_symmetric_matrix!(interp_matrix,
                                                       x[1:first_positive_ind-1],
                                                       x[first_positive_ind:end])

                    @test isapprox(interp_matrix * f[first_positive_ind:end], expected,
                                   rtol=rtol, atol=1.0e-14)
                end
            end
        end
    end
end

end # InterpolationTests


using .InterpolationTests

InterpolationTests.runtests()
