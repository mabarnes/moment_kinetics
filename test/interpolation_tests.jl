module InterpolationTests

include("setup.jl")

using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.interpolation:
    interpolate_to_grid_1d, interpolate_to_grid_z, interpolate_to_grid_vpa

using MPI

# periodic test function
# returns an array whose shape is the outer product of the 2nd, 3rd, ... arguments
test_function(L, coords...) =
    [cospi(2.0*sum(x)/L)*exp(-sinpi(2.0*sum(x)/L)) for x in Iterators.product(coords...)]

println("interpolation tests")

# define inputs needed for the test
ngrid = 33
L = 6.0
bc = "periodic"
# fd_option and adv_input not actually used so given values unimportant
fd_option = ""
adv_input = advection_input("default", 1.0, 0.0, 0.0)

function runtests()
    @testset "interpolation" verbose=use_verbose begin
        @testset "$discretization, $ntest, $nelement, $zlim" for
                (discretization, element_spacing_option, rtol) ∈
                    (("finite_difference", "uniform", 1.e-5), ("chebyshev_pseudospectral", "uniform", 1.e-8),
                    ("chebyshev_pseudospectral", "sqrt", 1.e-8)),
                    ntest ∈ (3, 14), nelement ∈ (2, 8), zlim ∈ (L/2.0, L/5.0)

            # create the 'input' struct containing input info needed to create a coordinate
			nelement_local = nelement
			nrank_per_block = 0 # dummy value
			irank = 0 # dummy value
			comm = MPI.COMM_NULL # dummy value
            #element_spacing_option = "uniform"
			input = grid_input("coord", ngrid, nelement,
				nelement_local, nrank_per_block, irank, L,
				discretization, fd_option, bc, adv_input, comm,
                element_spacing_option)
            # create the coordinate struct 'z'
            z, spectral = define_coordinate(input)

            test_grid = [z for z in range(-zlim, zlim, length=ntest)]

            @testset "1d" begin
                # create array for the function f(z) to be interpolated
                f = test_function(z.L, z.grid)

                # create expected output
                test_grid = [z for z in range(-zlim, zlim, length=ntest)]
                expected = test_function(z.L, test_grid)

                @test isapprox(interpolate_to_grid_1d(test_grid, f, z, spectral),
                               expected, rtol=rtol, atol=1.e-14)
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
    end
end

end # InterpolationTests


using .InterpolationTests

InterpolationTests.runtests()
