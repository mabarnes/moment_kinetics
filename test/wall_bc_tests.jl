module WallBC

# Regression test using wall boundary conditions. Runs to steady state and then
# checks phi profile against saved reference output.

include("setup.jl")

using Base.Filesystem: tempname
using TimerOutputs

using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.interpolation: interpolate_to_grid_z
using moment_kinetics.load_data: open_netcdf_file
using moment_kinetics.load_data: load_coordinate_data, load_fields_data,
                                 load_pdf_data

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

# default inputs for tests
test_input_finite_difference = Dict("n_ion_species" => 1,
                                    "n_neutral_species" => 1,
                                    "boltzmann_electron_response" => true,
                                    "run_name" => "finite_difference",
                                    "base_directory" => test_output_directory,
                                    "evolve_moments_density" => false,
                                    "evolve_moments_parallel_flow" => false,
                                    "evolve_moments_parallel_pressure" => false,
                                    "evolve_moments_conservation" => false,
                                    "T_e" => 1.0,
                                    "T_wall" => 1.0,
                                    "initial_density1" => 1.0,
                                    "initial_temperature1" => 1.0,
                                    "z_IC_option1" => "gaussian",
                                    "z_IC_density_amplitude1" => 0.001,
                                    "z_IC_density_phase1" => 0.0,
                                    "z_IC_upar_amplitude1" => 0.0,
                                    "z_IC_upar_phase1" => 0.0,
                                    "z_IC_temperature_amplitude1" => 0.0,
                                    "z_IC_temperature_phase1" => 0.0,
                                    "vpa_IC_option1" => "gaussian",
                                    "vpa_IC_density_amplitude1" => 1.0,
                                    "vpa_IC_density_phase1" => 0.0,
                                    "vpa_IC_upar_amplitude1" => 0.0,
                                    "vpa_IC_upar_phase1" => 0.0,
                                    "vpa_IC_temperature_amplitude1" => 0.0,
                                    "vpa_IC_temperature_phase1" => 0.0,
                                    "initial_density2" => 1.0,
                                    "initial_temperature2" => 1.0,
                                    "z_IC_option2" => "gaussian",
                                    "z_IC_density_amplitude2" => 0.001,
                                    "z_IC_density_phase2" => 0.0,
                                    "z_IC_upar_amplitude2" => 0.0,
                                    "z_IC_upar_phase2" => 0.0,
                                    "z_IC_temperature_amplitude2" => 0.0,
                                    "z_IC_temperature_phase2" => 0.0,
                                    "vpa_IC_option2" => "gaussian",
                                    "vpa_IC_density_amplitude2" => 1.0,
                                    "vpa_IC_density_phase2" => 0.0,
                                    "vpa_IC_upar_amplitude2" => 0.0,
                                    "vpa_IC_upar_phase2" => 0.0,
                                    "vpa_IC_temperature_amplitude2" => 0.0,
                                    "vpa_IC_temperature_phase2" => 0.0,
                                    "charge_exchange_frequency" => 2.0,
                                    "ionization_frequency" => 2.0,
                                    "constant_ionization_rate" => false,
                                    "nstep" => 10000,
                                    "dt" => 1.0e-5,
                                    "nwrite" => 100,
                                    "use_semi_lagrange" => false,
                                    "n_rk_stages" => 4,
                                    "split_operators" => false,
                                    "r_ngrid" => 1,
                                    "r_nelement" => 1,
                                    "r_bc" => "periodic",
                                    "r_discretization" => "finite_difference",
                                    "z_ngrid" => 200,
                                    "z_nelement" => 1,
                                    "z_bc" => "wall",
                                    "z_discretization" => "finite_difference",
                                    "vpa_ngrid" => 400,
                                    "vpa_nelement" => 1,
                                    "vpa_L" => 8.0,
                                    "vpa_bc" => "periodic",
                                    "vpa_discretization" => "finite_difference")

test_input_chebyshev = merge(test_input_finite_difference,
                             Dict("run_name" => "chebyshev_pseudospectral",
                                  "z_discretization" => "chebyshev_pseudospectral",
                                  "z_ngrid" => 9,
                                  "z_nelement" => 2,
                                  "vpa_discretization" => "chebyshev_pseudospectral",
                                  "vpa_ngrid" => 17,
                                  "vpa_nelement" => 10))

# Reference output interpolated onto a common set of points for comparing
# different discretizations, taken from a Chebyshev run with z_grid=9,
# z_nelement=8, nstep=40000, dt=0.00025
cross_compare_points = collect(LinRange(-0.5, 0.5, 7))
cross_compare_phi = [-1.1689445031600723, -0.7419935821024918, -0.7028946489842773,
                     -0.6917192346866861, -0.7028946489842764, -0.7419935821024903,
                     -1.1689445031600707]

# Not actually used in the tests, but needed for first argument of run_moment_kinetics
to = TimerOutput()

"""
Run a test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, expected_phi, tolerance; args...)
    # by passing keyword arguments to run_test, args becomes a Dict which can be used to
    # update the default inputs

    # Convert keyword arguments to a unique name
    name = test_input["run_name"]
    if length(args) > 0
        name = string(name, "_", (string(k, "-", v, "_") for (k, v) in args)...)

        # Remove trailing "_"
        name = chop(name)
    end

    # Provide some progress info
    println("    - testing ", name)

    # Convert dict from symbol keys to String keys
    modified_inputs = Dict(String(k) => v for (k, v) in args)

    # Update default inputs with values to be changed
    input = merge(test_input, modified_inputs)

    input["run_name"] = name

    # Suppress console output while running
    phi = undef
    quietoutput() do
        # run simulation
        run_moment_kinetics(to, input)
    end

    if global_rank[] == 0
        quietoutput() do
            # Load and analyse output
            #########################

            path = joinpath(realpath(input["base_directory"]), name, name)

            # open the netcdf file and give it the handle 'fid'
            fid = open_netcdf_file(path)

            # load space-time coordinate data
            nvpa, vpa, vpa_wgts, nz, z, z_wgts, Lz, nr, r, r_wgts, Lr, ntime, time = load_coordinate_data(fid)

            # load fields data
            phi_zrt = load_fields_data(fid)

            close(fid)
            
            phi = phi_zrt[:,1,:]
        end

        # Regression test
        actual_phi = phi[begin:3:end, end]
        if expected_phi == nothing
            # Error: no expected input provided
            println("data tested would be: ", actual_phi)
            @test false
        else
            @test isapprox(actual_phi, expected_phi, rtol=3.e-10, atol=1.e-15)
        end

        # Create coordinates
        #
        # create the 'input' struct containing input info needed to create a coordinate
        # adv_input not actually used in this test so given values unimportant
        adv_input = advection_input("default", 1.0, 0.0, 0.0)
        input = grid_input("coord", test_input["z_ngrid"], test_input["z_nelement"], 1.0,
                           test_input["z_discretization"], "", test_input["z_bc"],
                           adv_input)
        z, z_spectral = define_coordinate(input)

        # Cross comparison of all discretizations to same benchmark
        phi_interp = interpolate_to_grid_z(cross_compare_points, phi[:, end], z, z_spectral)
        @test isapprox(phi_interp, cross_compare_phi, rtol=tolerance, atol=1.e-15)
    end
end

function runtests()

    @testset "Wall boundary conditions" verbose=use_verbose begin
        println("Wall boundary condition tests")

        @testset_skip "FD test case does not conserve density" "finite difference" begin
            run_test(test_input_finite_difference, nothing, 2.e-3)
        end

        @testset "Chebyshev" begin
            run_test(test_input_chebyshev,
                     [-1.1689445031600718, -0.7479504438063098, -0.6947559936893813,
                      -0.6917252442591313, -0.7180152498764835, -0.9980114095597415],
                     2.e-3)
        end
    end
end

end # WallBC


using .WallBC

WallBC.runtests()
