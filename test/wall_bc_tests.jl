module WallBC

# Regression test using wall boundary conditions. Runs to steady state and then
# checks phi profile against saved reference output.

include("setup.jl")

using Base.Filesystem: tempname
using MPI
using TimerOutputs

using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.interpolation: interpolate_to_grid_z
using moment_kinetics.load_data: open_readonly_output_file
using moment_kinetics.load_data: load_fields_data,
                                 load_pdf_data, load_time_data,
                                 load_species_data

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
                                    "z_element_spacing_option" => "uniform",
                                    "vpa_ngrid" => 400,
                                    "vpa_nelement" => 1,
                                    "vpa_L" => 8.0,
                                    "vpa_bc" => "periodic",
                                    "vpa_discretization" => "finite_difference",
                                    "vz_ngrid" => 400,
                                    "vz_nelement" => 1,
                                    "vz_L" => 8.0,
                                    "vz_bc" => "periodic",
                                    "vz_discretization" => "finite_difference")

test_input_chebyshev = merge(test_input_finite_difference,
                             Dict("run_name" => "chebyshev_pseudospectral",
                                  "z_discretization" => "chebyshev_pseudospectral",
                                  "z_ngrid" => 9,
                                  "z_nelement" => 2,
                                  "z_element_spacing_option" => "uniform",
                                  "vpa_discretization" => "chebyshev_pseudospectral",
                                  "vpa_ngrid" => 17,
                                  "vpa_nelement" => 10,
                                  "vz_discretization" => "chebyshev_pseudospectral",
                                  "vz_ngrid" => 17,
                                  "vz_nelement" => 10))
                                  
test_input_chebyshev_sqrt_grid_odd = merge(test_input_finite_difference,
                             Dict("run_name" => "chebyshev_pseudospectral",
                                  "z_discretization" => "chebyshev_pseudospectral",
                                  "z_ngrid" => 9,
                                  "z_nelement" => 5, # minimum nontrival nelement (odd)
                                  "z_element_spacing_option" => "sqrt",
                                  "vpa_discretization" => "chebyshev_pseudospectral",
                                  "vpa_ngrid" => 17,
                                  "vpa_nelement" => 10,
                                  "vz_discretization" => "chebyshev_pseudospectral",
                                  "vz_ngrid" => 17,
                                  "vz_nelement" => 10))
test_input_chebyshev_sqrt_grid_even = merge(test_input_finite_difference,
                             Dict("run_name" => "chebyshev_pseudospectral",
                                  "z_discretization" => "chebyshev_pseudospectral",
                                  "z_ngrid" => 9,
                                  "z_nelement" => 6, # minimum nontrival nelement (even)
                                  "z_element_spacing_option" => "sqrt",
                                  "vpa_discretization" => "chebyshev_pseudospectral",
                                  "vpa_ngrid" => 17,
                                  "vpa_nelement" => 10,
                                  "vz_discretization" => "chebyshev_pseudospectral",
                                  "vz_ngrid" => 17,
                                  "vz_nelement" => 10))

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
    name = test_input["run_name"] * ", with element spacing: " * test_input["z_element_spacing_option"]
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
            fid = open_readonly_output_file(path,"moments")

            # load species, time coordinate data
            n_ion_species, n_neutral_species = load_species_data(fid)
            ntime, time = load_time_data(fid)
            n_ion_species, n_neutral_species = load_species_data(fid)
            
            # load fields data
            phi_zrt, Er_zrt, Ez_zrt = load_fields_data(fid)

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
		nrank_per_block = 0 # dummy value
		irank = 0 # dummy value
		comm = MPI.COMM_NULL # dummy value
        element_spacing_option = "uniform"
        input = grid_input("coord", test_input["z_ngrid"], test_input["z_nelement"], 
						   test_input["z_nelement"], nrank_per_block, irank, 1.0,
                           test_input["z_discretization"], "", test_input["z_bc"],
                           adv_input, comm, test_input["z_element_spacing_option"])
        z, z_spectral = define_coordinate(input)

        # Cross comparison of all discretizations to same benchmark
        if test_input["z_element_spacing_option"] == "uniform" 
            # Only support this test for uniform element spacing.
            # phi is better resolved by "sqrt" spacing grid, so disagrees with benchmark data from
            # simulation with uniform element spacing.
            phi_interp = interpolate_to_grid_z(cross_compare_points, phi[:, end], z, z_spectral)
            @test isapprox(phi_interp, cross_compare_phi, rtol=tolerance, atol=1.e-15)
        end
    end
end

function runtests()

    @testset "Wall boundary conditions" verbose=use_verbose begin
        println("Wall boundary condition tests")

        @testset_skip "FD test case does not conserve density" "finite difference" begin
            run_test(test_input_finite_difference, nothing, 2.e-3)
        end

        @testset "Chebyshev uniform" begin
            run_test(test_input_chebyshev,
                     [-1.1689445031600718, -0.7479504438063098, -0.6947559936893813,
                      -0.6917252442591313, -0.7180152498764835, -0.9980114095597415],
                     2.e-3)
        end
        
        @testset "Chebyshev sqrt grid odd" begin
            run_test(test_input_chebyshev_sqrt_grid_odd,
                     [-1.2047298885671576, -0.9431378294506091, -0.8084332392927167,
                     -0.7812620422650213, -0.7233303514000929, -0.7003878610612269,
                     -0.69572751349158, -0.6933148921301019, -0.6992503992521327,
                     -0.7115787972775218, -0.7596015032228407, -0.795776514029509,
                     -0.876303297135126, -1.1471244425913258],
                     2.e-3)
        end
        @testset "Chebyshev sqrt grid even" begin
            run_test(test_input_chebyshev_sqrt_grid_even,
                     [-1.213617049279473, -1.0054529928344382, -0.871444761913497,
                     -0.836017699317097, -0.7552110924643832, -0.7264644073096705,
                     -0.7149147366621806, -0.6950077192395091, -0.6923364889119271,
                     -0.6950077192395089, -0.7149147366621814, -0.7264644073096692,
                     -0.7552110924643836, -0.8360176993170979, -0.8714447619134948,
                     -1.0054529928344376, -1.2136170492794727],
                     2.e-3)
        end
    end
end

end # WallBC


using .WallBC

WallBC.runtests()
