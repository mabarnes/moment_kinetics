module WallBC

# Regression test using wall boundary conditions. Runs to steady state and then
# checks phi profile against saved reference output.

include("setup.jl")

using Base.Filesystem: tempname
using MPI

using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.input_structs: grid_input, advection_input, merge_dict_with_kwargs!
using moment_kinetics.interpolation: interpolate_to_grid_z
using moment_kinetics.load_data: open_readonly_output_file
using moment_kinetics.load_data: load_fields_data,
                                 load_pdf_data, load_time_data,
                                 load_species_data
using moment_kinetics.type_definitions: OptionsDict

# default inputs for tests
test_input_finite_difference = Dict("composition" => OptionsDict("n_ion_species" => 1,
                                                                      "n_neutral_species" => 1,
                                                                      "electron_physics" => "boltzmann_electron_response",
                                                                      "T_e" => 1.0,
                                                                      "T_wall" => 1.0),
                                    "ion_species_1" => OptionsDict("initial_density" => 1.0,
                                                                        "initial_temperature" => 1.0),
                                    "z_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                                             "density_amplitude" => 0.0,
                                                                             "density_phase" => 0.0,
                                                                             "upar_amplitude" => 0.0,
                                                                             "upar_phase" => 0.0,
                                                                             "temperature_amplitude" => 0.0,
                                                                             "temperature_phase" => 0.0),
                                    "vpa_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                                             "density_amplitude" => 1.0,
                                                                             "density_phase" => 0.0,
                                                                             "upar_amplitude" => 0.0,
                                                                             "upar_phase" => 0.0,
                                                                             "temperature_amplitude" => 0.0,
                                                                             "temperature_phase" => 0.0),
                                    "neutral_species_1" => OptionsDict("initial_density" => 1.0,
                                                                            "initial_temperature" => 1.0),
                                    "z_IC_neutral_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                                                 "density_amplitude" => 0.001,
                                                                                 "density_phase" => 0.0,
                                                                                 "upar_amplitude" => 0.0,
                                                                                 "upar_phase" => 0.0,
                                                                                 "temperature_amplitude" => 0.0,
                                                                                 "temperature_phase" => 0.0),  
                                    "vpa_IC_neutral_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                                                 "density_amplitude" => 1.0,
                                                                                 "density_phase" => 0.0,
                                                                                 "upar_amplitude" => 0.0,
                                                                                 "upar_phase" => 0.0,
                                                                                 "temperature_amplitude" => 0.0,
                                                                                 "temperature_phase" => 0.0),  
                                    "run_name" => "finite_difference",
                                    "evolve_moments_density" => false,
                                    "evolve_moments_parallel_flow" => false,
                                    "evolve_moments_parallel_pressure" => false,
                                    "evolve_moments_conservation" => false,
                                    "charge_exchange_frequency" => 2.0,
                                    "ionization_frequency" => 2.0,
                                    "constant_ionization_rate" => false,
                                    "timestepping" => OptionsDict("nstep" => 10000,
                                                                       "dt" => 1.0e-5,
                                                                       "nwrite" => 100,
                                                                       "split_operators" => false),
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

"""
Run a test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, expected_phi, tolerance; args...)
    # by passing keyword arguments to run_test, args becomes a Dict which can be used to
    # update the default inputs

    # Make a copy to make sure nothing modifies the input Dicts defined in this test
    # script.
    input = deepcopy(test_input)

    # Convert keyword arguments to a unique name
    name = input["run_name"] * ", with element spacing: " * input["z_element_spacing_option"]
    if length(args) > 0
        name = string(name, "_", (string(k, "-", v, "_") for (k, v) in args)...)

        # Remove trailing "_"
        name = chop(name)
    end

    # Provide some progress info
    println("    - testing ", name)

    # Update default inputs with values to be changed
    merge_dict_with_kwargs!(input; args...)
    input["run_name"] = name

    # Suppress console output while running
    phi = undef
    quietoutput() do
        # run simulation
        run_moment_kinetics(input)
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
        cheb_option = "FFT"
		nrank_per_block = 0 # dummy value
		irank = 0 # dummy value
		comm = MPI.COMM_NULL # dummy value
        element_spacing_option = "uniform"
        input = grid_input("coord", test_input["z_ngrid"], test_input["z_nelement"], 
                           test_input["z_nelement"], nrank_per_block, irank, 1.0,
                           test_input["z_discretization"], "", cheb_option, test_input["z_bc"],
                           adv_input, comm, test_input["z_element_spacing_option"])
        z, z_spectral = define_coordinate(input, nothing; ignore_MPI=true)

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
    # Create a temporary directory for test output
    test_output_directory = get_MPI_tempdir()

    @testset "Wall boundary conditions" verbose=use_verbose begin
        println("Wall boundary condition tests")

        @testset_skip "FD test case does not conserve density" "finite difference" begin
            test_input_finite_difference["base_directory"] = test_output_directory
            run_test(test_input_finite_difference, nothing, 2.e-3)
        end

        @testset "Chebyshev uniform" begin
            test_input_chebyshev["base_directory"] = test_output_directory
            run_test(test_input_chebyshev,
                     [-1.168944495073113, -0.747950464799219, -0.6947560093910274,
                      -0.6917252594440765, -0.7180152693147238, -0.9980114030684668],
                     2.e-3)
        end
        
        @testset "Chebyshev sqrt grid odd" begin
            test_input_chebyshev_sqrt_grid_odd["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_sqrt_grid_odd,
                     [-1.2047298844053338, -0.9431378244038217, -0.8084332486925859,
                      -0.7812620574297168, -0.7233303715713063, -0.700387877851292,
                      -0.695727529425101, -0.6933149075958859, -0.6992504158371133,
                      -0.7115788158947632, -0.7596015227027635, -0.7957765261319207,
                      -0.876303296785542, -1.1471244373220089],
                     2.e-3)
        end
        @testset "Chebyshev sqrt grid even" begin
            test_input_chebyshev_sqrt_grid_even["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_sqrt_grid_even,
                     [-1.213617044609117, -1.0054529856551995, -0.8714447622540997,
                      -0.836017704148175, -0.7552111126205924, -0.7264644278204795,
                      -0.7149147557607726, -0.6950077350352664, -0.6923365041825125,
                      -0.6950077350352668, -0.7149147557607729, -0.7264644278204795,
                      -0.7552111126205917, -0.8360177041481733, -0.8714447622540994,
                      -1.0054529856552, -1.213617044609118],
                     2.e-3)
        end
    end

    if global_rank[] == 0
        # Delete output directory to avoid using too much disk space
        rm(realpath(test_output_directory); recursive=true)
    end
end

end # WallBC


using .WallBC

WallBC.runtests()
