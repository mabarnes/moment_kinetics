module WallBC

# Regression test using wall boundary conditions. Runs to steady state and then
# checks phi profile against saved reference output.

include("setup.jl")

using Base.Filesystem: tempname
using MPI

using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.interpolation: interpolate_to_grid_z
using moment_kinetics.load_data: open_readonly_output_file
using moment_kinetics.load_data: load_fields_data,
                                 load_pdf_data, load_time_data,
                                 load_species_data
using moment_kinetics.utils: merge_dict_with_kwargs!

# default inputs for tests
test_input_finite_difference = OptionsDict("composition" => OptionsDict("n_ion_species" => 1,
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
                                           "vz_IC_neutral_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                                                    "density_amplitude" => 1.0,
                                                                                    "density_phase" => 0.0,
                                                                                    "upar_amplitude" => 0.0,
                                                                                    "upar_phase" => 0.0,
                                                                                    "temperature_amplitude" => 0.0,
                                                                                    "temperature_phase" => 0.0),
                                           "output" => OptionsDict("run_name" => "finite_difference"),
                                           "evolve_moments" => OptionsDict("density" => false,
                                                                           "parallel_flow" => false,
                                                                           "parallel_pressure" => false,
                                                                           "moments_conservation" => false),
                                           "reactions" => OptionsDict("charge_exchange_frequency" => 2.0,
                                                                      "ionization_frequency" => 2.0),
                                           "timestepping" => OptionsDict("nstep" => 10000,
                                                                         "dt" => 1.0e-5,
                                                                         "nwrite" => 100,
                                                                         "split_operators" => false),
                                           "r" => OptionsDict("ngrid" => 1,
                                                              "nelement" => 1,
                                                              "bc" => "periodic",
                                                              "discretization" => "finite_difference"),
                                           "z" => OptionsDict("ngrid" => 200,
                                                              "nelement" => 1,
                                                              "bc" => "wall",
                                                              "discretization" => "finite_difference",
                                                              "element_spacing_option" => "uniform"),
                                           "vpa" => OptionsDict("ngrid" => 400,
                                                                "nelement" => 1,
                                                                "L" => 8.0,
                                                                "bc" => "periodic",
                                                                "discretization" => "finite_difference"),
                                           "vz" => OptionsDict("ngrid" => 400,
                                                               "nelement" => 1,
                                                               "L" => 8.0,
                                                               "bc" => "periodic",
                                                               "discretization" => "finite_difference"),
                                          )

test_input_chebyshev = recursive_merge(test_input_finite_difference,
                                       OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral"),
                                                   "z" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                      "ngrid" => 9,
                                                                      "nelement" => 2,
                                                                      "element_spacing_option" => "uniform"),
                                                   "vpa" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                        "ngrid" => 17,
                                                                        "nelement" => 10),
                                                   "vz" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                       "ngrid" => 17,
                                                                       "nelement" => 10),
                                                  ))
                                  
test_input_chebyshev_sqrt_grid_odd = recursive_merge(test_input_finite_difference,
                                                     OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral"),
                                                                 "z" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                                    "ngrid" => 9,
                                                                                    "nelement" => 5, # minimum nontrival nelement (odd)
                                                                                    "element_spacing_option" => "sqrt"),
                                                                 "vpa" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                                      "ngrid" => 17,
                                                                                      "nelement" => 10),
                                                                 "vz" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                                     "ngrid" => 17,
                                                                                     "nelement" => 10),
                                                                ))
test_input_chebyshev_sqrt_grid_even = recursive_merge(test_input_finite_difference,
                                                      OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral"),
                                                                  "z" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                                     "ngrid" => 9,
                                                                                     "nelement" => 6, # minimum nontrival nelement (even)
                                                                                     "element_spacing_option" => "sqrt"),
                                                                  "vpa" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                                       "ngrid" => 17,
                                                                                       "nelement" => 10),
                                                                  "vz" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                                      "ngrid" => 17,
                                                                                      "nelement" => 10),
                                                                 ))

# Reference output interpolated onto a common set of points for comparing
# different discretizations, taken from a Chebyshev run with z_grid=9,
# z_nelement=8, nstep=40000, dt=0.00025
cross_compare_points = collect(LinRange(-0.5, 0.5, 7))
cross_compare_phi = [-1.168944495073113, -0.7419936034715402, -0.7028946661416742,
                     -0.6917192497940869, -0.7028946661416738, -0.7419936034715399,
                     -1.1689444950731127]

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
    name = input["output"]["run_name"] * ", with element spacing: " * input["z"]["element_spacing_option"]
    if length(args) > 0
        name = string(name, "_", (string(k, "-", v, "_") for (k, v) in args)...)

        # Remove trailing "_"
        name = chop(name)
    end

    # Provide some progress info
    println("    - testing ", name)

    # Update default inputs with values to be changed
    merge_dict_with_kwargs!(input; args...)
    input["output"]["run_name"] = name

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

            path = joinpath(realpath(input["output"]["base_directory"]), name, name)

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
        z, z_spectral = define_coordinate(test_input, "z"; ignore_MPI=true)

        # Cross comparison of all discretizations to same benchmark
        if test_input["z"]["element_spacing_option"] == "uniform"
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
            test_input_finite_difference["output"]["base_directory"] = test_output_directory
            run_test(test_input_finite_difference, nothing, 2.e-3)
        end

        @testset "Chebyshev uniform" begin
            test_input_chebyshev["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev,
                     [-1.168944495073113, -0.747950464799219, -0.6947560093910274,
                      -0.6917252594440765, -0.7180152693147238, -0.9980114030684668],
                     2.e-3)
        end
        
        @testset "Chebyshev sqrt grid odd" begin
            test_input_chebyshev_sqrt_grid_odd["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_sqrt_grid_odd,
                     [-1.2047298844053338, -0.9431378244038217, -0.8084332486925859,
                      -0.7812620574297168, -0.7233303715713063, -0.700387877851292,
                      -0.695727529425101, -0.6933149075958859, -0.6992504158371133,
                      -0.7115788158947632, -0.7596015227027635, -0.7957765261319207,
                      -0.876303296785542, -1.1471244373220089], 2.e-3)
        end
        @testset "Chebyshev sqrt grid even" begin
            test_input_chebyshev_sqrt_grid_even["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_sqrt_grid_even,
                     [-1.213617044609117, -1.0054529856551995, -0.8714447622540997,
                      -0.836017704148175, -0.7552111126205924, -0.7264644278204795,
                      -0.7149147557607726, -0.6950077350352664, -0.6923365041825125,
                      -0.6950077350352668, -0.7149147557607729, -0.7264644278204795,
                      -0.7552111126205917, -0.8360177041481733, -0.8714447622540994,
                      -1.0054529856552, -1.213617044609118], 2.e-3)
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
