module Multi_Source_Tests

# Test for multi source functionality 

include("setup.jl")

using Base.Filesystem: tempname
using MPI

using moment_kinetics.interpolation: interpolate_to_grid_z
using moment_kinetics.load_data: get_run_info_no_setup, close_run_info,
                                 postproc_load_variable
using moment_kinetics.utils: merge_dict_with_kwargs!

# default inputs for tests
test_input = OptionsDict("composition" => OptionsDict("n_ion_species" => 1,
                                                      "n_neutral_species" => 0,
                                                      "electron_physics" => "boltzmann_electron_response",
                                                      "T_e" => 0.8),
                         "ion_species_1" => OptionsDict("initial_density" => 5.0,
                                                        "initial_temperature" => 1.0),
                         "z_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                             "density_amplitude" => 0.001,
                                                             "density_phase" => 0.0,
                                                             "upar_amplitude" => 1.0,
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
                         "output" => OptionsDict("run_name" => "Multi Source Test",
                                                 "display_timing_info" => false),
                         "evolve_moments" => OptionsDict("density" => true,
                                                         "parallel_flow" => true,
                                                         "parallel_pressure" => true,
                                                         "moments_conservation" => true),
                         "timestepping" => OptionsDict("nstep" => 200,
                                                       "dt" => 0.001,
                                                       "nwrite" => 100,
                                                       "type" => "SSPRK4",
                                                       "nwrite_dfns" => 100,
                                                       "steady_state_residual" => true),
                         "r" => OptionsDict("ngrid" => 1,
                                            "nelement" => 1),
                         "z" => OptionsDict("ngrid" => 5,
                                            "nelement" => 20,
                                            "bc" => "wall",
                                            "discretization" => "chebyshev_pseudospectral"),
                         "vpa" => OptionsDict("ngrid" => 6,
                                              "nelement" => 20,
                                              "L" => 16.0,
                                              "bc" => "zero",
                                              "discretization" => "chebyshev_pseudospectral",
                                              "element_spacing_option" => "coarse_tails"),
                         "ion_source_1" => OptionsDict("active" => true,
                                                     "z_profile" => "super_gaussian_4",
                                                     "z_width" => 0.275816,
                                                     "source_strength" => 5.0,
                                                     "source_T" => 1.5,
                                                     "source_type" => "Maxwellian"),
                         "ion_source_2" => OptionsDict("active" => true,
                                                     "z_profile" => "wall_exp_decay",
                                                     "z_width" => 0.15,
                                                     "source_strength" => 5.0,
                                                     "source_T" => 0.1,
                                                     "source_type" => "Maxwellian"),
                         "krook_collisions" => OptionsDict("use_krook" => true,
                                                          "frequency_option" => "reference_parameters")
                        )

if global_size[] > 2 && global_size[] % 2 == 0
    # Test using distributed-memory
    test_input["z"]["nelement_local"] = test_input["z"]["nelement"] ÷ 2
end

test_input_1 = recursive_merge(test_input,
                                    OptionsDict("output" => OptionsDict("run_name" => "two_ion_sources_moments")))
test_input_2 = recursive_merge(test_input_1,
                                    OptionsDict("output" => OptionsDict("run_name" => "two_ion_sources"),
                                                "evolve_moments" => OptionsDict("density" => false,
                                                         "parallel_flow" => false,
                                                         "parallel_pressure" => false,
                                                         "moments_conservation" => false)))
test_input_3 = recursive_merge(test_input,
                                    OptionsDict("output" => OptionsDict("run_name" => "PI_controller_sources_moments"),
                                                "ion_source_1" => OptionsDict("active" => true,
                                                                "z_profile" => "super_gaussian_4",
                                                                "z_width" => 0.275816,
                                                                "source_strength" => 5.0,
                                                                "source_T" => 1.5,
                                                                "source_type" => "temperature_midpoint_control",
                                                                "PI_temperature_controller_I" => 5.0,
                                                                "PI_temperature_controller_P" => 5.0,
                                                                "PI_temperature_target_amplitude" => 1.0),
                                                "ion_source_2" => OptionsDict("active" => true,
                                                                "z_profile" => "wall_exp_decay",
                                                                "z_width" => 0.15,
                                                                "source_strength" => 5.0,
                                                                "source_T" => 0.1,
                                                                "source_type" => "density_midpoint_control",
                                                                "PI_density_controller_I" => 1.0,
                                                                "PI_density_controller_P" => 1.0,
                                                                "PI_density_target_amplitude" => 1.15)))
test_input_4 = recursive_merge(test_input,
                                    OptionsDict("output" => OptionsDict("run_name" => "PI_controller_sources"),
                                                "ion_source_1" => OptionsDict("active" => true,
                                                                "z_profile" => "super_gaussian_4",
                                                                "z_width" => 0.275816,
                                                                "source_strength" => 5.0,
                                                                "source_T" => 1.5,
                                                                "source_type" => "temperature_midpoint_control",
                                                                "PI_temperature_controller_I" => 5.0,
                                                                "PI_temperature_controller_P" => 5.0,
                                                                "PI_temperature_target_amplitude" => 1.0),
                                                "ion_source_2" => OptionsDict("active" => true,
                                                                "z_profile" => "wall_exp_decay",
                                                                "z_width" => 0.15,
                                                                "source_strength" => 5.0,
                                                                "source_T" => 0.1,
                                                                "source_type" => "density_midpoint_control",
                                                                "PI_density_controller_I" => 1.0,
                                                                "PI_density_controller_P" => 1.0,
                                                                "PI_density_target_amplitude" => 1.15),
                                                "evolve_moments" => OptionsDict("density" => false,
                                                         "parallel_flow" => false,
                                                         "parallel_pressure" => false,
                                                         "moments_conservation" => false)))
test_input_3["timestepping"] = recursive_merge(test_input_3["timestepping"],
                                                    OptionsDict("nstep" => 500))
test_input_4["timestepping"] = recursive_merge(test_input_4["timestepping"],
                                                    OptionsDict("nstep" => 500))


"""
Run a test for a single set of parameters
"""
function run_test(test_input, expected_phi; rtol=4.e-14, atol=1.e-15, args...)
    # by passing keyword arguments to run_test, args becomes a Tuple of Pairs which can be
    # used to update the default inputs

    # Make a copy to make sure nothing modifies the input Dicts defined in this test
    # script.
    input = deepcopy(test_input)

    # Convert keyword arguments to a unique name
    name = input["output"]["run_name"]
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

            path = joinpath(realpath(input["output"]["base_directory"]), name)

            # open the output file(s)
            run_info = get_run_info_no_setup(path)

            # load fields data
            phi_zrt = postproc_load_variable(run_info, "phi")

            close_run_info(run_info)
            
            phi = phi_zrt[:,1,:]
        end

        # Regression test
        actual_phi = phi[begin:3:end, end]
        if expected_phi == nothing
            # Error: no expected input provided
            println("data tested would be: ", actual_phi)
            @test false
        else
            @test isapprox(actual_phi, expected_phi, rtol=rtol, atol=atol)
        end
    end
end

function runtests()
    # Create a temporary directory for test output
    test_output_directory = get_MPI_tempdir()

    @testset "Multi Source Tests" verbose=use_verbose begin
        println("multi source tests")
        @testset "multi source test 1" begin
            test_input_1["output"]["base_directory"] = test_output_directory
            run_test(test_input_1,
                     [1.177039517487497, 1.2466870725132324, 1.2304057021635098, 1.2025800545745353,
                      1.170016480522288, 1.1470474082660567, 1.138147400564512, 1.1339953462315175,
                      1.1329990091992277, 1.1349572917198698, 1.1346735334889448, 1.133266904494638,
                      1.13204294530494, 1.1311808917927337, 1.131326906130658, 1.1321943017292304,
                      1.1335923595196986, 1.1350671519682625, 1.1342556940412936, 1.1328538547486326,
                      1.1346476716961131, 1.142185341606377, 1.1549428685708552, 1.1750764794074362,
                      1.209007556716328, 1.2433950674999232, 1.2404979440021302],
                    )
        end
        @testset "multi source test 2" begin
            test_input_2["output"]["base_directory"] = test_output_directory
            run_test(test_input_2,
                     [1.1655594865309673, 1.2495047470884797, 1.230997617271214, 1.2034822179184976, 
                     1.1697999553188756, 1.1463454370580297, 1.1367173452723358, 1.1331639419699535, 
                     1.1336775694711045, 1.1351193814970546, 1.1350547486105123, 1.1339927128368603, 
                     1.1320762538074045, 1.1310078946060595, 1.1312740709930746, 1.1323814549394349, 
                     1.1342953190135054, 1.1352369362109493, 1.1346639276500559, 1.1333979576182693, 
                     1.133544688363875, 1.1411360468552718, 1.1545894547758784, 1.174956401346126, 
                     1.209797710419845, 1.2427519584559887, 1.2408823667463018])
        end
        @testset "multi source test 3" begin
            test_input_3["output"]["base_directory"] = test_output_directory
            run_test(test_input_3,
                     [0.6167292220789702, 0.671266950980533, 0.652186427050238, 0.6171268652300546,
                      0.560627377718737, 0.4982499090983303, 0.4477949572528707, 0.3951008138944443,
                      0.3222087800130001, 0.245760938042688, 0.18711965408665104, 0.12992149771021133,
                      0.06700995390392604, 0.038880925615974005, 0.04537287142815689, 0.07571977951536125,
                      0.14258443396554177, 0.21915262084593945, 0.2777465760185855, 0.3349936097689908,
                      0.4072240721346049, 0.4756945447869558, 0.5247079546747913, 0.5707894523523914,
                      0.6259766060393908, 0.6659325341619956, 0.6671576192192057],
                    )
        end
        @testset "multi source test 4" begin
            test_input_4["output"]["base_directory"] = test_output_directory
            run_test(test_input_4,
                     [0.5984365214316864, 0.6744906059289919, 0.6534852815849308, 0.6178888606892614,
                     0.5614296327200545, 0.49778285412482914, 0.44710694784149596, 0.39467742579032516,
                     0.3209369740409064, 0.24518024027297647, 0.18643336780127007, 0.12988039805427473,
                     0.06835594800600428, 0.03752999665967363, 0.04533170640973584, 0.07717489277321508,
                     0.1422937389561499, 0.21852502360651355, 0.2769230904373071, 0.33371818093027633,
                     0.4068546785524141, 0.474933015820431, 0.5249009953973673, 0.5716423767122453,
                     0.6267156044798625, 0.6674142974837349, 0.6701223308005594])
        end
    end
    if global_rank[] == 0
        # Delete output directory to avoid using too much disk space
        rm(realpath(test_output_directory); recursive=true)
    end
end

end

using .Multi_Source_Tests

Multi_Source_Tests.runtests()
