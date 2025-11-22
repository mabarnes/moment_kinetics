module moment_kinetic_2V_tests

# Test for 2V moment kinetics with all four combinations (no evolution, evolve n, evolve n and upar and evolve all)

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
                         "ion_species_1" => OptionsDict("initial_density" => 1.0,
                                                        "initial_temperature" => 0.5),
                         "z_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                             "density_amplitude" => 0.001,
                                                             "density_phase" => 0.0,
                                                             "upar_amplitude" => 1.4142135623730951,
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
                         "output" => OptionsDict("run_name" => "moment kinetic 2V test",
                                                 "display_timing_info" => false),
                         "evolve_moments" => OptionsDict("density" => false,
                                                         "parallel_flow" => false,
                                                         "pressure" => false),
                         "timestepping" => OptionsDict("nstep" => 50,
                                                       "dt" => 0.002,
                                                       "nwrite" => 10,
                                                       "type" => "SSPRK4",
                                                       "nwrite_dfns" => 10,
                                                       "steady_state_residual" => true),
                         "r" => OptionsDict("ngrid" => 1,
                                            "nelement" => 1),
                         "z" => OptionsDict("ngrid" => 5,
                                            "nelement" => 32,
                                            "bc" => "wall",
                                            "discretization" => "chebyshev_pseudospectral",
                                            "L" => 14.0),
                         "vpa" => OptionsDict("ngrid" => 6,
                                              "nelement" => 41,
                                              "L" => 14.0,
                                              "bc" => "zero",
                                              "discretization" => "chebyshev_pseudospectral"),
                         "vperp" => OptionsDict("ngrid" => 5,
                                              "nelement" => 5,
                                              "L" => 5.0,
                                              "bc" => "zero",
                                              "discretization" => "chebyshev_pseudospectral"),
                         "ion_source_1" => OptionsDict("active" => true,
                                                       "z_profile" => "wall_exp_decay",
                                                       "z_width" => 2.0,
                                                       "source_strength" => 0.4,
                                                       "source_T" => 1.5),
                         "krook_collisions" => OptionsDict("use_krook" => true,
                                                           "frequency_option" => "reference_parameters"),
                        )

if global_size[] > 2 && global_size[] % 2 == 0
    # Test using distributed-memory
    test_input["z"]["nelement_local"] = test_input["z"]["nelement"] ÷ 2
end

test_input_1 = recursive_merge(test_input,
                               OptionsDict("output" => OptionsDict("run_name" => "evolve no moments")))
test_input_2 = recursive_merge(test_input_1,
                               OptionsDict("output" => OptionsDict("run_name" => "evolve n"),
                                           "evolve_moments" => OptionsDict("density" => true,
                                                                           "parallel_flow" => false,
                                                                           "pressure" => false)
                                          ))
test_input_3 = recursive_merge(test_input,
                               OptionsDict("output" => OptionsDict("run_name" => "evolve n, upar"),
                                           "evolve_moments" => OptionsDict("density" => true,
                                                                           "parallel_flow" => true,
                                                                           "pressure" => false)))
test_input_4 = recursive_merge(test_input,
                               OptionsDict("output" => OptionsDict("run_name" => "evolve n, upar, p"),
                                           "evolve_moments" => OptionsDict("density" => true,
                                                                           "parallel_flow" => true,
                                                                           "pressure" => true)))


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

    @testset "moment kinetic 2V tests" verbose=use_verbose begin
        println("moment kinetic 2V tests")
        @testset "evolve no moments" begin
            test_input_1["output"]["base_directory"] = test_output_directory
            run_test(test_input_1,
                     [-0.03195786582291607, 0.01595292950155847, 0.038436066947735874, 
                     0.05936462847409581, 0.08514257464064963, 0.10887464988738589, 
                     0.12552898857392722, 0.1410803177405943, 0.15997622918737975, 
                     0.17707099582666258, 0.1888588241905269, 0.19967668538969116, 
                     0.2125154607172428, 0.2237563716469047, 0.23123421291126808, 
                     0.23783877139418538, 0.24524719958562446, 0.25118963210216166, 
                     0.25472956025294435, 0.25745078207953126, 0.2598001366512172, 
                     0.2607425318230604, 0.26052249750500767, 0.25949723264883867, 
                     0.25690568278248727, 0.2528954283251802, 0.24890437257503556, 
                     0.24408105680556866, 0.23641825280144688, 0.22725370318360702, 
                     0.21929106330662576, 0.2104275718504746, 0.19730968637439739, 
                     0.18253365387769074, 0.17020439381047614, 0.15686406073443557, 
                     0.13765233886800962, 0.11655907208907308, 0.09928837336657659, 
                     0.08086702389500915, 0.054731525875648196, 0.026507336955891067, 
                     0.0044016399485068])
        end
        @testset "evolve n" begin
            test_input_2["output"]["base_directory"] = test_output_directory
            run_test(test_input_2,
                     [-0.029586446500606146, 0.015344646368164136, 0.03850575295195752, 
                     0.05936284875171169, 0.08515102152258382, 0.10887984403684092, 
                     0.12553366639808577, 0.1410843990584764, 0.15997973032629198, 
                     0.17707411731474043, 0.18886177872551557, 0.19967956920012334, 
                     0.21251837732667866, 0.22375944157905647, 0.2312374587053675, 
                     0.23784222867177746, 0.2452509614700188, 0.25119369467944774, 
                     0.254733826208002, 0.2574552164994963, 0.2598047213622178, 
                     0.2607471720662237, 0.2605271261537658, 0.25950179822159525, 
                     0.25691008259374765, 0.2528995867312877, 0.24890831328663215, 
                     0.24408476587003913, 0.2364216597861909, 0.22725684765918075, 
                     0.21929405681337683, 0.21043047367537088, 0.19731257824802692, 
                     0.18253668792046318, 0.17020764850748915, 0.15686764632348738, 
                     0.13765654634544056, 0.11656409864476265, 0.09929450439252614, 
                     0.08086978333666692, 0.05476356406449743, 0.026429080272430433, 
                     0.004992870293784367])
        end
        @testset "evolve n, upar" begin
            test_input_3["output"]["base_directory"] = test_output_directory
            run_test(test_input_3,
                     [-0.025867297270728076, 0.015696601141151355, 0.03835776422034526,
                      0.0593949473474276, 0.085154778794157, 0.10888089024961954,
                      0.12553114962578887, 0.14108400283146671, 0.15997315582863553,
                      0.17707778392741258, 0.18886319150598452, 0.19967716854004447,
                      0.21251872960152665, 0.22375544720813273, 0.23123881093468338,
                      0.23784364139600644, 0.24525093586808275, 0.2511948759635007,
                      0.25473226798567195, 0.2574562018686875, 0.2598050512359017,
                      0.260746864404421, 0.260527203460507, 0.2595015484361873,
                      0.25691231296675854, 0.2528988231452713, 0.24890794149555948,
                      0.24408313013466454, 0.23641831683912537, 0.22726091856608985,
                      0.2192926416674225, 0.21043036617921845, 0.1973117026568394,
                      0.18253057172502554, 0.17021271086315498, 0.15686758347910665,
                      0.13765684409126983, 0.11656954338100976, 0.09928664958285563,
                      0.08088644350496133, 0.05472810120392803, 0.026645255395328166,
                      0.004900963335426387])
        end
        @testset "evolve n, upar, p" begin
            test_input_4["output"]["base_directory"] = test_output_directory
            run_test(test_input_4,
                     [-0.01874858260560208, 0.016605285762353518, 0.03849558252466628,
                      0.059290558315720845, 0.08516996826553869, 0.10889747560061305,
                      0.12557778770726127, 0.14106645585753771, 0.15995968871179,
                      0.17708056476202555, 0.18887848253327869, 0.19967896679102304,
                      0.21251623476251302, 0.22375686865509464, 0.23124182890124662,
                      0.2378365948035242, 0.24524958304602415, 0.2511886327737058,
                      0.2547325638065351, 0.25745311363781154, 0.2598037368941537,
                      0.2607478787530877, 0.26052506377248763, 0.25950181635567576,
                      0.2569082604508303, 0.2528983924467731, 0.24890777593156,
                      0.2440840579991563, 0.23642056530072386, 0.22725511184953331,
                      0.21929364864927173, 0.21043097297506708, 0.19730653641803186,
                      0.18255601941185895, 0.17019994322711485, 0.15684382619347856,
                      0.1376713946425803, 0.1165955685877491, 0.09930294270558504,
                      0.08084622678608196, 0.05474517067040633, 0.026528762269052736,
                      0.0038647271118907745])
        end
    end
    if global_rank[] == 0
        # Delete output directory to avoid using too much disk space
        rm(realpath(test_output_directory); recursive=true)
    end
end

end

using .moment_kinetic_2V_tests

moment_kinetic_2V_tests.runtests()
