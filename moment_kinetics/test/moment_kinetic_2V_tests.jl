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
                                                         "pressure" => false,
                                                         "moments_conservation" => false),
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
                                                                           "pressure" => false,
                                                                           "moments_conservation" => true)
                                          ))
test_input_3 = recursive_merge(test_input,
                               OptionsDict("output" => OptionsDict("run_name" => "evolve n, upar"),
                                           "evolve_moments" => OptionsDict("density" => true,
                                                                           "parallel_flow" => true,
                                                                           "pressure" => false,
                                                                           "moments_conservation" => true)))
test_input_4 = recursive_merge(test_input,
                               OptionsDict("output" => OptionsDict("run_name" => "evolve n, upar, p"),
                                           "evolve_moments" => OptionsDict("density" => true,
                                                                           "parallel_flow" => true,
                                                                           "pressure" => true,
                                                                           "moments_conservation" => true)))


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
                     [-0.02958641888473927, 0.015345894495704743, 0.03850648789378049,
                      0.05936260290156026, 0.08515114987106931, 0.10887983553400411,
                      0.12553366722501813, 0.1410843989813511, 0.15997973030413892,
                      0.17707411731056913, 0.18886177871470247, 0.1996795692125457,
                      0.21251837726690656, 0.22375944157912953, 0.23123745869691767,
                      0.23784222867864271, 0.24525096142682568, 0.25119369468110175,
                      0.25473382620375873, 0.25745521650148273, 0.25980472134494303,
                      0.2607471720672994, 0.2605271261523872, 0.25950179822300207,
                      0.2569100825678205, 0.2528995867345716, 0.24890831328094976,
                      0.24408476587142025, 0.23642165973574403, 0.22725684766803372,
                      0.2192940568037614, 0.21043047367445064, 0.19731257818731687,
                      0.18253668793383526, 0.17020764849835712, 0.15686764629365388,
                      0.13765654696773685, 0.11656409680993901, 0.0992945172270152,
                      0.08086970085097524, 0.05476440166614846, 0.026427758025466038,
                      0.004992972429059984])
        end
        @testset "evolve n, upar" begin
            test_input_3["output"]["base_directory"] = test_output_directory
            run_test(test_input_3,
                     [-0.025878300996710603, 0.015756973214508592, 0.038400053081466184,
                      0.05938057769510571, 0.08515732535709798, 0.10888041177629737,
                      0.12553128843884281, 0.14108395707363672, 0.15997327003967873,
                      0.17707780549074287, 0.18886310123992378, 0.19967728716341343,
                      0.21251855717104015, 0.2237555140047125, 0.2312388384262849,
                      0.23784363336210299, 0.24525104836221417, 0.2511948578987215,
                      0.25473227797286413, 0.2574561966120675, 0.2598050729574235,
                      0.2607468580471476, 0.26052720656822376, 0.25950154716015666,
                      0.2569123862164899, 0.25289881970307143, 0.24890796924273628,
                      0.24408314495127376, 0.2364181716901413, 0.22726086495095166,
                      0.21929262779550493, 0.21043038973042544, 0.197311571822666,
                      0.1825307242909392, 0.17021257876752158, 0.15686743651744298,
                      0.13765711019017288, 0.11656909264570159, 0.09928762318048594,
                      0.08088094414830731, 0.05474041607016818, 0.02657435105118261,
                      0.004914589811856145])
        end
        @testset "evolve n, upar, p" begin
            test_input_4["output"]["base_directory"] = test_output_directory
            run_test(test_input_4,
                     [-0.018746759055651218, 0.016608588239259578, 0.038490969760901855,
                      0.05930552108501646, 0.08516565113092933, 0.10889791240521163,
                      0.12557822391956658, 0.14106614893862587, 0.15995979209186154,
                      0.17708058007592134, 0.1888785309261968, 0.19967916602799565,
                      0.2125162628463107, 0.22375681421622465, 0.23124178743639986,
                      0.23783659766223586, 0.24524941831579983, 0.25118859841737445,
                      0.2547325563317703, 0.2574530943999433, 0.25980379363668255,
                      0.2607478764975119, 0.26052506625821714, 0.2595018214634028,
                      0.2569082767568533, 0.25289843192835215, 0.2489077861921658,
                      0.24408404029362957, 0.23642073293716637, 0.2272551910193792,
                      0.2192936326553607, 0.21043100949179153, 0.19730618966969257,
                      0.18255598887926416, 0.1702002160305327, 0.15684390339006393,
                      0.13767104762780405, 0.11659550935872982, 0.0993022134329089,
                      0.08085219075685837, 0.05476050396823248, 0.02652584794398487,
                      0.003857640308930099])
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
