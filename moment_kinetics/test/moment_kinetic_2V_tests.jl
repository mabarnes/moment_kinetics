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
                     [-0.025878667437429656, 0.0157577559105578, 0.03840065058006514, 
                     0.05938022389166592, 0.08515739834179331, 0.10888039807986871, 
                     0.12553127941569595, 0.1410839456075142, 0.15997339601216806, 
                     0.17707781398879963, 0.18886311301378922, 0.19967730534305167, 
                     0.21251860384731094, 0.2237555305300224, 0.2312388644395613, 
                     0.23784362584711427, 0.24525097062387033, 0.2511948727345223, 
                     0.254732288142268, 0.25745619829270927, 0.25980504806708943, 
                     0.2607468709934845, 0.26052720120809436, 0.2595015444209003, 
                     0.25691232576450357, 0.2528987772622762, 0.24890796319529654, 
                     0.2440831572878942, 0.23641832407562147, 0.2272608359308803, 
                     0.21929262001148367, 0.21043039443536152, 0.19731165788793928, 
                     0.18253071541378027, 0.17021258270778422, 0.15686744035456499, 
                     0.1376571141504603, 0.11656908146502086, 0.09928764084076458, 
                     0.0808808071458333, 0.05474122177845513, 0.026573407117920167, 
                     0.004914837738343153])
        end
        @testset "evolve n, upar, p" begin
            test_input_4["output"]["base_directory"] = test_output_directory
            run_test(test_input_4,
                     [-0.01874815844123328, 0.016610035562085568, 0.03849098228232876, 
                     0.05930611336545814, 0.08516530769574408, 0.10889791093292411, 
                     0.1255781580228268, 0.14106614397593564, 0.15995996530565965, 
                     0.1770805799498577, 0.18887839659404995, 0.19967915609431408, 
                     0.2125162463279919, 0.22375683480696643, 0.2312418114073167, 
                     0.23783673266538807, 0.24524953337890698, 0.25118870174313324, 
                     0.25473259107259594, 0.25745312979480556, 0.25980373294226283, 
                     0.2607478830659452, 0.26052505844792945, 0.2595018077334283, 
                     0.2569082929463626, 0.25289831750616276, 0.24890776034305398, 
                     0.24408409466761372, 0.23642059581680788, 0.22725518859354343, 
                     0.2192936022775337, 0.21043099274568344, 0.19730666983070919, 
                     0.18255607086483028, 0.17020015544981804, 0.1568439954813532, 
                     0.13767100073326607, 0.11659563528112725, 0.09930213476025657, 
                     0.08085247537347456, 0.05476014224000251, 0.026524533179720977, 
                     0.003856919631693684])
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
