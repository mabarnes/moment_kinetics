module RecyclingFraction

# Regression test using wall boundary conditions, with recycling fraction less than 1 and
# a plasma source. Runs for a while and then checks phi profile against saved reference
# output.

include("setup.jl")

using Base.Filesystem: tempname
using MPI

using moment_kinetics.interpolation: interpolate_to_grid_z
using moment_kinetics.load_data: get_run_info_no_setup, close_run_info,
                                 postproc_load_variable
using moment_kinetics.utils: merge_dict_with_kwargs!

# default inputs for tests
test_input = OptionsDict("composition" => OptionsDict("n_ion_species" => 1,
                                                      "n_neutral_species" => 1,
                                                      "electron_physics" => "boltzmann_electron_response",
                                                      "T_e" => 0.2,
                                                      "T_wall" => 0.1,
                                                      "recycling_fraction" => 0.5),
                         "ion_species_1" => OptionsDict("initial_density" => 1.0,
                                                        "initial_temperature" => 1.0),
                         "z_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                             "density_amplitude" => 0.0,
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
                         "neutral_species_1" => OptionsDict("initial_density" => 1.0,
                                                            "initial_temperature" => 1.0),
                         "z_IC_neutral_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                                 "density_amplitude" => 0.001,
                                                                 "density_phase" => 0.0,
                                                                 "upar_amplitude" => 1.0,
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
                         "output" => OptionsDict("run_name" => "full-f"),
                         "evolve_moments" => OptionsDict("density" => false,
                                                         "parallel_flow" => false,
                                                         "parallel_pressure" => false,
                                                         "moments_conservation" => false),
                         "reactions" => OptionsDict("charge_exchange_frequency" => 0.75,
                                                    "ionization_frequency" => 0.5),
                         "timestepping" => OptionsDict("nstep" => 1000,
                                                       "dt" => 1.0e-4,
                                                       "nwrite" => 1000,
                                                       "split_operators" => false),
                         "r" => OptionsDict("ngrid" => 1,
                                            "nelement" => 1),
                         "z" => OptionsDict("ngrid" => 9,
                                            "nelement" => 8,
                                            "bc" => "wall",
                                            "discretization" => "chebyshev_pseudospectral",
                                            "element_spacing_option" => "sqrt"),
                         "vpa" => OptionsDict("ngrid" => 10,
                                              "nelement" => 15,
                                              "L" => 18.0,
                                              "bc" => "zero",
                                              "discretization" => "chebyshev_pseudospectral",
                                              "element_spacing_option" => "coarse_tails"),
                         "vz" => OptionsDict("ngrid" => 10,
                                             "nelement" => 15,
                                             "L" => 18.0,
                                             "bc" => "zero",
                                             "discretization" => "chebyshev_pseudospectral",
                                             "element_spacing_option" => "coarse_tails"),
                         "ion_source_1" => OptionsDict("active" => true,
                                                     "z_profile" => "gaussian",
                                                     "z_width" => 0.125,
                                                     "source_strength" => 2.0,
                                                     "source_T" => 2.0),
                        )

if global_size[] > 2 && global_size[] % 2 == 0
    # Test using distributed-memory
    test_input["z"]["nelement_local"] = test_input["z"]["nelement"] ÷ 2
end

test_input_split1 = recursive_merge(test_input,
                                    OptionsDict("output" => OptionsDict("run_name" => "split1"),
                                                "evolve_moments" => OptionsDict("density" => true,
                                                                                "moments_conservation" => true)))
test_input_split2 = recursive_merge(test_input_split1,
                                    OptionsDict("output" => OptionsDict("run_name" => "split2"),
                                                "evolve_moments" => OptionsDict("parallel_flow" => true)))
test_input_split3 = recursive_merge(test_input_split2,
                                    OptionsDict("output" => OptionsDict("run_name" => "split3"),
                                                "z" => OptionsDict("ngrid" => 5,
                                                                   "nelement" => 32),
                                                "vpa" => OptionsDict("nelement" => 31),
                                                "vz" => OptionsDict("nelement" => 31),
                                                "evolve_moments" => OptionsDict("parallel_pressure" => true),
                                                "ion_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0),
                                                "neutral_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0)
                                               ))
test_input_split3["timestepping"] = recursive_merge(test_input_split3["timestepping"],
                                                    OptionsDict("dt" => 1.0e-5,
                                                                "write_error_diagnostics" => true,
                                                                "write_steady_state_diagnostics" => true))

# default inputs for adaptive timestepping tests
test_input_adaptive = recursive_merge(test_input,
                                      OptionsDict("output" => OptionsDict("run_name" => "adaptive full-f"),
                                                  "z" => OptionsDict("ngrid" => 5,
                                                                     "nelement" => 16),
                                                  "vpa" => OptionsDict("ngrid" => 6,
                                                                       "nelement" => 31),
                                                  "vz" => OptionsDict("ngrid" => 6,
                                                                      "nelement" => 31)),
                                     )
# Note, use excessively conservative timestepping settings here, because
# we want to avoid any timestep failures in the test. If failures
# occur, the number or when exactly they occur could depend on the
# round-off error, which could make the results less reproducible (even
# though the difference should be negligible compared to the
# discretization error of the simulation).
test_input_adaptive["timestepping"] = recursive_merge(test_input_adaptive["timestepping"],
                                                      OptionsDict("type" => "Fekete4(3)",
                                                                  "nstep" => 5000,
                                                                  "dt" => 1.0e-5,
                                                                  "minimum_dt" => 1.0e-5,
                                                                  "CFL_prefactor" => 1.0,
                                                                  "step_update_prefactor" => 0.5,
                                                                  "nwrite" => 1000,
                                                                  "split_operators" => false),
                                                     )

test_input_adaptive_split1 = recursive_merge(test_input_adaptive,
                                             OptionsDict("output" => OptionsDict("run_name" => "adaptive split1"),
                                                         "evolve_moments" => OptionsDict("density" => true,
                                                                                         "moments_conservation" => true)))
test_input_adaptive_split2 = recursive_merge(test_input_adaptive_split1,
                                             OptionsDict("output" => OptionsDict("run_name" => "adaptive split2"),
                                                         "evolve_moments" => OptionsDict("parallel_flow" => true)))
test_input_adaptive_split2["timestepping"] = recursive_merge(test_input_adaptive_split2["timestepping"],
                                                             OptionsDict("step_update_prefactor" => 0.4))
test_input_adaptive_split3 = recursive_merge(test_input_adaptive_split2,
                                             OptionsDict("output" => OptionsDict("run_name" => "adaptive split3"),
                                                         "evolve_moments" => OptionsDict("parallel_pressure" => true),
                                                         "ion_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0),
                                                         "neutral_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0)))
# The initial conditions seem to make the split3 case hard to advance without any
# failures. In a real simulation, would just set the minimum_dt higher to try to get
# through this without crashing. For this test, want the timestep to adapt (not just sit
# at minimum_dt), so just set a very small timestep.
test_input_adaptive_split3["timestepping"] = recursive_merge(test_input_adaptive_split3["timestepping"],
                                                             OptionsDict("dt" => 1.0e-7,
                                                                         "rtol" => 2.0e-4,
                                                                         "atol" => 2.0e-10,
                                                                         "minimum_dt" => 1.0e-7,
                                                                         "step_update_prefactor" => 0.064))

# Test exact_output_times option in full-f/split1/split2 cases
test_input_adaptive["timestepping"]["exact_output_times"] = true
test_input_adaptive_split1["timestepping"]["exact_output_times"] = true
test_input_adaptive_split2["timestepping"]["exact_output_times"] = true

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

    @testset "Recycling fraction" verbose=use_verbose begin
        println("Recycling fraction tests")

        @long @testset "Full-f" begin
            test_input["output"]["base_directory"] = test_output_directory
            run_test(test_input,
                     [-0.08248729935092564, -0.0068871583342621084, 0.039072432627366205,
                      0.04693870010589992, 0.05150706650565362, 0.04863426346608422,
                      0.0473083638177342, 0.04470973516451102, 0.04499280855240238,
                      0.04954730803537196, 0.06518108834766248, 0.06636324426299682,
                      0.05510570743405892, 0.04525268506850688, 0.04462918792987642,
                      0.046236393436293884, 0.04848556048397911, 0.05023315101138076,
                      0.05183197614172413, 0.04149963707449863, 0.014868291243593535,
                      -0.06917188928099624])
        end
        @long @testset "Split 1" begin
            test_input_split1["output"]["base_directory"] = test_output_directory
            run_test(test_input_split1,
                     [-0.08227633463338031, -0.0071358543437329116, 0.03928189120789499,
                      0.04721955765691388, 0.05122739409762711, 0.04910057401688113,
                      0.04704742545272498, 0.04490556032129777, 0.044808028630569895,
                      0.0495822812342297, 0.06518219267369472, 0.0663799710068059,
                      0.05509147172366616, 0.045315712030082195, 0.044456099676376606,
                      0.04646245178090258, 0.048357365450345, 0.05063961127735492,
                      0.051580462480704026, 0.04049988898861208, 0.015026988159329766,
                      -0.06816429195564531])
        end
        @long @testset "Split 2" begin
            test_input_split2["output"]["base_directory"] = test_output_directory
            run_test(test_input_split2,
                     [-0.07540720242558024, -0.012215936614592013, 0.03719872765590104,
                      0.04698103121369704, 0.05165670496014653, 0.04858108758621746,
                      0.0471560227622779, 0.04479278749716742, 0.04504127679218328,
                      0.04950353293493882, 0.06523358923266383, 0.06629669682555932,
                      0.05503302217925922, 0.04535511633539406, 0.04455941812203692,
                      0.04604589649002669, 0.04841777101341742, 0.05027051359300514,
                      0.05168079983767343, 0.04238354065327748, 0.012679178294098465,
                      -0.06448944047879421])
        end
        @long @testset "Split 3" begin
            test_input_split3["output"]["base_directory"] = test_output_directory
            run_test(test_input_split3,
                     [-0.04193263648303976, -0.029252245961908147, -0.02442862865278743,
                      -0.023495424780076817, -0.021218702925586936, -0.0175137673174747,
                      -0.014147249354188124, -0.010578808249367246,
                      -0.0054123391485627935, 0.0002496409027928137, 0.004696874850572911,
                      0.009073612343120277, 0.014872636546990831, 0.020638240855909998,
                      0.024828897215039916, 0.02873872765245768, 0.03355964489516307,
                      0.03800723296691896, 0.04109627070324763, 0.04386632942913408,
                      0.04685146950915689, 0.04834447861643572, 0.047986299492812835,
                      0.046434795212366146, 0.04324402903754499, 0.03942532151501047,
                      0.03625638152568082, 0.03278650998309304, 0.027824317716378874,
                      0.022537980746503067, 0.0183501045099858, 0.013934577339079934,
                      0.008010459888136036, 0.0022238610414135186, -0.0020340880648959266,
                      -0.006273371184150506, -0.011509711109921564, -0.01608354415559066,
                      -0.019068436253618693, -0.02165307457245153, -0.02374536776323688,
                      -0.026193225296752082, -0.03316168704620608])
        end

        fullf_expected_output = [-0.06485643672684559, -0.01393132401823833,
                                 0.011101377561621318, 0.014316795271888227,
                                 0.017153423792138713, 0.022446026558875336,
                                 0.02704022128763932, 0.0312708246641749,
                                 0.03687198166300583, 0.04513610953946474,
                                 0.05365557522739836, 0.05644944131511773,
                                 0.04690174365954638, 0.038059023968408545,
                                 0.03363220794997435, 0.029342946872462045,
                                 0.02335249209150473, 0.018006020904528016,
                                 0.01533652544061934, 0.013618448370696435,
                                 -0.011068038048902508, -0.05202549530770043]
        @testset "Adaptive timestep - full-f" begin
            test_input_adaptive["output"]["base_directory"] = test_output_directory
            run_test(test_input_adaptive,
                     fullf_expected_output, rtol=6.0e-4, atol=2.0e-12)
        end
        @testset "Adaptive timestep - split 1" begin
            test_input_adaptive_split1["output"]["base_directory"] = test_output_directory
            run_test(test_input_adaptive_split1,
                     [-0.06481156248488046, -0.013538803385323265, 0.011323667220073967,
                      0.014114887050256292, 0.017191717681080885, 0.02238431047775197,
                      0.027048121467700623, 0.031265619584043225, 0.03687116571351925,
                      0.04513292687646795, 0.05365135451877774, 0.056449651183667585,
                      0.04690045402594628, 0.038054818778305306, 0.033631107189163366,
                      0.029328697021190592, 0.02336956047980598, 0.017931456213012606,
                      0.015430873876177927, 0.013019311204642696, -0.01005690972539733,
                      -0.05241542841909974], rtol=6.0e-4, atol=2.0e-12)
        end
        @testset "Adaptive timestep - split 2" begin
            test_input_adaptive_split2["output"]["base_directory"] = test_output_directory
            run_test(test_input_adaptive_split2,
                     [-0.0657608236268271, -0.014190436712130003, 0.010968614554922509,
                      0.014365160035706915, 0.0172322487273797, 0.022428056290294318,
                      0.02698995818587201, 0.0312381471067878, 0.03683052762194124,
                      0.045132400084352924, 0.05365988862697188, 0.05641900300526736,
                      0.04690765450535961, 0.03807661571669711, 0.03371937764416954,
                      0.029312757797190604, 0.023326353109475284, 0.01804787967483964,
                      0.015433406790148586, 0.013580473449141255, -0.009739014707901247,
                      -0.0524306423232703], rtol=6.0e-4, atol=2.0e-12)
        end
        @testset "Adaptive timestep - split 3" begin
            test_input_adaptive_split3["output"]["base_directory"] = test_output_directory
            run_test(test_input_adaptive_split3,
                     [-0.034365284173257286, -0.03174419604712593, -0.0268868752546732,
                      -0.02068073646930811, -0.00992276648582255, 0.0029962667912489267,
                      0.013027765095553695, 0.022269773126771912, 0.033187984215071624,
                      0.04166938473091811, 0.04551362270498313, 0.04637663998599519,
                      0.042687804867641926, 0.034964968149446475, 0.02724494541583088,
                      0.01806746476857576, 0.004988029198591618, -0.0075380980824496695,
                      -0.016056024620199446, -0.023885297522794335, -0.03130293304134906,
                      -0.033910172140015765], rtol=6.0e-4, atol=2.0e-12)
        end

        @long @testset "Check other timestep - $type" for
                type ∈ ("RKF5(4)", "Fekete10(4)", "Fekete6(4)", "Fekete4(2)", "SSPRK3",
                        "SSPRK2", "SSPRK1")

            timestep_check_input = deepcopy(test_input_adaptive)
            timestep_check_input["output"]["base_directory"] = test_output_directory
            timestep_check_input["output"]["run_name"] = type
            timestep_check_input["timestepping"]["type"] = type
            run_test(timestep_check_input,
                     fullf_expected_output, rtol=8.e-4, atol=1.e-10)
        end
    end

    if global_rank[] == 0
        # Delete output directory to avoid using too much disk space
        rm(realpath(test_output_directory); recursive=true)
    end
end

end # RecyclingFraction


using .RecyclingFraction

RecyclingFraction.runtests()
