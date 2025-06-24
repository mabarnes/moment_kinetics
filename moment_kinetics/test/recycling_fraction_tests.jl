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
                                                        "initial_temperature" => 0.3333333333333333),
                         "z_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                             "density_amplitude" => 0.0,
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
                         "neutral_species_1" => OptionsDict("initial_density" => 1.0,
                                                            "initial_temperature" => 0.3333333333333333),
                         "z_IC_neutral_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                                 "density_amplitude" => 0.001,
                                                                 "density_phase" => 0.0,
                                                                 "upar_amplitude" => 1.4142135623730951,
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
                                                         "pressure" => false,
                                                         "moments_conservation" => false),
                         "reactions" => OptionsDict("charge_exchange_frequency" => 1.0606601717798214,
                                                    "ionization_frequency" => 0.7071067811865476),
                         "timestepping" => OptionsDict("nstep" => 1000,
                                                       "dt" => 7.071067811865475e-5,
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
                                              "L" => 25.455844122715714,
                                              "bc" => "zero",
                                              "discretization" => "chebyshev_pseudospectral",
                                              "element_spacing_option" => "coarse_tails7.0710678118654755"),
                         "vz" => OptionsDict("ngrid" => 10,
                                             "nelement" => 15,
                                             "L" => 25.455844122715714,
                                             "bc" => "zero",
                                             "discretization" => "chebyshev_pseudospectral",
                                             "element_spacing_option" => "coarse_tails7.0710678118654755"),
                         "ion_source_1" => OptionsDict("active" => true,
                                                       "z_profile" => "gaussian",
                                                       "z_width" => 0.125,
                                                       "source_strength" => 2.8284271247461903,
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
                                                "vpa" => OptionsDict("nelement" => 31,
                                                                     "L" => 31.17691453623979,
                                                                     "element_spacing_option" => "coarse_tails8.660254037844386"),
                                                "vz" => OptionsDict("nelement" => 31,
                                                                     "L" => 31.17691453623979,
                                                                     "element_spacing_option" => "coarse_tails8.660254037844386"),
                                                "evolve_moments" => OptionsDict("pressure" => true),
                                                "ion_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0),
                                                "neutral_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0)
                                               ))
test_input_split3["timestepping"] = recursive_merge(test_input_split3["timestepping"],
                                                    OptionsDict("dt" => 7.0710678118654756e-6,
                                                                "write_error_diagnostics" => true,
                                                                "write_steady_state_diagnostics" => true))
if global_size[] > 2 && global_size[] % 2 == 0
    # Test using distributed-memory
    test_input_split3["z"]["nelement_local"] = test_input_split3["z"]["nelement"] ÷ 2
end


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
                                                                  "dt" => 7.0710678118654756e-6,
                                                                  "minimum_dt" => 7.0710678118654756e-6,
                                                                  "CFL_prefactor" => 1.0,
                                                                  "step_update_prefactor" => 0.5,
                                                                  "nwrite" => 1000,
                                                                  "split_operators" => false),
                                                     )
if global_size[] > 2 && global_size[] % 2 == 0
    # Test using distributed-memory
    test_input_adaptive["z"]["nelement_local"] = test_input_adaptive["z"]["nelement"] ÷ 2
end

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
                                                         "evolve_moments" => OptionsDict("pressure" => true),
                                                         "vpa" => OptionsDict("element_spacing_option" => "coarse_tails8.660254037844386"),
                                                         "vz" => OptionsDict("element_spacing_option" => "coarse_tails8.660254037844386"),
                                                         "ion_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0),
                                                         "neutral_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0)))
# The initial conditions seem to make the split3 case hard to advance without any
# failures. In a real simulation, would just set the minimum_dt higher to try to get
# through this without crashing. For this test, want the timestep to adapt (not just sit
# at minimum_dt), so just set a very small timestep.
test_input_adaptive_split3["timestepping"] = recursive_merge(test_input_adaptive_split3["timestepping"],
                                                             OptionsDict("dt" => 7.071067811865474e-8,
                                                                         "rtol" => 2.0e-4,
                                                                         "atol" => 7.978845608028654e-11,
                                                                         "minimum_dt" => 7.071067811865474e-8,
                                                                         "step_update_prefactor" => 0.064))

# Test exact_output_times option in full-f/split1/split2 cases
test_input_adaptive["timestepping"]["exact_output_times"] = true
test_input_adaptive_split1["timestepping"]["exact_output_times"] = true
test_input_adaptive_split2["timestepping"]["exact_output_times"] = true

"""
Run a test for a single set of parameters
"""
function run_test(test_input, expected_phi; rtol=4.e-14, atol=1.e-14, args...)
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
            @test elementwise_isapprox(actual_phi, expected_phi, rtol=rtol, atol=atol)
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
                     [-0.08230231923193426, -0.007151336810940476, 0.03928150090508273,
                      0.04720942888256958, 0.051218664186784106, 0.04911040142326366,
                      0.04702432784283641, 0.04492608290270342, 0.04475371329383271,
                      0.04958840721002605, 0.06518289391282486, 0.06638198028979324,
                      0.055089818910184156, 0.04532934164595189, 0.04443837950994483,
                      0.046484318611771415, 0.04833407709402107, 0.05065051959194318,
                      0.05157917428668296, 0.040512007909999925, 0.015010111676104812,
                      -0.06817832486057487])
        end
        @long @testset "Split 2" begin
            test_input_split2["output"]["base_directory"] = test_output_directory
            run_test(test_input_split2,
                     [-0.07540737817438857, -0.012214539206225028, 0.037196479755700826,
                      0.04698586843686626, 0.05165392616096757, 0.04858268862255285,
                      0.047154769525900565, 0.044793673972038206, 0.045039985302168344,
                      0.049502281537099874, 0.06523423744817512, 0.0662961505892883,
                      0.055033736393419276, 0.04535400725445224, 0.04455993117036574,
                      0.04604714447324928, 0.048419958341796826, 0.05027132853538914,
                      0.05167751985550966, 0.04237898424191952, 0.012679694092107363,
                      -0.06448952988489469])
        end
        @long @testset "Split 3" begin
            test_input_split3["output"]["base_directory"] = test_output_directory
            run_test(test_input_split3,
                     [-0.04193262944422516, -0.0292523348567384, -0.024428783699460957,
                      -0.023495456673507387, -0.02121870968547827, -0.01751376788727926,
                      -0.014147250801955736, -0.010578798726573838, -0.005412339084223595,
                      0.0002496331825925261, 0.004696874026424771, 0.00907361138188349,
                      0.014872532751724394, 0.02063821042643102, 0.024828933919423243,
                      0.02873876440706485, 0.03355968721853348, 0.038007230640005135,
                      0.0410962759082342, 0.0438662686459695, 0.0468513254249662,
                      0.04834446506965362, 0.047986318186205676, 0.046434807777311474,
                      0.043244013543231465, 0.03942531233342713, 0.03625641027521971,
                      0.03278654806039884, 0.027824205726523013, 0.022537972849814434,
                      0.01835013483611784, 0.013934580193619907, 0.008010442737366919,
                      0.0022238653126705784, -0.002034084705240401, -0.0062733653547992,
                      -0.011509711916522641, -0.01608354569620076, -0.019068425421793166,
                      -0.021653048907287035, -0.02374519642625697, -0.026193097081716344,
                      -0.03316170067767647], rtol=0.0, atol=1.0e-13)
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
