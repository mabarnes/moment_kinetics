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
                     [-0.07540857994933484, -0.012208843206798775, 0.03720233816559015,
                      0.04698788326747544, 0.05164683599385346, 0.04858930067940162,
                      0.04715113560430553, 0.04479033129863091, 0.04504249440395111,
                      0.04949906113971007, 0.06523039648058614, 0.06629849631963537,
                      0.055033266818536236, 0.04534543534152951, 0.04456401961984096,
                      0.04604481498279238, 0.0484153480784729, 0.05028357121913322,
                      0.05167173248709537, 0.0423818532488774, 0.012684309241598926,
                      -0.06449780658748071])
        end
        @long @testset "Split 3" begin
            test_input_split3["output"]["base_directory"] = test_output_directory
            run_test(test_input_split3,
                     [-0.04193017544040616, -0.029250494473353723, -0.02442741424960529,
                      -0.023494352755659486, -0.021215124699096363, -0.017516371626742005,
                      -0.014146922000027005, -0.010579020332652537, -0.005412461568939978,
                      0.00024964583074676164, 0.00469707504718307, 0.009073529319030582,
                      0.014872300744589643, 0.02063823775360385, 0.024828987616161716,
                      0.028738875972251887, 0.033559571565994664, 0.038007245774339005,
                      0.04109626837141106, 0.04386627713480251, 0.0468514799577687,
                      0.04834447485097812, 0.0479863054817698, 0.04643480367224086,
                      0.04324402877177113, 0.03942529792161823, 0.03625640296551428,
                      0.03278663234022876, 0.027824447109690426, 0.022537793844978678,
                      0.018350264436309764, 0.013934803730121254, 0.00801024983671142,
                      0.0022235848029891823, -0.002034038641182105, -0.006273946901194949,
                      -0.011509418646918046, -0.016083520056215172, -0.01906733893093257,
                      -0.021644058518558995, -0.023759351513767804, -0.026190820575707555,
                      -0.033156452558027634], rtol=0.0, atol=1.0e-13)
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
                     [-0.06574236425655763, -0.014177485851077622, 0.010966387236024945,
                      0.01436849241257219, 0.01722749787836073, 0.02242366464885637,
                      0.02699412837391181, 0.03123195902286819, 0.036833902052962816,
                      0.045133288123115095, 0.0536602966984219, 0.056418490736693416,
                      0.04690731048801407, 0.03807718213643924, 0.03372154310170833,
                      0.029312970447037554, 0.023328041373306796, 0.018040045381164712,
                      0.015429857741735657, 0.013591639292817873, -0.009719000188787711,
                      -0.052415949143534216], rtol=6.0e-4, atol=2.0e-12)
        end
        @testset "Adaptive timestep - split 3" begin
            test_input_adaptive_split3["output"]["base_directory"] = test_output_directory
            run_test(test_input_adaptive_split3,
                     [-0.03436290012305614, -0.031743187785688014, -0.026885788970081172,
                      -0.020679703620686454, -0.00992191748087324, 0.002996932141779539,
                      0.013028293970066947, 0.022270187375949758, 0.03318831404128298,
                      0.041669901623978926, 0.04551455772598044, 0.046377736366511976,
                      0.04268839700647822, 0.0349653033195384, 0.02724530974553528,
                      0.01806792678595286, 0.004988667308836913, -0.00753727365806825,
                      -0.016055068732846314, -0.023884180397275464, -0.03130151148633419,
                      -0.03390815373863548], rtol=6.0e-4, atol=2.0e-12)
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
