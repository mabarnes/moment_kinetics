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
                                                                 "upar_amplitude" => -1.0,
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
                                                "z" => OptionsDict("nelement" => 16),
                                                "vpa" => OptionsDict("nelement" => 31),
                                                "vz" => OptionsDict("nelement" => 31),
                                                "evolve_moments" => OptionsDict("parallel_pressure" => true),
                                                "ion_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0, "vpa_dissipation_coefficient" => 1e-2),
                                                "neutral_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0, "vz_dissipation_coefficient" => 1e-2)))
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
                                                         "ion_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0,
                                                                                                "vpa_dissipation_coefficient" => 1e-2),
                                                         "neutral_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0,
                                                                                                        "vz_dissipation_coefficient" => 1e-2)))
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
                     [-0.0546579889285807, -0.019016549127873168, -0.0014860800466385304,
                      0.0009959205072609873, 0.0018297472055798175, 0.001071042733974246,
                      0.0010872542779497558, 0.0034088344425557424, 0.006495346769181668,
                      0.014654320273015945, 0.03434558121245117, 0.0357330046889818,
                      0.02209990163293976, 0.0072248464387002585, 0.004816169992667908,
                      0.001426397872859017, 0.001056590777701787, 0.0014051997473547636,
                      0.002295772235866664, -0.0002868846663738925, -0.010975801853574857,
                      -0.04842263416979855])
        end
        @long @testset "Split 1" begin
            test_input_split1["output"]["base_directory"] = test_output_directory
            run_test(test_input_split1,
                     [-0.054563857216788185, -0.018799904320771384,
                      -0.0013813755926268458, 0.0009410807399870439,
                      0.0018697742453887686, 0.0010051005159105932, 0.0010876435266858567,
                      0.0034066435133874823, 0.006496566986134889, 0.014652736122697263,
                      0.03434472857478834, 0.03573031916933763, 0.022099731991587105,
                      0.007224104967108221, 0.004816906089578281, 0.0014256169181264293,
                      0.0010495384704166898, 0.0013581293677004877, 0.002353403977796533,
                      -9.749645953005884e-5, -0.010989893326671225,
                      -0.048206551108154555])
        end
        @long @testset "Split 2" begin
            test_input_split2["output"]["base_directory"] = test_output_directory
            run_test(test_input_split2,
                     [-0.05533765039692548, -0.019988930832647665, -0.0011360425985310237,
                      0.0012634193218142506, 0.001973468809930324, 0.0011030262051711455,
                      0.0010894927799806951, 0.0034211556308428115, 0.006489998905405594,
                      0.014662480637535611, 0.03436426327423981, 0.03570508542770078,
                      0.02201839105070786, 0.007240224076830071, 0.004831922106750905,
                      0.0013687623664681752, 0.001052085275258212, 0.001443601147195592,
                      0.002332348337948356, 0.0002552089321632552, -0.009924802755589088,
                      -0.04713668261221819])
        end
        @long @testset "Split 3" begin
            test_input_split3["output"]["base_directory"] = test_output_directory
            run_test(test_input_split3,
                     [-0.036201005590413136, -0.030491117907753874, -0.028973296620063916,
                      -0.028564862452844172, -0.025512618660072896, -0.02196792239883357,
                      -0.019842218654093358, -0.013330062672677903, -0.00986714090777353,
                      -0.00527481405661857, 0.0021548155387342443, 0.004603881763382613,
                      0.011494880559269222, 0.017553205830050052, 0.020142013053642518,
                      0.027112781087200055, 0.030495538742431827, 0.034040011991442055,
                      0.03931819038952405, 0.04088876176973523, 0.04435714998863926,
                      0.04581950734730935, 0.04569761355540199, 0.04311624813911122,
                      0.040540483545858805, 0.03786566112681951, 0.032230202566340176,
                      0.0300992957991312, 0.02482049364721276, 0.018584209008067783,
                      0.016208714957221627, 0.008758063368963747, 0.00399905983221347,
                      2.6245859593182608e-5, -0.007645815312542831, -0.01028435155697857,
                      -0.015550824790382023, -0.021192069467468835, -0.0227966269988034,
                      -0.026899932526209005, -0.028755147997832256, -0.029230736228611692,
                      -0.03231799818320131])
        end

        fullf_expected_output = [-0.04372543535228032, -0.02233515082616229,
                                 -0.012793688037658377, -0.010786492944264052,
                                 -0.007051439902278702, -0.0001605908774545327,
                                 0.005982619745890949, 0.0118094191749825,
                                 0.01954207152061524, 0.02978202423468538,
                                 0.039384279904624404, 0.042446003403153604,
                                 0.03181914367119813, 0.021111423438351817,
                                 0.015103049638495273, 0.009135485828230407,
                                 0.0010369322036392606, -0.005949066066045502,
                                 -0.00942148866222427, -0.011607485576226423,
                                 -0.020871221194795328, -0.03762871759968933]
        @testset "Adaptive timestep - full-f" begin
            test_input_adaptive["output"]["base_directory"] = test_output_directory
            run_test(test_input_adaptive,
                     fullf_expected_output, rtol=6.0e-4, atol=2.0e-12)
        end
        @testset "Adaptive timestep - split 1" begin
            test_input_adaptive_split1["output"]["base_directory"] = test_output_directory
            run_test(test_input_adaptive_split1,
                     [-0.04375862714017892, -0.022363510973059945, -0.012739964397542611,
                      -0.010806509398868007, -0.007052551067569563,
                      -0.0001618866835357178, 0.005980921838191561, 0.011808361372364367,
                      0.019540868336503224, 0.02978014755372564, 0.03938085813395519,
                      0.042446888380863836, 0.031821059258512106, 0.021109010112552534,
                      0.015101702015235266, 0.009134407186439548, 0.0010347434646523774,
                      -0.005951302261109976, -0.009412276056941643, -0.011636393512121094,
                      -0.020739923046188418, -0.03769486232955374], rtol=6.0e-4,
                     atol=2.0e-12)
        end
        @testset "Adaptive timestep - split 2" begin
            test_input_adaptive_split2["output"]["base_directory"] = test_output_directory
            run_test(test_input_adaptive_split2,
                     [-0.043902745384700015, -0.022550462296292408, -0.012831220868706328,
                      -0.010841526838840985, -0.007058696149876899,
                      -0.00020546115043597736, 0.005958897121894739, 0.011800033866913003,
                      0.019540465120580244, 0.02976675948935943, 0.039378879076308826,
                      0.042435475680453, 0.031824430350042995, 0.021101126707196695,
                      0.015090219386897738, 0.009144719909922057, 0.001010963073164482,
                      -0.005935409254325001, -0.009395462491654565, -0.011530687837538266,
                      -0.020829706057523486, -0.037774166127716446], rtol=6.0e-4,
                      atol=2.0e-12)
        end
        @testset "Adaptive timestep - split 3" begin
            test_input_adaptive_split3["output"]["base_directory"] = test_output_directory
            run_test(test_input_adaptive_split3,
                     [-0.034619481497656486, -0.032001843791679654, -0.027137501498240093,
                      -0.020923013640973737, -0.010149996704146695, 0.002788290244354704,
                      0.012835678785462054, 0.022093096980103734, 0.033030570055962855,
                      0.04152810760686341, 0.04538042982311558, 0.04624539107810815,
                      0.04254859680630147, 0.03481082088855338, 0.027076891429474205,
                      0.017883687065968965, 0.0047831407960558375, -0.007761873419235209,
                      -0.016291940103607693, -0.024131855580845837, -0.0315589223354697,
                      -0.034165527022754195], rtol=6.0e-4, atol=2.0e-12)
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
