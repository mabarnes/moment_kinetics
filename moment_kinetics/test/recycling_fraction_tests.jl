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
                     [-0.054605184755567575, -0.018924742429310467, -0.001490294393938991,
                      0.0010478821661319092, 0.001794078174490352, 0.0011156955937095606,
                      0.0010500480498606113, 0.003426231650337435, 0.006482800686571286,
                      0.014654506441798191, 0.03434808501583094, 0.03573157997956781,
                      0.022100535957809704, 0.007228878631671411, 0.004797433035497096,
                      0.0014539964057633523, 0.0010162605559513277, 0.001453906718097685,
                      0.0022621646648742617, -0.0002234581701885997,
                      -0.011002974597681126, -0.048285757044344964])
        end
        @long @testset "Split 2" begin
            test_input_split2["output"]["base_directory"] = test_output_directory
            run_test(test_input_split2,
                     [-0.0553605629303649, -0.0200578745274288, -0.0009982856556390678,
                      0.0011162158487700833, 0.001990656066255945, 0.0011958810815700098,
                      0.0012418062375573464, 0.0033857106837047457, 0.006465864015646492,
                      0.014667258936344657, 0.034362739564980156, 0.035706994697431264,
                      0.022016687200817217, 0.007247970747381021, 0.004852241393797722,
                      0.0012772593741711448, 0.0008886651723149342, 0.0014225505910423768,
                      0.002384091230841396, 0.0003535365076204503, -0.009964299368892577,
                      -0.04717082160422334])
        end
        @long @testset "Split 3" begin
            test_input_split3["output"]["base_directory"] = test_output_directory
            run_test(test_input_split3,
                     [-0.03619541456502106, -0.03048815264000018, -0.0289752901461535,
                      -0.0285607185338613, -0.02551344807111824, -0.021969237777543855,
                      -0.019843174019920886, -0.013329503542790564, -0.009865690314547893,
                      -0.005274660276261934, 0.0021551178094804456, 0.00460404591061373,
                      0.011494967591586436, 0.017553200481471323, 0.020142044042573953,
                      0.0271127742463413, 0.030495570082082302, 0.03404001245314961,
                      0.03931818810946109, 0.040888752388576165, 0.04435715083842175,
                      0.0458195069124898, 0.04569761327730698, 0.04311624638510156,
                      0.04054045787355211, 0.03786566212732171, 0.03223020823255074,
                      0.030099323023946085, 0.02482050522873986, 0.018584182557364992,
                      0.016208780074455798, 0.008757945646534438, 0.003998386597411979,
                      2.623477286710047e-5, -0.007645973957438945, -0.010282850103230351,
                      -0.015551493765485503, -0.021191569357145227, -0.022798925808646432,
                      -0.02689556051631625, -0.028759636408319134, -0.02923535124735689,
                      -0.032318350700783344])
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
                     [-0.043738639979155336, -0.022291274335936265, -0.012732019011476862,
                      -0.010820829608132652, -0.007052624822978379,
                      -0.00016633266759769646, 0.005981726941181142, 0.011808282498196942,
                      0.01954161140511405, 0.029780113019397654, 0.03938062653684496,
                      0.0424470451063335, 0.03181914854635646, 0.02110853735011568,
                      0.015101934236822627, 0.009133443309446937, 0.001036508267434348,
                      -0.00595575503479751, -0.009404524198476915, -0.011671134546870241,
                      -0.02071098205296615, -0.037674973785710573], rtol=6.0e-4,
                     atol=2.0e-12)
        end
        @testset "Adaptive timestep - split 2" begin
            test_input_adaptive_split2["output"]["base_directory"] = test_output_directory
            run_test(test_input_adaptive_split2,
                     [-0.04399078184696034, -0.022751021620615642, -0.01291299787865667,
                      -0.010959674298924109, -0.007104189286612762,
                      -0.00020400828543462016, 0.005962002320996209, 0.01180501356591465,
                      0.019549551740928053, 0.029764649056026507, 0.039377386669400126,
                      0.042436000769750994, 0.031820216219169195, 0.021096136207796035,
                      0.015087014270727023, 0.009138223367425275, 0.0010121863942479008,
                      -0.005929928872255217, -0.009323880222721409, -0.011284068874867409,
                      -0.020879540574606982, -0.03787906204870825], rtol=6.0e-4,
                      atol=2.0e-12)
        end
        @testset "Adaptive timestep - split 3" begin
            test_input_adaptive_split3["output"]["base_directory"] = test_output_directory
            run_test(test_input_adaptive_split3,
                     [-0.03461957353695876, -0.0320019353494458, -0.027137577284403983,
                      -0.020923066905994287, -0.010150012462338548, 0.0027883179270034,
                      0.01283573893667811, 0.022093185715449926, 0.03303068590467864,
                      0.041528208409286164, 0.045380475232019435, 0.04624541420039302,
                      0.04254868820465708, 0.0348109386942286, 0.02707699429521452,
                      0.017883763028987577, 0.004783175015047571, -0.0077618809993052484,
                      -0.016291977117329014, -0.02413192030270414, -0.03155900803717345,
                      -0.03416562437312799], rtol=6.0e-4, atol=2.0e-12)
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
