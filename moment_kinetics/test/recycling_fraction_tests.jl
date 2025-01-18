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
                     [-0.054564400690150644, -0.01880050885497155, -0.0013804889155909434,
                      0.0009426267362423344, 0.0018708794999890794, 0.0010048035580616115,
                      0.0010869046046222948, 0.0034056101774940675, 0.0064957786740579455,
                      0.01465251207308323, 0.034345864665430555, 0.03573156224898501,
                      0.022099994048796447, 0.007223371167561782, 0.004815984673023357,
                      0.0014246219209092864, 0.0010492088800026246, 0.001358555443115789,
                      0.002355419499405167, -9.616282952819355e-5, -0.010989980720963986,
                      -0.04820698953610652])
        end
        @long @testset "Split 2" begin
            test_input_split2["output"]["base_directory"] = test_output_directory
            run_test(test_input_split2,
                     [-0.055351930552923125, -0.0200209368236471, -0.0010274232338285407,
                      0.0011445828881595096, 0.001990016623266284, 0.0011847791295251302,
                      0.0012178159250162924, 0.003389968304925306, 0.006471876165122262,
                      0.014666710317521392, 0.034363113016583055, 0.03570763645228305,
                      0.022015422551673866, 0.007249734444536539, 0.004848925796605709,
                      0.0012928880774780984, 0.0009085783825923187, 0.0014263120887848147,
                      0.0023594712302784752, 0.0002954708425330566, -0.009955206404411004,
                      -0.04714624635817171])
        end
        @long @testset "Split 3" begin
            test_input_split3["output"]["base_directory"] = test_output_directory
            run_test(test_input_split3,
                     [-0.036195418620494954, -0.030489030308458488, -0.028975057418733397,
                      -0.02856021807109163, -0.025513413807863268, -0.0219696963676536,
                      -0.019843060635768725, -0.013329641098584045, -0.009865951845346884,
                      -0.005274664099474674, 0.0021551620276032725, 0.004604147727212109,
                      0.011494982185711863, 0.017553116141190095, 0.020142011471035323,
                      0.02711278698585964, 0.0304955647516801, 0.03404001571064709,
                      0.039318184954120955, 0.040888755835348824, 0.04435714972215338,
                      0.04581950770664514, 0.04569761337067911, 0.04311624840495188,
                      0.0405404864794332, 0.0378656646347818, 0.032230200940856144,
                      0.030099328699440486, 0.024820489910128835, 0.018584207663623988,
                      0.01620878075694702, 0.00875792663483109, 0.003998273774255805,
                      2.6225005465422193e-5, -0.007645978144613239, -0.010283112908873734,
                      -0.01555130379029211, -0.02119151686032106, -0.022798520707706906,
                      -0.026895590584576676, -0.02875783077548144, -0.029235490314989378,
                      -0.03231930055546638])
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
                     [-0.0440004026002034, -0.022740771274011903, -0.012908307424861458,
                      -0.010957840207013755, -0.007098397545728348,
                      -0.00020528082369771782, 0.0059601664366275495,
                      0.011802849958075045, 0.019546525347314055, 0.02976627257948505,
                      0.03937863700013744, 0.0424357099316342, 0.03182280591999282,
                      0.021097437598690982, 0.015088139859851357, 0.009141943563305855,
                      0.0010114496202298438, -0.0059305646605444726,
                      -0.009328644946126851, -0.011284802819960504, -0.020872565412882932,
                      -0.03785398593006839], rtol=6.0e-4, atol=2.0e-12)
        end
        @testset "Adaptive timestep - split 3" begin
            test_input_adaptive_split3["output"]["base_directory"] = test_output_directory
            run_test(test_input_adaptive_split3,
                     [-0.0346196925024167, -0.03200201693849987, -0.02713764319615098,
                      -0.02092311349672712, -0.010150026206894121, 0.0027883420935253572,
                      0.012835791449600767, 0.02209326318113659, 0.03303078703903627,
                      0.04152829640863164, 0.04538051487359227, 0.04624543438581702,
                      0.04254876799453081, 0.03481104153755928, 0.027077084096581314,
                      0.01788382934269672, 0.00478320487966262, -0.0077618876322877485,
                      -0.016292009420807548, -0.024131976958124225, -0.031559093785483404,
                      -0.0341657304695615], rtol=6.0e-4, atol=2.0e-12)
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
