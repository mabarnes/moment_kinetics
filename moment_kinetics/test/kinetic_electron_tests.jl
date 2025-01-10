module KineticElectronsTests

# Regression test with kinetic electrons, using wall boundary conditions, with recycling
# fraction less than 1 and a plasma source. Runs a Boltzmann electron simulation, restarts
# as a kinetic electron simulation, and checks the final Ez profile.

include("setup.jl")

using moment_kinetics.load_data: get_run_info_no_setup, close_run_info,
                                 postproc_load_variable
using moment_kinetics.looping

using moment_kinetics.Glob

# Input for Boltzmann electron part of run
boltzmann_input = OptionsDict(
    "output" => OptionsDict("run_name" => "kinetic_electron_test_boltzmann_initialisation",
                           ),
    "evolve_moments" => OptionsDict("parallel_pressure" => true,
                                    "density" => true,
                                    "moments_conservation" => true,
                                    "parallel_flow" => true,
                                   ),
    "r" => OptionsDict("ngrid" => 1,
                       "nelement" => 1,
                      ),
    "z" => OptionsDict("ngrid" => 5,
                       "discretization" => "gausslegendre_pseudospectral",
                       "nelement" => 8,
                       "bc" => "wall",
                      ),
    "vpa" => OptionsDict("ngrid" => 6,
                         "discretization" => "gausslegendre_pseudospectral",
                         "nelement" => 17,
                         "L" => 24.0,
                         "element_spacing_option" => "coarse_tails",
                         "bc" => "zero",
                        ),
    "composition" => OptionsDict("T_e" => 0.2,
                                 "n_ion_species" => 1,
                                 "n_neutral_species" => 0,
                                ),
    "ion_species_1" => OptionsDict("initial_temperature" => 0.2,
                                   "initial_density" => 1.0,
                                  ),
    "z_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                        "density_amplitude" => 1.0,
                                        "temperature_amplitude" => 0.0,
                                        "density_phase" => 0.0,
                                        "upar_amplitude" => 1.0,
                                        "temperature_phase" => 0.0,
                                        "upar_phase" => 0.0,
                                       ),
    "vpa_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                          "density_amplitude" => 1.0,
                                          "temperature_amplitude" => 0.0,
                                          "density_phase" => 0.0,
                                          "upar_amplitude" => 0.0,
                                          "temperature_phase" => 0.0,
                                          "upar_phase" => 0.0,
                                         ),
    "krook_collisions" => OptionsDict("use_krook" => true,
                                     ),
    "reactions" => OptionsDict("electron_ionization_frequency" => 0.0,
                               "ionization_frequency" => 0.5,
                               "charge_exchange_frequency" => 0.75,
                              ),
    "ion_source_1" => OptionsDict("active" => true,
                                  "z_profile" => "gaussian",
                                  "z_width" => 0.25,
                                  "source_strength" => 2.0,
                                  "source_T" => 2.0,
                                 ),
    "ion_source_2" => OptionsDict("active" => true,
                                  "z_profile" => "wall_exp_decay",
                                  "z_width" => 0.25,
                                  "source_strength" => 0.5,
                                  "source_T" => 0.2,
                                 ),
    "timestepping" => OptionsDict("type" => "SSPRK4",
                                  "nstep" => 20000,
                                  "dt" => 1.0e-4,
                                  "nwrite" => 2500,
                                  "nwrite_dfns" => 2500,
                                  "steady_state_residual" => true,
                                 ),
    "ion_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0,
                                              ),
    "electron_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0,
                                                   ),
   )

# Test use distributed-memory when possible
boltzmann_input["z"]["nelement_local"] = boltzmann_input["z"]["nelement"] ÷ gcd(boltzmann_input["z"]["nelement"], global_size[])

kinetic_input = deepcopy(boltzmann_input)
kinetic_input["output"]["run_name"] = "kinetic_electron_test"
kinetic_input["composition"]["electron_physics"] = "kinetic_electrons"
kinetic_input["timestepping"] = OptionsDict("type" => "PareschiRusso2(2,2,2)",
                                            "implicit_electron_advance" => false,
                                            "implicit_electron_ppar" => true,
                                            "implicit_ion_advance" => false,
                                            "implicit_vpa_advection" => false,
                                            "nstep" => 100,
                                            "dt" => 1.0e-5,
                                            "nwrite" => 100,
                                            "nwrite_dfns" => 100,
                                           )

kinetic_input["electron_timestepping"] = OptionsDict("nstep" => 5000000,
                                                     "dt" => 2.0e-5,
                                                     "nwrite" => 10000,
                                                     "nwrite_dfns" => 100000,
                                                     "decrease_dt_iteration_threshold" => 5000,
                                                     "increase_dt_iteration_threshold" => 0,
                                                     "cap_factor_ion_dt" => 10.0,
                                                     "initialization_residual_value" => 1.0e0,
                                                     "converged_residual_value" => 1.0e-1,
                                                    )

kinetic_input["nonlinear_solver"] = OptionsDict("nonlinear_max_iterations" => 1000,
                                                "rtol" => 1.0e-8,
                                                "atol" => 1.0e-14,
                                                "linear_restart" => 5,
                                                "preconditioner_update_interval" => 1000,
                                               )


"""
Run a test for a single set of parameters
"""
function run_test()
    test_output_directory = get_MPI_tempdir()

    this_boltzmann_input = deepcopy(boltzmann_input)
    this_boltzmann_input["output"]["base_directory"] = test_output_directory

    this_kinetic_input = deepcopy(kinetic_input)
    this_kinetic_input["output"]["base_directory"] = test_output_directory

    # Provide some progress info
    println("    - testing kinetic electrons")

    # Suppress console output while running? Test is pretty long, so maybe better to leave
    # intermediate output visible. Leaving `quietoutput()` commented out for now...
    quietoutput() do
        run_moment_kinetics(this_boltzmann_input)

        restart_from_directory = joinpath(this_boltzmann_input["output"]["base_directory"], this_boltzmann_input["output"]["run_name"])
        restart_from_file_pattern = this_boltzmann_input["output"]["run_name"] * ".dfns*.h5"
        restart_from_file = glob(restart_from_file_pattern, restart_from_directory)[1]

        # run kinetic electron simulation
        run_moment_kinetics(this_kinetic_input; restart=restart_from_file)
    end

    if global_rank[] == 0
        # Load and analyse output
        #########################

        path = joinpath(realpath(this_kinetic_input["output"]["base_directory"]), this_kinetic_input["output"]["run_name"])

        # open the output file(s)
        run_info = get_run_info_no_setup(path, dfns=true)

        # load fields data
        Ez = postproc_load_variable(run_info, "Ez")[:,1,:]
        vthe = postproc_load_variable(run_info, "electron_thermal_speed")[:,1,:]
        electron_advance_linear_iterations = postproc_load_variable(run_info, "electron_advance_linear_iterations")[end]

        close_run_info(run_info)

        # Regression test
        # Benchmark data generated in serial on Linux
        expected_Ez = [-0.5990683230706185 -1.136483186157602;
                       -0.4944296396481284 -0.9873296990705788;
                       -0.30889032954504736 -0.6694380824928302;
                       -0.2064830747303776 -0.4471331690708596;
                       -0.21232457328748663 -0.423069171542538;
                       -0.18233875912042674 -0.3586467595624931;
                       -0.16711429522309232 -0.3018272987758344;
                       -0.16920776495088916 -0.27814384649305496;
                       -0.1629417555658927 -0.26124630661090814;
                       -0.16619150334079993 -0.2572789330163811;
                       -0.15918194883360942 -0.23720078037362732;
                       -0.14034706409006803 -0.20520396656341475;
                       -0.12602184032280567 -0.1827016549071128;
                       -0.10928716440800472 -0.15808919669899502;
                       -0.07053969674257217 -0.10137753767917096;
                       -0.0249577746169536 -0.0358411459260082;
                       -2.8327303308330514e-15 -2.0803303361189427e-5;
                       0.024957774616960776 0.03584490974053962;
                       0.07053969674257636 0.1013692898656727;
                       0.10928716440799909 0.15807862358546687;
                       0.1260218403227975 0.18263049748179466;
                       0.1403470640900294 0.20516566362571026;
                       0.1591819488336015 0.23711236692241613;
                       0.16619150334082114 0.257126146434857;
                       0.16294175556587748 0.2609881259705107;
                       0.16920776495090983 0.2778978154805798;
                       0.1671142952230893 0.3015349192528757;
                       0.1823387591204167 0.3585291689672981;
                       0.21232457328753865 0.4231179549656996;
                       0.20648307473037922 0.44816400221269476;
                       0.3088903295450278 0.6716787105435247;
                       0.4944296396481271 0.9861165590258743;
                       0.5990683230705801 1.1300034111861956]
        expected_vthe = [22.64555285302391 22.485481713141688;
                         23.763411647653097 23.63281883616836;
                         25.26907160117684 25.181703459470448;
                         26.17920352818247 26.12461016686916;
                         26.514772631426933 26.476018852279974;
                         26.798783188585713 26.774387562937218;
                         27.202255545479264 27.203662204308202;
                         27.50424749120107 27.527732850637264;
                         27.630498656270504 27.6642323848215;
                         27.748483758260697 27.79134809261204;
                         27.933760382468346 27.990808336620802;
                         28.08611508251559 28.153978618442775;
                         28.14959662643782 28.221734439130564;
                         28.207730844115044 28.283677711828023;
                         28.28567669896009 28.36634261525836;
                         28.32728392065335 28.410489883644782;
                         28.331064506972027 28.41437629072209;
                         28.32729968986601 28.41050992096321;
                         28.285678151542136 28.366352683865195;
                         28.207765527709956 28.28373408727703;
                         28.149604559462947 28.221771261090687;
                         28.086248527111163 28.154158507899695;
                         27.933979289064936 27.991103719847732;
                         27.74906125092813 27.792046191405188;
                         27.631210333523736 27.66508092926101;
                         27.505479130159543 27.529115937508752;
                         27.20422756527604 27.20578114592589;
                         26.801712351383053 26.77740066591359;
                         26.517644511297203 26.478915386575462;
                         26.18176436913143 26.127099000267552;
                         25.26635932097994 25.178676836919877;
                         23.756593489029708 23.625697695979085;
                         22.64390166090378 22.48400980852866]

        if expected_Ez == nothing
            # Error: no expected input provided
            println("data tested would be: Ez=", Ez)
            @test false
        else
            @test elementwise_isapprox(Ez, expected_Ez, rtol=0.0, atol=2.0e-6)
        end
        if expected_vthe == nothing
            # Error: no expected input provided
            println("data tested would be: vthe=", vthe)
            @test false
        else
            @test elementwise_isapprox(vthe, expected_vthe, rtol=1.0e-6, atol=0.0)
        end

        # Iteration counts are fairly inconsistent, but it's good to check that they at
        # least don't unexpectedly increase by an order of magnitude.
        # Expected iteration count is from a serial run on Linux.
        expected_electron_advance_linear_iterations = 49307
        @test electron_advance_linear_iterations < 2 * expected_electron_advance_linear_iterations
        if !(electron_advance_linear_iterations < 2 * expected_electron_advance_linear_iterations)
            println("electron_advance_linear_iterations=$electron_advance_linear_iterations was greater than twice the expected $expected_electron_advance_linear_iterations.")
        end
    end

    if global_rank[] == 0
        # Delete output directory to avoid using too much disk space
        rm(realpath(test_output_directory); recursive=true)
    end
end

function runtests()
    @testset "kinetic electrons" begin
        println("Kinetic electron tests")
        run_test()
    end
    return nothing
end

end # KineticElectronsTests


using .KineticElectronsTests

KineticElectronsTests.runtests()
