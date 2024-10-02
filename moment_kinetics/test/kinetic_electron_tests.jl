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
                                                     "dt" => 5.0e-6,
                                                     "nwrite" => 10000,
                                                     "nwrite_dfns" => 100000,
                                                     "decrease_dt_iteration_threshold" => 5000,
                                                     "increase_dt_iteration_threshold" => 0,
                                                     "cap_factor_ion_dt" => 10.0,
                                                     "initialization_residual_value" => 1.0e10,
                                                     "converged_residual_value" => 1.0e-1,
                                                    )

kinetic_input["nonlinear_solver"] = OptionsDict("nonlinear_max_iterations" => 1000,
                                                "rtol" => 1.0e-8,
                                                "atol" => 1.0e-14,
                                                "linear_restart" => 5,
                                                "preconditioner_update_interval" => 100,
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
        expected_Ez = [-0.5990683230706185 -0.604849806235434;
                       -0.4944296396481284 -0.49739671491727844;
                       -0.30889032954504736 -0.30924318765687464;
                       -0.2064830747303776 -0.20682475071884582;
                       -0.21232457328748663 -0.21299072376949116;
                       -0.18233875912042674 -0.18256905463006085;
                       -0.16711429522309232 -0.1673112962636778;
                       -0.16920776495088916 -0.1693227707158167;
                       -0.1629417555658927 -0.16304933113558318;
                       -0.16619150334079993 -0.16629539618289285;
                       -0.15918194883360942 -0.1592799009526323;
                       -0.14034706409006803 -0.140437217833422;
                       -0.12602184032280567 -0.12610387949683538;
                       -0.10928716440800472 -0.10935785133612701;
                       -0.07053969674257217 -0.07058573063123225;
                       -0.0249577746169536 -0.024974174596810936;
                       -2.8327303308330514e-15 -1.441401377024236e-10;
                       0.024957774616960776 0.02497417427570905;
                       0.07053969674257636 0.07058572965952663;
                       0.10928716440799909 0.10935785264749627;
                       0.1260218403227975 0.12610388283669527;
                       0.1403470640900294 0.1404372197714126;
                       0.1591819488336015 0.15927992284761766;
                       0.16619150334082114 0.1662953275454769;
                       0.16294175556587748 0.1630489871826757;
                       0.16920776495090983 0.1693233489685909;
                       0.1671142952230893 0.16731075590341918;
                       0.1823387591204167 0.1825740389953209;
                       0.21232457328753865 0.21297925141919793;
                       0.20648307473037922 0.20682690396901446;
                       0.3088903295450278 0.30925854110074175;
                       0.4944296396481271 0.49731601862961966;
                       0.5990683230705801 0.6046564647413697]
        expected_vthe = [27.08102229345079 27.08346736523219;
                         27.087730258479823 27.089003820908527;
                         27.091898844901323 27.09181784480061;
                         27.092455021687254 27.092742387764524;
                         27.09350739287911 27.094148133125078;
                         27.093817059011126 27.093889601910092;
                         27.09443981315218 27.094610141036807;
                         27.09484177005478 27.094996783801374;
                         27.094985914811055 27.0950626278904;
                         27.095122128675094 27.09525702879687;
                         27.09536357532887 27.09548558966323;
                         27.095582117080163 27.095716810823177;
                         27.09568783962135 27.09578276803757;
                         27.0957775472326 27.095878610625554;
                         27.095909169276535 27.09600041573683;
                         27.095978269355648 27.096074922150624;
                         27.095988166679223 27.096080134292468;
                         27.095978269713978 27.096074922508883;
                         27.095909171602027 27.096000418062378;
                         27.09577755035281 27.095878613746088;
                         27.095687838236376 27.095782766652857;
                         27.09558211622511 27.095716809968053;
                         27.09536353456768 27.09548554890375;
                         27.095122105596843 27.095257005693973;
                         27.094986093051983 27.09506280663278;
                         27.094841563692096 27.094996577040796;
                         27.094439553087433 27.094609881510113;
                         27.093813728418613 27.09388627063591;
                         27.093489818175936 27.094130555874184;
                         27.09246140309467 27.092748772044477;
                         27.09185903467811 27.09177803239964;
                         27.08774827015981 27.089021820036553;
                         27.081240668889404 27.0836857414255]

        if expected_Ez == nothing
            # Error: no expected input provided
            println("data tested would be: Ez=", Ez)
            @test false
        else
            @test isapprox(Ez, expected_Ez, rtol=1.0e-7, atol=1.0e-9)
        end
        if expected_vthe == nothing
            # Error: no expected input provided
            println("data tested would be: vthe=", vthe)
            @test false
        else
            @test isapprox(vthe, expected_vthe, rtol=2.0e-9, atol=0.0)
        end

        # Iteration counts are fairly inconsistent, but it's good to check that they at
        # least don't unexpectedly increase by an order of magnitude.
        # Expected iteration count is from a serial run on Linux.
        expected_electron_advance_linear_iterations = 11394
        @test electron_advance_linear_iterations < 2.0 * expected_electron_advance_linear_iterations
        if !(electron_advance_linear_iterations < 2.0 * expected_electron_advance_linear_iterations)
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
