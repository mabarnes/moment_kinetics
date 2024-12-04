module KineticElectronsTests

# Regression test with kinetic electrons, using wall boundary conditions, with recycling
# fraction less than 1 and a plasma source. Runs a Boltzmann electron simulation, restarts
# as a kinetic electron simulation, and checks the final Ez profile.

include("setup.jl")

using moment_kinetics.communication
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
if global_size[] % 2 == 0
    # Divide by 2 so that we use shared memory when running in parallel, and so test the
    # ADI preconditioner.
    procs_to_divide_by = global_size[] ÷ 2
else
    procs_to_divide_by = global_size[]
end
boltzmann_input["z"]["nelement_local"] = boltzmann_input["z"]["nelement"] ÷ gcd(boltzmann_input["z"]["nelement"], procs_to_divide_by)

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
                                                "preconditioner_update_interval" => 100,
                                               )

kinetic_input_adaptive_timestep = deepcopy(kinetic_input)
kinetic_input_adaptive_timestep["output"]["run_name"] = "kinetic_electron_adaptive_timestep_test"
kinetic_input_adaptive_timestep["timestepping"]["type"] = "KennedyCarpenterARK324"
kinetic_input_adaptive_timestep["timestepping"]["maximum_dt"] = 1.0e-5


"""
Run a test for a single set of parameters
"""
function run_test()
    test_output_directory = get_MPI_tempdir()

    this_boltzmann_input = deepcopy(boltzmann_input)
    this_boltzmann_input["output"]["base_directory"] = test_output_directory

    # Suppress console output while running.
    quietoutput() do
        run_moment_kinetics(this_boltzmann_input)
    end

    for (this_kinetic_input, label, tol) ∈ ((deepcopy(kinetic_input), "", 1.0e-6),
                                             (deepcopy(kinetic_input_adaptive_timestep), "adaptive timestep", 1.0e-4))
        # Provide some progress info
        println("    - testing kinetic electrons $label")

        this_kinetic_input["output"]["base_directory"] = test_output_directory

        # Suppress console output while running.
        quietoutput() do
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
            if global_size[] == 1
                # Serial solves use LU preconditioner
                expected_Ez = [-0.5990683230706185 -1.1053138725180998;
                               -0.4944296396481284 -0.9819332128466166;
                               -0.30889032954504736 -0.6745656961983237;
                               -0.2064830747303776 -0.4459531272930669;
                               -0.21232457328748663 -0.4253218487528007;
                               -0.18233875912042674 -0.3596054334022437;
                               -0.16711429522309232 -0.3021381799340685;
                               -0.16920776495088916 -0.2784335484692499;
                               -0.1629417555658927 -0.2612551389558109;
                               -0.16619150334079993 -0.2574841927015592;
                               -0.15918194883360942 -0.23740132549636406;
                               -0.14034706409006803 -0.20534503972256973;
                               -0.12602184032280567 -0.1827098539044343;
                               -0.10928716440800472 -0.1582133200686042;
                               -0.07053969674257217 -0.10145491369831482;
                               -0.0249577746169536 -0.03585934915825971;
                               -2.8327303308330514e-15 3.742211718942586e-14;
                               0.024957774616960776 0.03585934915827381;
                               0.07053969674257636 0.10145491369829167;
                               0.10928716440799909 0.15821332006862954;
                               0.1260218403227975 0.18270985390445083;
                               0.1403470640900294 0.20534503972250218;
                               0.1591819488336015 0.23740132549634094;
                               0.16619150334082114 0.2574841927015898;
                               0.16294175556587748 0.261255138955811;
                               0.16920776495090983 0.2784335484692798;
                               0.1671142952230893 0.3021381799340713;
                               0.1823387591204167 0.3596054334022252;
                               0.21232457328753865 0.4253218487528467;
                               0.20648307473037922 0.44595312729305947;
                               0.3088903295450278 0.6745656961983009;
                               0.4944296396481271 0.9819332128466268;
                               0.5990683230705801 1.1053138725180645]
                expected_vthe = [22.654024448490784 22.494016350356883;
                                 23.744503682730446 23.61361063067715;
                                 25.26061134578617 25.173128418725682;
                                 26.177253875120066 26.122412383901523;
                                 26.510545637302872 26.47158368991228;
                                 26.798827552847246 26.77429043464489;
                                 27.202535498354287 27.2038739551587;
                                 27.506373594650846 27.529813468465488;
                                 27.631027625644876 27.664719606410365;
                                 27.750902611036295 27.793759280909274;
                                 27.935780521313532 27.992775960575692;
                                 28.089380398280714 28.157198480516957;
                                 28.15152314377127 28.223553488629253;
                                 28.211115085781678 28.2870195116558;
                                 28.28856778918977 28.369130039283018;
                                 28.330972960680672 28.41411592647979;
                                 28.33351348538364 28.416680586218863;
                                 28.330972960680675 28.41411592647976;
                                 28.288567789189763 28.369130039283064;
                                 28.211115085781678 28.287019511655785;
                                 28.15152314377127 28.223553488629236;
                                 28.089380398280724 28.157198480516957;
                                 27.93578052131354 27.992775960575713;
                                 27.750902611036295 27.79375928090935;
                                 27.63102762564488 27.664719606410383;
                                 27.506373594650853 27.529813468465495;
                                 27.202535498354287 27.2038739551587;
                                 26.79882755284725 26.774290434644872;
                                 26.510545637302886 26.471583689912283;
                                 26.177253875120083 26.122412383901523;
                                 25.26061134578619 25.173128418725696;
                                 23.744503682730446 23.613610630677236;
                                 22.65402444849082 22.494016350356937]
            else
                # Parallel solves, which here use only shared-memory parallelism, use the ADI
                # preconditioner, which should be as accurate, but may give different results
                # within Newton-Krylov tolerances.
                expected_Ez = [-0.5990683230706185 -1.1053137071260657;
                               -0.4944296396481284 -0.9819330928307715;
                               -0.30889032954504736 -0.6745656725019216;
                               -0.2064830747303776 -0.44595313784207047;
                               -0.21232457328748663 -0.425321828548;
                               -0.18233875912042674 -0.3596054340570364;
                               -0.16711429522309232 -0.30213818089568956;
                               -0.16920776495088916 -0.27843354821637;
                               -0.1629417555658927 -0.2612551385019989;
                               -0.16619150334079993 -0.2574841930766524;
                               -0.15918194883360942 -0.23740132557788143;
                               -0.14034706409006803 -0.20534504018275174;
                               -0.12602184032280567 -0.18270985430997166;
                               -0.10928716440800472 -0.1582133189704785;
                               -0.07053969674257217 -0.101454914566153;
                               -0.0249577746169536 -0.035859347929368034;
                               -2.8327303308330514e-15 -4.536628997349189e-9;
                               0.024957774616960776 0.035859348624052545;
                               0.07053969674257636 0.10145491474282464;
                               0.10928716440799909 0.15821331955573922;
                               0.1260218403227975 0.18270985667178208;
                               0.1403470640900294 0.2053450392202274;
                               0.1591819488336015 0.23740132578753803;
                               0.16619150334082114 0.25748419283426127;
                               0.16294175556587748 0.2612551396310432;
                               0.16920776495090983 0.2784335479625835;
                               0.1671142952230893 0.3021381809909585;
                               0.1823387591204167 0.35960543399747075;
                               0.21232457328753865 0.4253218286915096;
                               0.20648307473037922 0.44595313782295487;
                               0.3088903295450278 0.6745656725300222;
                               0.4944296396481271 0.9819330927685747;
                               0.5990683230705801 1.1053137082172033]
                expected_vthe = [22.654024454479018 22.494016869931663;
                                 23.74450367962989 23.61361086266046;
                                 25.260611341892094 25.173128419566062;
                                 26.17725387357487 26.122412390676395;
                                 26.510545632956767 26.47158369227529;
                                 26.7988275507785 26.774290427357606;
                                 27.20253549703805 27.20387395613098;
                                 27.506373594719115 27.529813465559865;
                                 27.63102762567087 27.6647196112545;
                                 27.75090260968854 27.79375927764987;
                                 27.935780521822277 27.992775962652605;
                                 28.08938039775227 28.157198478502867;
                                 28.151523156278788 28.223553495610926;
                                 28.211115080270424 28.28701950947455;
                                 28.288567793141777 28.369130040934596;
                                 28.330972955353705 28.414115925374524;
                                 28.333513456094945 28.41668058720323;
                                 28.330972961606466 28.414115929999316;
                                 28.288567792143006 28.369130041232697;
                                 28.211115083430062 28.287019512466056;
                                 28.15152314952673 28.223553491119628;
                                 28.089380398299795 28.157198479157458;
                                 27.93578052229754 27.99277596224337;
                                 27.750902609816293 27.79375927871885;
                                 27.631027625671482 27.664719609967122;
                                 27.50637359506551 27.52981346582775;
                                 27.20253549697429 27.203873955958308;
                                 26.798827550864885 26.77429042759387;
                                 26.510545632587316 26.471583691722795;
                                 26.177253873758893 26.122412390844207;
                                 25.26061134158348 25.17312841929966;
                                 23.7445036798294 23.613610862832093;
                                 22.654024453873603 22.494016869407307]
            end

            if expected_Ez == nothing
                # Error: no expected input provided
                println("data tested would be: Ez=", Ez)
                @test false
            else
                @test elementwise_isapprox(Ez, expected_Ez, rtol=0.0, atol=2.0*tol)
            end
            if expected_vthe == nothing
                # Error: no expected input provided
                println("data tested would be: vthe=", vthe)
                @test false
            else
                @test elementwise_isapprox(vthe, expected_vthe, rtol=tol, atol=0.0)
            end

            # Iteration counts are fairly inconsistent, but it's good to check that they at
            # least don't unexpectedly increase by an order of magnitude.
            # Expected iteration count is from a serial run on Linux.
            expected_electron_advance_linear_iterations = 48716
            @test electron_advance_linear_iterations < 2 * expected_electron_advance_linear_iterations
            if !(electron_advance_linear_iterations < 2 * expected_electron_advance_linear_iterations)
                println("electron_advance_linear_iterations=$electron_advance_linear_iterations was greater than twice the expected $expected_electron_advance_linear_iterations.")
            end
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
