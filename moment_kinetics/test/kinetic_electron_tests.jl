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
                                                     "maximum_dt" => Inf,
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

    if ("nelement_local" ∈ keys(kinetic_input["z"])
        && kinetic_input["z"]["nelement"] ÷ kinetic_input["z"]["nelement_local"] < global_size[]
       )
        # Using shared-memory parallelism, so should be using ADI preconditioner
        adi_precon_iterations_values = (1,2)
    else
        adi_precon_iterations_values = -1
    end

    for (this_kinetic_input, label, tol) ∈ ((deepcopy(kinetic_input), "", 1.0e-6),
                                             (deepcopy(kinetic_input_adaptive_timestep), ", adaptive timestep", 1.0e-4))
        this_kinetic_input["output"]["base_directory"] = test_output_directory

        for adi_precon_iterations ∈ adi_precon_iterations_values
            if adi_precon_iterations < 0
                # Provide some progress info
                println("    - testing kinetic electrons$label")
            else
                this_kinetic_input["nonlinear_solver"]["adi_precon_iterations"] = adi_precon_iterations

                # Provide some progress info
                println("    - testing kinetic electrons $adi_precon_iterations ADI iterations$label")
            end

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
                # Benchmark data generated in serial on Linux with the LU preconditioner
                expected_Ez = [-0.6033424563381473 -1.105845120863604;
                               -0.4895419697191283 -0.9734213545329831;
                               -0.30168823643837755 -0.6662833575273305;
                               -0.20976461471715463 -0.44877273987873173;
                               -0.2175960342447592 -0.42827069843745663;
                               -0.1804495496290851 -0.3568833465810143;
                               -0.16688563950898355 -0.30143398994207093;
                               -0.170318966939107 -0.2789900249568236;
                               -0.1650028468535594 -0.26343244994867243;
                               -0.16673227810832084 -0.2578755213890533;
                               -0.15843447937814578 -0.2365025383210448;
                               -0.14003947178553747 -0.20483477141409565;
                               -0.125672015633878 -0.1821809228942641;
                               -0.10947740486614427 -0.15826273873202973;
                               -0.07072471924440067 -0.10158128936681422;
                               -0.025126135733338766 -0.03602496693598949;
                               6.52676233840434e-8 7.637703021912418e-9;
                               0.025126279789117134 0.036025105432062894;
                               0.07072487668116205 0.1015814175333404;
                               0.10947760565872611 0.1582629072215589;
                               0.125672125478251 0.18218086497398145;
                               0.14003970052336284 0.20483494379853798;
                               0.1584344537120472 0.23650232137173438;
                               0.16673255277088347 0.25787557826742674;
                               0.16500281037501285 0.26343245699239154;
                               0.17031906011855846 0.27898947421456344;
                               0.16688647584126126 0.30143428183481996;
                               0.18044756568512157 0.3568788442259826;
                               0.21759682994322424 0.42826892606811484;
                               0.20976711189674635 0.4487738275151079;
                               0.3016853245160196 0.6662715723952268;
                               0.48953776156708956 0.9734023826407825;
                               0.6033395995972192 1.1058283447607065]
                expected_vthe = [22.686407468089133 22.52535365092282;
                                 23.767450540593128 23.63603824478141;
                                 25.270041373369654 25.1827760711247;
                                 26.18346239621591 26.12862398979303;
                                 26.51535081692685 26.476257880079608;
                                 26.801299607080647 26.776863619533138;
                                 27.203517776214046 27.204912269531828;
                                 27.505570687008557 27.52913575029894;
                                 27.629992670843336 27.663814878106542;
                                 27.74951027515746 27.792479018979055;
                                 27.934117832250404 27.991209297910466;
                                 28.087401411335957 28.15530235798202;
                                 28.149328984670746 28.221469919579203;
                                 28.208742497593338 28.28476062467688;
                                 28.285906822152512 28.366594024897104;
                                 28.32824453742463 28.411507690825665;
                                 28.33072134274691 28.414013220914725;
                                 28.32824462013829 28.411507778447735;
                                 28.285906937557698 28.366594161394122;
                                 28.208742771972684 28.2847609380403;
                                 28.149329237421927 28.221470240333534;
                                 28.087401973888298 28.15530297507487;
                                 27.934118587496034 27.99121017239033;
                                 27.749511756290957 27.79248065303357;
                                 27.62999431921291 27.663816798041353;
                                 27.50557278991246 27.529138110079824;
                                 27.203521667306514 27.204916632592532;
                                 26.801306951412922 26.776871591867334;
                                 26.51536367187478 26.476271051253967;
                                 26.183476399089166 26.128639185253913;
                                 25.270071887764608 25.182807853359904;
                                 23.7675264094922 23.636114533721084;
                                 22.686518428630276 22.52546504637422]

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
