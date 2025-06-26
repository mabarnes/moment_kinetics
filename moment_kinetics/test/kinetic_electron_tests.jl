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
    "output" => OptionsDict("run_name" => "kinetic_electron_test_boltzmann_initialisation"),
    "evolve_moments" => OptionsDict("density" => true,
                                    "moments_conservation" => true,
                                    "parallel_flow" => true,
                                    "pressure" => true),
    "r" => OptionsDict("ngrid" => 1,
                       "nelement" => 1),
    "z" => OptionsDict("ngrid" => 5,
                       "discretization" => "gausslegendre_pseudospectral",
                       "nelement" => 8,
                       "bc" => "wall"),
    "vpa" => OptionsDict("ngrid" => 6,
                         "discretization" => "gausslegendre_pseudospectral",
                         "nelement" => 17,
                         "L" => 41.569219381653056,
                         "bc" => "zero",
                         "element_spacing_option" => "coarse_tails8.660254037844386"),
    "composition" => OptionsDict("T_e" => 0.2,
                                 "n_ion_species" => 1,
                                 "n_neutral_species" => 0),
    "ion_species_1" => OptionsDict("initial_temperature" => 0.06666666666666667,
                                   "initial_density" => 1.0),
    "z_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                        "density_amplitude" => 1.0,
                                        "temperature_amplitude" => 0.0,
                                        "density_phase" => 0.0,
                                        "upar_amplitude" => 1.4142135623730951,
                                        "temperature_phase" => 0.0,
                                        "upar_phase" => 0.0),
    "vpa_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                          "density_amplitude" => 1.0,
                                          "temperature_amplitude" => 0.0,
                                          "density_phase" => 0.0,
                                          "upar_amplitude" => 0.0,
                                          "temperature_phase" => 0.0,
                                          "upar_phase" => 0.0),
    "krook_collisions" => OptionsDict("use_krook" => true),
    "reactions" => OptionsDict("electron_ionization_frequency" => 0.0,
                               "ionization_frequency" => 0.7071067811865476,
                               "charge_exchange_frequency" => 1.0606601717798214),
    "ion_source_1" => OptionsDict("active" => true,
                                  "z_profile" => "gaussian",
                                  "z_width" => 0.25,
                                  "source_strength" => 2.8284271247461903,
                                  "source_T" => 2.0),
    "ion_source_2" => OptionsDict("active" => true,
                                  "z_profile" => "wall_exp_decay",
                                  "z_width" => 0.25,
                                  "source_strength" => 0.7071067811865476,
                                  "source_T" => 0.2),
    "timestepping" => OptionsDict("type" => "SSPRK4",
                                  "nstep" => 20000,
                                  "dt" => 7.071067811865475e-5,
                                  "nwrite" => 2500,
                                  "nwrite_dfns" => 2500,
                                  "steady_state_residual" => true),
    "ion_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0),
    "electron_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0),
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
                                            "kinetic_electron_solver" => "implicit_p_implicit_pseudotimestep",
                                            "implicit_ion_advance" => false,
                                            "implicit_vpa_advection" => false,
                                            "nstep" => 100,
                                            "dt" => 7.0710678118654756e-6,
                                            "nwrite" => 100,
                                            "nwrite_dfns" => 100,
                                           )

kinetic_input["electron_timestepping"] = OptionsDict("nstep" => 5000000,
                                                     "dt" => 1.4142135623730951e-5,
                                                     "maximum_dt" => Inf,
                                                     "nwrite" => 10000,
                                                     "nwrite_dfns" => 100000,
                                                     "decrease_dt_iteration_threshold" => 5000,
                                                     "increase_dt_iteration_threshold" => 0,
                                                     "cap_factor_ion_dt" => 10.0,
                                                     "initialization_residual_value" => 1.0e0,
                                                     "converged_residual_value" => 0.1,
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
kinetic_input_adaptive_timestep["timestepping"]["maximum_dt"] = 7.0710678118654756e-6


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
                expected_Ez = [-1.0871513260418357 -1.0947537193214023;
                               -0.9538884304781137 -0.9634159093914267;
                               -0.6478148195310983 -0.6576463140012202;
                               -0.43303997374397113 -0.442623774961247;
                               -0.41327695255309155 -0.4233105731914575;
                               -0.3398049527625787 -0.34959349683499225;
                               -0.2861951654835422 -0.2953204538608628;
                               -0.26536684639491576 -0.27340052266275994;
                               -0.24993654678855057 -0.25756301086344857;
                               -0.2463960432506467 -0.2531168853033742;
                               -0.2269862411857663 -0.2325060509172384;
                               -0.19742081692578775 -0.201728318587799;
                               -0.17552325020116863 -0.1793339490862449;
                               -0.15297568593105343 -0.1560459987118946;
                               -0.09829990758178349 -0.10019961172824715;
                               -0.03487260945230809 -0.035533852113549626;
                               8.644100348662068e-15 -2.6081517363852597e-14;
                               0.03487260945230394 0.035533852113539516;
                               0.09829990758177094 0.10019961172824937;
                               0.15297568593106695 0.1560459987119035;
                               0.17552325020124174 0.17933394908630151;
                               0.19742081692580593 0.201728318587827;
                               0.22698624118573818 0.23250605091720783;
                               0.24639604325065867 0.2531168853033884;
                               0.2499365467885465 0.2575630108634985;
                               0.2653668463949038 0.27340052266273135;
                               0.2861951654835539 0.29532045386087996;
                               0.33980495276256456 0.3495934968349792;
                               0.4132769525530601 0.423310573191383;
                               0.43303997374399944 0.44262377496126837;
                               0.6478148195311217 0.6576463140012321;
                               0.9538884304780975 0.9634159093913963;
                               1.0871513260417849 1.0947537193213208]
                expected_vthe = [18.637814808315827 18.498808185305915;
                                 19.49582587276955 19.38220334304171;
                                 20.687856737128122 20.61223815586221;
                                 21.411658934552694 21.364099746944127;
                                 21.672847179676697 21.638882863545714;
                                 21.896771657010312 21.875469753525703;
                                 22.209292752345632 22.210117121084863;
                                 22.442689151892257 22.462283397135014;
                                 22.538316510612656 22.56651298921619;
                                 22.63065735641257 22.66649629606997;
                                 22.773294856432212 22.820835711897246;
                                 22.892256865931568 22.948749951301405;
                                 22.940374308665586 23.000352679410515;
                                 22.986656597288555 23.049870304701027;
                                 23.046947428625213 23.114026933548317;
                                 23.080015634590158 23.149254613188496;
                                 23.082021874167225 23.151260513878995;
                                 23.08001563459016 23.149254613188493;
                                 23.04694742862521 23.11402693354831;
                                 22.986656597288555 23.049870304701052;
                                 22.940374308665586 23.000352679410515;
                                 22.89225686593156 22.948749951301437;
                                 22.773294856432223 22.82083571189723;
                                 22.630657356412584 22.66649629606997;
                                 22.538316510612667 22.56651298921618;
                                 22.44268915189226 22.46228339713497;
                                 22.209292752345643 22.210117121084846;
                                 21.896771657010316 21.875469753525653;
                                 21.67284717967671 21.63888286354574;
                                 21.411658934552708 21.364099746944127;
                                 20.687856737128147 20.612238155862244;
                                 19.495825872769583 19.382203343041837;
                                 18.6378148083159 18.498808185306135]

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
