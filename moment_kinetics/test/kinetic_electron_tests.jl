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
                                            "kinetic_ion_solver" => "full_explicit_ion_advance",
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
                                             (deepcopy(kinetic_input_adaptive_timestep), ", adaptive timestep", 2.0e-4))
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
                expected_Ez = [-1.1725882218891623 -1.1776336622726766;
                               -1.0238608089434689 -1.0322062236537615;
                               -0.6635650715847973 -0.6735224968546698;
                               -0.4281091018911448 -0.43818164733207154;
                               -0.4307326771084564 -0.44111594822983197;
                               -0.34263847904677724 -0.3527698667635337;
                               -0.2848762661779086 -0.2942649171566735;
                               -0.26345168742207964 -0.2716267222973865;
                               -0.24437326252130007 -0.25211993616621897;
                               -0.2441434440827379 -0.25093038261480466;
                               -0.2256969989670498 -0.2312604708231275;
                               -0.19613183153304686 -0.20047448426408893;
                               -0.173688463060076 -0.17753980131185232;
                               -0.15171664075925836 -0.15481540597366636;
                               -0.09749011345887852 -0.09941190105001846;
                               -0.0345345905614875 -0.03520384775311136;
                               1.0136720856087798e-14 1.2820071743800603e-14;
                               0.03453459056147705 0.035203847753093;
                               0.0974901134588791 0.09941190105001832;
                               0.15171664075927124 0.15481540597370938;
                               0.1736884630600449 0.1775398013117877;
                               0.1961318315330768 0.20047448426410092;
                               0.22569699896706918 0.23126047082315318;
                               0.2441434440826907 0.25093038261474365;
                               0.2443732625212673 0.25211993616616396;
                               0.2634516874220768 0.27162672229737905;
                               0.28487626617792133 0.29426491715670267;
                               0.34263847904677097 0.35276986676353034;
                               0.4307326771084532 0.44111594822984607;
                               0.4281091018911668 0.43818164733210624;
                               0.6635650715848012 0.673522496854653;
                               1.0238608089434755 1.0322062236537737;
                               1.1725882218891814 1.1776336622727863]
                expected_vthe = [18.676826599688518 18.53470329289466;
                                 19.525199576911888 19.407756666941655;
                                 20.70872008146677 20.629767577551725;
                                 21.425373425297394 21.375624256352353;
                                 21.681774750114645 21.646269036297042;
                                 21.901226657252224 21.87888290595702;
                                 22.205588593755774 22.20610827501164;
                                 22.432224302260742 22.451919858547495;
                                 22.524426687095215 22.552888718220892;
                                 22.61409579788143 22.65030245174724;
                                 22.75198657730116 22.800019400346628;
                                 22.867599820270556 22.924662790410167;
                                 22.914088377045655 22.97466424154602;
                                 22.959357906177164 23.023211976762504;
                                 23.017927918956982 23.085693103965276;
                                 23.050232625793747 23.120203221415636;
                                 23.0519641128348 23.121923892065464;
                                 23.050232625793743 23.120203221415636;
                                 23.01792791895698 23.085693103965244;
                                 22.95935790617716 23.023211976762486;
                                 22.91408837704565 22.974664241545955;
                                 22.867599820270556 22.924662790410167;
                                 22.751986577301167 22.800019400346613;
                                 22.614095797881433 22.65030245174726;
                                 22.52442668709523 22.552888718220917;
                                 22.432224302260742 22.45191985854757;
                                 22.205588593755785 22.206108275011673;
                                 21.90122665725223 21.87888290595702;
                                 21.681774750114652 21.646269036297035;
                                 21.42537342529739 21.375624256352353;
                                 20.70872008146678 20.629767577551707;
                                 19.525199576911906 19.407756666941705;
                                 18.67682659968857 18.534703292894708]

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
