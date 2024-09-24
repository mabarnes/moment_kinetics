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
                                                     "dt" => 1.0e-5,
                                                     "nwrite" => 10000,
                                                     "nwrite_dfns" => 100000,
                                                     "decrease_dt_iteration_threshold" => 1000,
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
        expected_Ez = [-0.5990683230706185 -0.6042082363495851;
                       -0.4944296396481284 -0.49692371894536586;
                       -0.30889032954504736 -0.3090990586904173;
                       -0.2064830747303776 -0.20700297720010077;
                       -0.21232457328748663 -0.2132748045598696;
                       -0.18233875912042674 -0.18276920923500758;
                       -0.16711429522309232 -0.1674324272230308;
                       -0.16920776495088916 -0.16937992443371716;
                       -0.1629417555658927 -0.16309341722744303;
                       -0.16619150334079993 -0.16633546753735795;
                       -0.15918194883360942 -0.15931554370144113;
                       -0.14034706409006803 -0.140469880250037;
                       -0.12602184032280567 -0.12613381924054493;
                       -0.10928716440800472 -0.10938345602505639;
                       -0.07053969674257217 -0.0706024520856333;
                       -0.0249577746169536 -0.024980098134854842;
                       -2.8327303308330514e-15 -1.599033453711614e-10;
                       0.024957774616960776 0.02498009782733815;
                       0.07053969674257636 0.07060245115760132;
                       0.10928716440799909 0.10938345732933795;
                       0.1260218403227975 0.1261338225947928;
                       0.1403470640900294 0.14046988178255268;
                       0.1591819488336015 0.15931556545456152;
                       0.16619150334082114 0.1663353993955267;
                       0.16294175556587748 0.16309307445724816;
                       0.16920776495090983 0.1693805039915145;
                       0.1671142952230893 0.1674318780154963;
                       0.1823387591204167 0.18277420263305205;
                       0.21232457328753865 0.21326329266495697;
                       0.20648307473037922 0.20700517064938181;
                       0.3088903295450278 0.3091144991453789;
                       0.4944296396481271 0.49684270193048663;
                       0.5990683230705801 0.6040141042995336]
        expected_vthe = [27.08122333732766 27.083668406411196;
                         27.087128061238488 27.08840157326006;
                         27.090525010446868 27.090443986816897;
                         27.091202856161452 27.0914901864659;
                         27.09265674296987 27.093297466503625;
                         27.093298138334738 27.09337068853881;
                         27.094377689895747 27.094548022524926;
                         27.09501542767647 27.095170446421935;
                         27.095227831625575 27.095304545176944;
                         27.095420218946682 27.09555512096241;
                         27.095754478126825 27.095876494374046;
                         27.096054218271775 27.096188914603825;
                         27.096199500698383 27.096294431476554;
                         27.09632238748948 27.096423453543142;
                         27.096502792691805 27.096594041947167;
                         27.096597492028636 27.096694147970585;
                         27.096610989303674 27.096702959927107;
                         27.096597492397745 27.096694148339555;
                         27.096502794930903 27.096594044186332;
                         27.096322390449956 27.09642345650393;
                         27.096199499205674 27.096294429984052;
                         27.09605421760595 27.096188913937898;
                         27.095754438597055 27.095876454845936;
                         27.09542019655419 27.095555098545283;
                         27.095228009815475 27.095304723869976;
                         27.095015217848847 27.09517023619458;
                         27.094377437638478 27.09454777080713;
                         27.093294828184774 27.093367377705533;
                         27.092639150183448 27.09327987116632;
                         27.0912092735745 27.091496606764487;
                         27.09048496370012 27.090403937882265;
                         27.08714601914595 27.08841951855733;
                         27.08144246136634 27.08388753119234]

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
        expected_electron_advance_linear_iterations = 10695
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
    if Sys.isapple()
        @testset_skip "MINPACK is broken on macOS (https://github.com/sglyon/MINPACK.jl/issues/18)" "non-linear solvers" begin
        end
        return nothing
    end
    @testset "kinetic electrons" begin
        println("Kinetic electron tests")
        run_test()
    end
    return nothing
end

end # KineticElectronsTests


using .KineticElectronsTests

KineticElectronsTests.runtests()
