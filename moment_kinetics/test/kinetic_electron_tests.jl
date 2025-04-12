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
                                            "kinetic_electron_solver" => "implicit_ppar_implicit_pseudotimestep",
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
                expected_Ez = [-0.6041204601483995 -1.1031866332806448;
                               -0.4899985007615793 -0.9702557521523882;
                               -0.30157125359653525 -0.6643237428722496;
                               -0.20954461530743507 -0.448626721111969;
                               -0.21725774052171043 -0.4277318601037891;
                               -0.18018470169411022 -0.3565527283262676;
                               -0.16701297966245815 -0.3017761888449479;
                               -0.17029852097347278 -0.27919190408161193;
                               -0.16479810222163319 -0.26349777285866993;
                               -0.16650177136254102 -0.2578342975000088;
                               -0.1582591452248845 -0.2364406501155452;
                               -0.1399685108558434 -0.2048400200281007;
                               -0.12562263699714843 -0.1822002667319203;
                               -0.10941855412335265 -0.15825017226759588;
                               -0.07065980133329615 -0.10154006187571918;
                               -0.02509457740112119 -0.03600071035464205;
                               -4.9183836139672385e-15 -1.3413867913678009e-15;
                               0.02509457740109921 0.036000710354635534;
                               0.07065980133330539 0.10154006187572036;
                               0.10941855412336662 0.15825017226760096;
                               0.12562263699711657 0.1822002667318202;
                               0.13996851085584108 0.20484002002811758;
                               0.15825914522490123 0.23644065011557544;
                               0.16650177136252906 0.2578342974999823;
                               0.16479810222160715 0.26349777285869874;
                               0.17029852097352824 0.27919190408166017;
                               0.16701297966244733 0.3017761888449203;
                               0.18018470169405 0.35655272832620327;
                               0.2172577405217179 0.42773186010381914;
                               0.20954461530746524 0.4486267211119983;
                               0.30157125359655934 0.6643237428722507;
                               0.48999850076161194 0.9702557521524036;
                               0.6041204601484808 1.1031866332808118]
                expected_vthe = [22.701979703912098 22.5411136516573;
                                 23.775975068878136 23.6447117144965;
                                 25.269573404652828 25.182460531284597;
                                 26.180827153256086 26.126134875486123;
                                 26.51309766224318 26.474059474646683;
                                 26.79872099105839 26.774324987310887;
                                 27.201519125627595 27.202888862469653;
                                 27.504241804407613 27.527721913513908;
                                 27.62917500457502 27.662892119742363;
                                 27.748959650467643 27.7918012928706;
                                 27.93406504180257 27.99102536351066;
                                 28.087581454272264 28.155353828397885;
                                 28.149676510047023 28.22169681984343;
                                 28.209108374685112 28.285001462447372;
                                 28.28640026687845 28.366967786478135;
                                 28.328755126231776 28.411895167214865;
                                 28.33129226306051 28.414467418812272;
                                 28.328755126231776 28.411895167214773;
                                 28.286400266878456 28.366967786478064;
                                 28.20910837468512 28.285001462447372;
                                 28.14967651004702 28.22169681984338;
                                 28.087581454272268 28.15535382839796;
                                 27.934065041802572 27.991025363510623;
                                 27.748959650467654 27.791801292870637;
                                 27.62917500457503 27.6628921197423;
                                 27.504241804407616 27.52772191351386;
                                 27.201519125627602 27.202888862469674;
                                 26.79872099105839 26.774324987310933;
                                 26.513097662243187 26.474059474646687;
                                 26.18082715325609 26.12613487548611;
                                 25.269573404652864 25.182460531284644;
                                 23.775975068878168 23.644711714496555;
                                 22.701979703912126 22.54111365165734]

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
