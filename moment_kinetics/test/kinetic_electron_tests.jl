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
                println("    - testing kinetic electrons $label")
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
                expected_Ez = [-0.5974877305606054 -1.1040344548872603;
                               -0.49399426014349546 -0.9813416066189391;
                               -0.30923257449352126 -0.6748283311978367;
                               -0.2067152536408228 -0.44624566424639206;
                               -0.21236207233466906 -0.4252575905173472;
                               -0.18210283213826237 -0.3594267209906836;
                               -0.16697427713052254 -0.30203552148563384;
                               -0.16950896359411624 -0.27878642224633565;
                               -0.16290688120701216 -0.26126443310271924;
                               -0.16583486350526067 -0.2571362703736024;
                               -0.1592530586758104 -0.23750844133661628;
                               -0.14037715225912714 -0.2054005951424109;
                               -0.125930140150202 -0.18263573686407794;
                               -0.10917177016228309 -0.15809921470240096;
                               -0.07059664981949007 -0.10152217515636708;
                               -0.025029940946091716 -0.03594082353385175;
                               -4.92193574812872e-15 5.816871271951027e-15;
                               0.025029940946095 0.03594082353385728;
                               0.0705966498194857 0.10152217515635517;
                               0.1091717701622755 0.15809921470239682;
                               0.12593014015020326 0.1826357368641122;
                               0.1403771522591299 0.20540059514239975;
                               0.1592530586758358 0.23750844133665577;
                               0.16583486350525747 0.2571362703736182;
                               0.16290688120698285 0.2612644331026281;
                               0.1695089635941299 0.27878642224634215;
                               0.16697427713050292 0.30203552148562235;
                               0.18210283213824596 0.359426720990645;
                               0.21236207233468643 0.42525759051737055;
                               0.20671525364083 0.4462456642463948;
                               0.30923257449352626 0.674828331197867;
                               0.49399426014348236 0.9813416066189327;
                               0.5974877305605455 1.1040344548870393]
                expected_vthe = [22.65500671742333 22.494999883931623;
                                 23.745013899614875 23.61415107488815;
                                 25.260567578719666 25.1731305153563;
                                 26.1773971864201 26.1225699000901;
                                 26.51061844968467 26.47166984825304;
                                 26.798965263934015 26.77445314978074;
                                 27.20279319823905 27.20414800877869;
                                 27.50670868757608 27.530172042480288;
                                 27.631441975660216 27.66515124522187;
                                 27.751331352573228 27.794211957804183;
                                 27.936274643780138 27.993294136006696;
                                 28.08991863328286 28.157762001986363;
                                 28.152092171573834 28.22414659642932;
                                 28.211691790434642 28.287622701512305;
                                 28.2891535742392 28.369743550998418;
                                 28.33155660907617 28.414728860050044;
                                 28.334104883973886 28.417300414464158;
                                 28.33155660907617 28.414728860050026;
                                 28.289153574239204 28.369743550998418;
                                 28.211691790434628 28.28762270151233;
                                 28.152092171573827 28.224146596429343;
                                 28.089918633282835 28.15776200198637;
                                 27.936274643780138 27.99329413600675;
                                 27.751331352573242 27.794211957804148;
                                 27.631441975660206 27.665151245221853;
                                 27.506708687576072 27.530172042480295;
                                 27.20279319823905 27.204148008778624;
                                 26.798965263934008 26.77445314978071;
                                 26.51061844968468 26.47166984825307;
                                 26.1773971864201 26.122569900090113;
                                 25.26056757871967 25.1731305153563;
                                 23.74501389961485 23.614151074888035;
                                 22.65500671742336 22.494999883931595]

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
