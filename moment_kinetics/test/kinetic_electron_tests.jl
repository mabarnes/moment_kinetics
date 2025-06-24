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
                expected_Ez = [-1.0893489409351322 -1.096842883323186;
                               -0.9536586385931908 -0.9631448543384181;
                               -0.6477059711749636 -0.6575326613707106;
                               -0.4323871489049043 -0.44199589539228207;
                               -0.41047309553007894 -0.42054017056698234;
                               -0.3398309779732648 -0.34962673099176694;
                               -0.2861141357467261 -0.29524914266226093;
                               -0.26529154703154006 -0.2733311800202578;
                               -0.25037979569122654 -0.2580078505470731;
                               -0.24623829868283093 -0.25296253855845474;
                               -0.22691926459109818 -0.23243944153464977;
                               -0.19735583437735973 -0.20166508382438852;
                               -0.17572166715266438 -0.1795343166614933;
                               -0.1529513894940026 -0.15602316105319733;
                               -0.09827388375022258 -0.10017436970767879;
                               -0.03485240281783157 -0.035513851748922694;
                               -2.6231921799602032e-14 -1.505365171000642e-14;
                               0.034852402817833895 0.03551385174893684;
                               0.09827388375022911 0.1001743697076772;
                               0.15295138949400447 0.15602316105319133;
                               0.17572166715266602 0.17953431666150194;
                               0.19735583437735593 0.20166508382437795;
                               0.22691926459111977 0.23243944153466506;
                               0.2462382986828151 0.25296253855844314;
                               0.2503797956911993 0.2580078505470559;
                               0.26529154703153734 0.27333118002027035;
                               0.2861141357467314 0.2952491426622479;
                               0.33983097797328277 0.3496267309917725;
                               0.4104730955300676 0.42054017056700005;
                               0.4323871489049083 0.4419958953923121;
                               0.6477059711749663 0.6575326613707073;
                               0.9536586385931917 0.9631448543384294;
                               1.0893489409351493 1.096842883323302]
                expected_vthe = [18.64261426434276 18.50354409844108;
                                 19.499487961574797 19.385754605644518;
                                 20.689993562190157 20.614281898809583;
                                 21.412974545098496 21.365357094116852;
                                 21.67384648365705 21.639847959789318;
                                 21.89736153390059 21.876037792806713;
                                 22.20937955406744 22.210214344849895;
                                 22.442351588454205 22.461966076355885;
                                 22.53790297532629 22.566129519188898;
                                 22.62998527423001 22.665848738566407;
                                 22.772409195528525 22.819983251667573;
                                 22.89112909904556 22.947654563130563;
                                 22.939252748338887 22.999271533340103;
                                 22.985397557257386 23.048645206031335;
                                 23.045645010740074 23.11276352989787;
                                 23.078628910305103 23.147904763694267;
                                 23.08070100045152 23.14998359506551;
                                 23.078628910305106 23.14790476369421;
                                 23.04564501074008 23.112763529897833;
                                 22.985397557257386 23.048645206031335;
                                 22.93925274833889 22.99927153334008;
                                 22.891129099045564 22.947654563130563;
                                 22.772409195528525 22.819983251667615;
                                 22.629985274230012 22.665848738566414;
                                 22.5379029753263 22.566129519188912;
                                 22.44235158845421 22.461966076355875;
                                 22.209379554067446 22.210214344849895;
                                 21.89736153390059 21.876037792806734;
                                 21.67384648365705 21.639847959789336;
                                 21.412974545098486 21.365357094116888;
                                 20.68999356219015 20.61428189880952;
                                 19.49948796157476 19.385754605644482;
                                 18.642614264342736 18.503544098441008]

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
