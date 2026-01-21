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
                                            "kinetic_electron_preconditioner" => "lu",
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

kinetic_input_time_evolving = deepcopy(kinetic_input_adaptive_timestep)
kinetic_input_time_evolving["output"]["run_name"] = "kinetic_electron_implicit_time_evolving_test"
kinetic_input_time_evolving["timestepping"]["kinetic_electron_solver"] = "implicit_time_evolving"

# The following settings don't run for long enough to give a very good test, but at least
# make sure that the explicit electron solver can take a single timestep.
kinetic_input_implicit_ppar_explicit_pseudotimestep = deepcopy(kinetic_input)
kinetic_input_implicit_ppar_explicit_pseudotimestep["output"]["run_name"] = "kinetic_electron_implicit_ppar_explicit_pseudotimestep_test"
kinetic_input_implicit_ppar_explicit_pseudotimestep["timestepping"]["nstep"] = 1
kinetic_input_implicit_ppar_explicit_pseudotimestep["timestepping"]["kinetic_electron_solver"] = "implicit_p_explicit_pseudotimestep"
kinetic_input_implicit_ppar_explicit_pseudotimestep["electron_timestepping"]["nstep"] = 5000000
kinetic_input_implicit_ppar_explicit_pseudotimestep["electron_timestepping"]["dt"] = 7.071067811865474e-8
kinetic_input_implicit_ppar_explicit_pseudotimestep["electron_timestepping"]["maximum_dt"] = 7.0710678118654756e-6
kinetic_input_implicit_ppar_explicit_pseudotimestep["electron_timestepping"]["nwrite"] = 10000
kinetic_input_implicit_ppar_explicit_pseudotimestep["electron_timestepping"]["nwrite_dfns"] = 100000
kinetic_input_implicit_ppar_explicit_pseudotimestep["electron_timestepping"]["type"] = "Fekete4(3)"
kinetic_input_implicit_ppar_explicit_pseudotimestep["electron_timestepping"]["rtol"] = 0.001
kinetic_input_implicit_ppar_explicit_pseudotimestep["electron_timestepping"]["atol"] = 1.0e-14
kinetic_input_implicit_ppar_explicit_pseudotimestep["electron_timestepping"]["minimum_dt"] = 7.071067811865476e-10

kinetic_input_explicit_time_evolving = deepcopy(kinetic_input_implicit_ppar_explicit_pseudotimestep)
kinetic_input_explicit_time_evolving["output"]["run_name"] = "kinetic_electron_explicit_time_evolving_test"
kinetic_input_explicit_time_evolving["timestepping"]["kinetic_electron_solver"] = "explicit_time_evolving"
kinetic_input_explicit_time_evolving["timestepping"]["type"] = "Fekete4(3)"
kinetic_input_explicit_time_evolving["timestepping"]["maximum_dt"] = 7.0710678118654756e-6

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
        # Using shared-memory parallelism, test both LU and ADI preconditioners
        adi_precon_iterations_values = (-1,1,2)
    else
        adi_precon_iterations_values = -1
    end

    # Test implicit electron solve
    test_inputs = [(deepcopy(kinetic_input), "fixed timestep", 1.0e-6),
                   (deepcopy(kinetic_input_time_evolving), "time evolving", 4.0e-4),]
    @long push!(test_inputs, (deepcopy(kinetic_input_adaptive_timestep), "adaptive timestep", 2.0e-4))
    @testset "$label$(adi_precon_iterations < 0 ? "" : " $adi_precon_iterations")" for (this_kinetic_input, label, tol) ∈ test_inputs,
                                                                                       adi_precon_iterations ∈ adi_precon_iterations_values

        this_kinetic_input["output"]["base_directory"] = test_output_directory

        if adi_precon_iterations < 0
            # Provide some progress info
            println("    - testing kinetic electrons, $label")
            # Don't use distributed memory parallelism with LU preconditioner
            pop!(this_kinetic_input["z"], "nelement_local")
        else
            this_kinetic_input["output"]["run_name"] *= "_adi$adi_precon_iterations"
            this_kinetic_input["timestepping"]["kinetic_electron_preconditioner"] = "adi"
            this_kinetic_input["nonlinear_solver"]["adi_precon_iterations"] = adi_precon_iterations

            # Provide some progress info
            println("    - testing kinetic electrons $adi_precon_iterations ADI iterations, $label")
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
            expected_Ez = [-1.1788590436647437 -1.1840066422007116;
                           -1.0344358145444665 -1.0422399694447615;
                           -0.6771873517748579 -0.6862495190878175;
                           -0.4429803756115142 -0.4522188959227873;
                           -0.4462673194829653 -0.45586997091929643;
                           -0.3579963781264046 -0.36747603353595837;
                           -0.2995802841275946 -0.30857329293600727;
                           -0.27684540547619774 -0.2848796448622787;
                           -0.2571946982529915 -0.2648520496937654;
                           -0.25588638307279576 -0.2627031003513376;
                           -0.23582274277344592 -0.24146603103657005;
                           -0.2044204542827916 -0.20883834271463425;
                           -0.18106623875887787 -0.18496799235997463;
                           -0.1578697236606448 -0.16101148346266028;
                           -0.10137732814548026 -0.10331418573596948;
                           -0.0359017194937635 -0.036574790244110936;
                           4.7702215793354345e-15 -3.5776944945312564e-15;
                           0.03590171949376696 0.036574790244111685;
                           0.10137732814546131 0.10331418573596737;
                           0.15786972366065144 0.1610114834626606;
                           0.18106623875891134 0.18496799236005154;
                           0.2044204542827868 0.20883834271461643;
                           0.23582274277344414 0.24146603103655473;
                           0.25588638307280565 0.262703100351364;
                           0.2571946982529717 0.2648520496937277;
                           0.2768454054761966 0.28487964486225836;
                           0.2995802841276055 0.308573292936013;
                           0.35799637812640017 0.36747603353597746;
                           0.4462673194829772 0.455869970919262;
                           0.4429803756114905 0.45221889592278036;
                           0.6771873517748456 0.6862495190878125;
                           1.0344358145444779 1.0422399694447657;
                           1.1788590436646862 1.184006642200692]
            expected_vthe = [18.573794132263707 18.4499488082634;
                             19.444841126659096 19.34350795987584;
                             20.668608300455663 20.60189475165951;
                             21.421019727364992 21.380616077068147;
                             21.69613903490998 21.668672578028303;
                             21.93326133463881 21.917872274534997;
                             22.269571037205782 22.275492071545948;
                             22.52405980326533 22.548260720128788;
                             22.62947360515329 22.662172885068017;
                             22.73092623667227 22.771180221180742;
                             22.887805541258086 22.9397693691662;
                             23.018458810036844 23.07936318715505;
                             23.071237493188057 23.135659947649604;
                             23.122059826631247 23.189673603667828;
                             23.18790510377407 23.259377796874347;
                             23.22411672753825 23.29771125142667;
                             23.226131506653278 23.299758497838557;
                             23.22411672753825 23.29771125142667;
                             23.18790510377407 23.259377796874347;
                             23.122059826631244 23.18967360366779;
                             23.071237493188057 23.13565994764964;
                             23.018458810036844 23.07936318715499;
                             22.887805541258082 22.939769369166182;
                             22.73092623667226 22.771180221180696;
                             22.629473605153283 22.66217288506802;
                             22.52405980326532 22.548260720128788;
                             22.269571037205775 22.275492071545955;
                             21.933261334638807 21.917872274534975;
                             21.696139034909987 21.6686725780283;
                             21.421019727364985 21.380616077068147;
                             20.668608300455666 20.60189475165953;
                             19.444841126659085 19.34350795987586;
                             18.57379413226372 18.44994880826337]

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

    # Test explicit electron solves - separate loop for different expected results due to
    # a shorter simulation time, and no test for iteration counts.
    @long @testset "$label" for (this_kinetic_input, label, tol) ∈ ((deepcopy(kinetic_input_implicit_ppar_explicit_pseudotimestep), "explicit solve", 1.0e-4),
                                                                    (deepcopy(kinetic_input_explicit_time_evolving), "explicit time evolving", 4.0e-4),)
        this_kinetic_input["output"]["base_directory"] = test_output_directory

        # Provide some progress info
        println("    - testing kinetic electrons, $label")

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

            close_run_info(run_info)

            # Regression test
            # Benchmark data generated in serial on Linux
            expected_Ez = [-1.1358841010652496 -1.1364689644707748;
                           -1.037945220983319 -1.0381797177889809;
                           -0.6997319672524219 -0.699741507729716;
                           -0.45925852108834664 -0.45939173860523835;
                           -0.4721196116902184 -0.47222847527657624;
                           -0.380341922666808 -0.38038543639232775;
                           -0.320088560816652 -0.32017869246792585;
                           -0.29518043582568293 -0.29524998381868245;
                           -0.27420798548746056 -0.2742859921152289;
                           -0.2715054655984765 -0.2715710557296557;
                           -0.24898834528359867 -0.24904303323270555;
                           -0.21485514226110913 -0.2148978955562049;
                           -0.19014991998106778 -0.1901879082478011;
                           -0.16547068209218954 -0.16550073032441462;
                           -0.10612643332828586 -0.1061448054725634;
                           -0.03753767247134856 -0.03754404037878592;
                           3.100644026568032e-14 2.653435960243343e-14;
                           0.03753767247134918 0.03754404037878893;
                           0.10612643332825304 0.10614480547253548;
                           0.16547068209219631 0.16550073032441873;
                           0.19014991998111486 0.1901879082478401;
                           0.21485514226111177 0.21489789555620745;
                           0.24898834528359332 0.24904303323270113;
                           0.27150546559849786 0.27157105572967344;
                           0.27420798548745295 0.27428599211522525;
                           0.29518043582567133 0.29524998381867007;
                           0.32008856081665704 0.3201786924679299;
                           0.38034192266679934 0.3803854363923201;
                           0.4721196116902259 0.47222847527658296;
                           0.4592585210883342 0.45939173860522425;
                           0.6997319672524094 0.6997415077297056;
                           1.0379452209833306 1.0381797177889915;
                           1.1358841010651903 1.136468964470705]
            expected_vthe = [18.3344978466833 18.332976418538053;
                             19.21046207878694 19.209628177638518;
                             20.509904014827004 20.50938334888601;
                             21.319068841823384 21.318708046757877;
                             21.619852228865756 21.619751434761277;
                             21.889778936625653 21.88967718880338;
                             22.274006843881185 22.274084850703346;
                             22.572274605751605 22.572520861430363;
                             22.694327103242752 22.694651846277694;
                             22.81567771226113 22.816074698058593;
                             22.998600346691283 22.999109078302453;
                             23.151894728718528 23.152489330219048;
                             23.210910193305846 23.211537902369802;
                             23.271624439784215 23.272283053757977;
                             23.346057654310396 23.346751789790467;
                             23.38873277664779 23.38944697535822;
                             23.388883003204164 23.38959669595534;
                             23.388732776647768 23.389446975358197;
                             23.3460576543104 23.346751789790474;
                             23.271624439784222 23.272283053757977;
                             23.21091019330587 23.211537902369827;
                             23.15189472871852 23.152489330219037;
                             22.99860034669128 22.999109078302446;
                             22.815677712261117 22.816074698058575;
                             22.69432710324273 22.694651846277676;
                             22.572274605751574 22.57252086143033;
                             22.274006843881185 22.274084850703346;
                             21.88977893662566 21.889677188803386;
                             21.619852228865792 21.619751434761312;
                             21.319068841823384 21.318708046757877;
                             20.50990401482701 20.509383348886022;
                             19.21046207878693 19.2096281776385;
                             18.334497846683316 18.332976418538074]

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
        end
    end

    if global_rank[] == 0
        # Delete output directory to avoid using too much disk space
        rm(realpath(test_output_directory); recursive=true)
    end
end

function runtests()
    @testset "kinetic electrons" verbose=use_verbose begin
        println("Kinetic electron tests")
        run_test()
    end
    return nothing
end

end # KineticElectronsTests


using .KineticElectronsTests

KineticElectronsTests.runtests()
