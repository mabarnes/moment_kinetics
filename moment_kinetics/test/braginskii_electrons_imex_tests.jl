module BraginskiiElectronsIMEX

# Regression test using wall boundary conditions, with recycling fraction less than 1 and
# a plasma source. Runs for a while and then checks phi profile against saved reference
# output.

include("setup.jl")

using Base.Filesystem: tempname
using MPI

using moment_kinetics.interpolation: interpolate_to_grid_z
using moment_kinetics.load_data: get_run_info_no_setup, close_run_info, get_variable
using moment_kinetics.utils: merge_dict_with_kwargs!

# default inputs for tests
test_input = OptionsDict("composition" => OptionsDict("n_ion_species" => 1,

                                                      "n_neutral_species" => 1,
                                                      "electron_physics" => "braginskii_fluid",
                                                      "T_e" => 0.2),
                         "output" => OptionsDict("run_name" => "braginskii-electrons-imex"),
                         "evolve_moments" => OptionsDict("density" => true,
                                                         "parallel_flow" => true,
                                                         "moments_conservation" => true,
                                                         "pressure" => true),
                         "ion_species_1" => OptionsDict("initial_density" => 1.0,
                                                        "initial_temperature" => 0.3333333333333333),
                         "z_IC_ion_species_1" => OptionsDict("initialization_option" => "sinusoid",
                                                             "density_amplitude" => 0.1,
                                                             "density_phase" => 0.0,
                                                             "upar_amplitude" => 1.4142135623730951,
                                                             "upar_phase" => 0.0,
                                                             "temperature_amplitude" => 0.1,
                                                             "temperature_phase" => 1.0),
                         "vpa_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                               "density_amplitude" => 1.0,
                                                               "density_phase" => 0.0,
                                                               "upar_amplitude" => 0.0,
                                                               "upar_phase" => 0.0,
                                                               "temperature_amplitude" => 0.0,
                                                               "temperature_phase" => 0.0),
                         "neutral_species_1" => OptionsDict("initial_density" => 1.0,
                                                            "initial_temperature" => 0.3333333333333333),
                         "z_IC_neutral_species_1" => OptionsDict("initialization_option" => "sinusoid",
                                                                 "density_amplitude" => 0.001,
                                                                 "density_phase" => 0.0,
                                                                 "upar_amplitude" => 0.0,
                                                                 "upar_phase" => 0.0,
                                                                 "temperature_amplitude" => 0.0,
                                                                 "temperature_phase" => 0.0),
                         "vz_IC_neutral_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                                  "density_amplitude" => 1.0,
                                                                  "density_phase" => 0.0,
                                                                  "upar_amplitude" => 0.0,
                                                                  "upar_phase" => 0.0,
                                                                  "temperature_amplitude" => 0.0,
                                                                  "temperature_phase" => 0.0),
                         "electron_fluid_collisions" => OptionsDict("nu_ei" => 1414.213562373095),
                         "reactions" => OptionsDict("charge_exchange_frequency" => 1.0606601717798214,
                                                    "electron_charge_exchange_frequency" => 0.0,
                                                    "ionization_frequency" => 0.7071067811865476,
                                                    "electron_ionization_frequency" => 0.7071067811865476),
                         "timestepping" => OptionsDict("type" => "KennedyCarpenterARK324",
                                                       "implicit_ion_advance" => false,
                                                       "implicit_vpa_advection" => false,
                                                       "nstep" => 10000,
                                                       "dt" => 7.071067811865475e-7,
                                                       "minimum_dt" => 7.071067811865474e-8,
                                                       "rtol" => 1.0e-7,
                                                       "nwrite" => 10000,
                                                       "exact_output_times" => true,
                                                       "high_precision_error_sum" => true),
                         "nonlinear_solver" => OptionsDict("nonlinear_max_iterations" => 100),
                         "r" => OptionsDict("ngrid" => 1,
                                            "nelement" => 1),
                         "z" => OptionsDict("ngrid" => 17,
                                            "nelement" => 16,
                                            "bc" => "periodic",
                                            "discretization" => "chebyshev_pseudospectral"),
                         "vpa" => OptionsDict("ngrid" => 6,
                                              "nelement" => 31,
                                              "L" => 20.784609690826528,
                                              "bc" => "zero",
                                              "discretization" => "chebyshev_pseudospectral"),
                         "vz" => OptionsDict("ngrid" => 6,
                                             "nelement" => 31,
                                             "L" => 20.784609690826528,
                                             "bc" => "zero",
                                             "discretization" => "chebyshev_pseudospectral"),
                         "ion_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0,
                                                                    "vpa_dissipation_coefficient" => 4.242640687119286),
                         "neutral_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0,
                                                                        "vz_dissipation_coefficient" => 0.42426406871192857),
                        )

if global_size[] > 2 && global_size[] % 2 == 0
    # Test using distributed-memory
    test_input["z"]["nelement_local"] = test_input["z"]["nelement"] ÷ 2
end

"""
Run a test for a single set of parameters
"""
function run_test(test_input, expected_p, expected_q, expected_vt; rtol=1.e-6,
                  atol=2.e-8, args...)
    # by passing keyword arguments to run_test, args becomes a Tuple of Pairs which can be
    # used to update the default inputs

    # Make a copy to make sure nothing modifies the input Dicts defined in this test
    # script.
    input = deepcopy(test_input)

    # Convert keyword arguments to a unique name
    name = input["output"]["run_name"]
    if length(args) > 0
        name = string(name, "_", (string(k, "-", v, "_") for (k, v) in args)...)

        # Remove trailing "_"
        name = chop(name)
    end

    # Provide some progress info
    println("    - testing ", name)

    # Update default inputs with values to be changed
    merge_dict_with_kwargs!(input; args...)
    input["output"]["run_name"] = name

    # Suppress console output while running
    p = undef
    q = undef
    vt = undef
    quietoutput() do
        # run simulation
        run_moment_kinetics(input)
    end

    if global_rank[] == 0
        quietoutput() do
            # Load and analyse output
            #########################

            path = joinpath(realpath(input["output"]["base_directory"]), name)

            # open the output file
            run_info = get_run_info_no_setup(path)

            pressure_zrt = get_variable(run_info, "electron_pressure")
            parallel_heat_flux_zrt = get_variable(run_info, "electron_parallel_heat_flux")
            thermal_speed_zrt = get_variable(run_info, "electron_thermal_speed")

            close_run_info(run_info)

            p = pressure_zrt[:,1,:]
            q = parallel_heat_flux_zrt[:,1,:]
            vt = thermal_speed_zrt[:,1,:]
        end

        # Regression test
        actual_p = p[begin:3:end, end]
        actual_q = q[begin:3:end, end]
        actual_vt = vt[begin:3:end, end]
        if expected_p == nothing
            # Error: no expected input provided
            println("data tested would be: ", actual_p)
            @test false
        else
            @test isapprox(actual_p, expected_p, rtol=rtol, atol=atol)
        end
        if expected_q == nothing
            # Error: no expected input provided
            println("data tested would be: ", actual_q)
            @test false
        else
            @test isapprox(actual_q, expected_q, rtol=10.0*rtol, atol=atol)
        end
        if expected_vt == nothing
            # Error: no expected input provided
            println("data tested would be: ", actual_vt)
            @test false
        else
            @test isapprox(actual_vt, expected_vt, rtol=rtol, atol=atol)
        end
    end
end

function runtests()
    # Create a temporary directory for test output
    test_output_directory = get_MPI_tempdir()

    expected_p = [0.29122673782671016, 0.2902544145496486, 0.28792429833302446,
                  0.28548221974848065, 0.2838423587124402, 0.28316644123602797,
                  0.2829635627029506, 0.28233662026989437, 0.2818560042572241,
                  0.2819217889356028, 0.2822584361023034, 0.2823828492151794,
                  0.2828237838888311, 0.28397557969355125, 0.2857696291348859,
                  0.2875099754915623, 0.2882391354029117, 0.28900858995870604,
                  0.2912491739815254, 0.2945273273219457, 0.2977819875283332,
                  0.2996490315618628, 0.30031945100062307, 0.30300340745207505,
                  0.30731984371186893, 0.3119320274905717, 0.3151372083732714,
                  0.31598457154082893, 0.31845505029114146, 0.32321561997462767,
                  0.3287723756476595, 0.333195276889545, 0.3348749259067474,
                  0.33656278232296116, 0.3410870907603372, 0.3469396258915971,
                  0.3521063539966746, 0.354843704560761, 0.35579141637172335,
                  0.35941655357074664, 0.3647423404209601, 0.3698353864170826,
                  0.37304830576158676, 0.37385549180123867, 0.3761113613588437,
                  0.3800615239497253, 0.3840329410048753, 0.38671074652463266,
                  0.3876156658769676, 0.3884625906129405, 0.39042095873979576,
                  0.39226377758152126, 0.3932129212253168, 0.3934437062808398,
                  0.39347774270493335, 0.3933807221660591, 0.3925377912337862,
                  0.3908511902915902, 0.38927236350452643, 0.3888048327495758,
                  0.3873306784748779, 0.38406235400918864, 0.3796234162140163,
                  0.3756757146639089, 0.37409062654990766, 0.3724535607922821,
                  0.36786192177493027, 0.36152685513865457, 0.3556079082234039,
                  0.35235975045899104, 0.35121798392177117, 0.34677138378213956,
                  0.34001542693063513, 0.33329973210206765, 0.3289212009444422,
                  0.32780170798435415, 0.32462636174807724, 0.31886821383401837,
                  0.3127213763666692, 0.3082503419394835, 0.30664695611770654,
                  0.3050870525964152, 0.30115455035644223, 0.2965914446293617,
                  0.29304186704043594, 0.2913409512125173]
    expected_q = [0.23557328126645913, 0.22888465724382456, 0.2104152485370436,
                  0.1854955486247594, 0.1626242630580381, 0.1501796820047745,
                  0.14581701620839432, 0.12885648365918403, 0.10305459167939787,
                  0.07714485691529575, 0.05997921200220598, 0.05554122804724605,
                  0.042823743208368, 0.019145633726924374, -0.00733649048993169,
                  -0.02769594334265656, -0.03528633234956319, -0.04284624373362639,
                  -0.06281285686690176, -0.08813366725484097, -0.11016246696311266,
                  -0.12176097209930227, -0.12576948409902047, -0.14109714685635047,
                  -0.16367834064497536, -0.18551170719230314, -0.19950515524758086,
                  -0.20305785661949904, -0.2130821883733232, -0.2310878444739801,
                  -0.25004448599102014, -0.26360429050754264, -0.2684019118792068,
                  -0.2730259110978752, -0.28442314926196416, -0.29692399905412153,
                  -0.3056730901746804, -0.30935117949166385, -0.31045833902516307,
                  -0.31384889120289405, -0.31611074088513236, -0.3146380284143208,
                  -0.3114453614988543, -0.3103187591560904, -0.3063773390510246,
                  -0.2961162640084848, -0.27987679023109435, -0.26379138301601057,
                  -0.25696564231577873, -0.24971497587514693, -0.22836383669827912,
                  -0.19666306052765978, -0.16490049884216484, -0.14664414685697508,
                  -0.140092303662593, -0.11393199734536093, -0.07228740013644226,
                  -0.02875545355360291, 0.0006967667638368611, 0.008354228267100835,
                  0.03034564155983787, 0.07115961053565374, 0.11585501368510419,
                  0.14889488709303933, 0.16081069847053694, 0.17241994153246842,
                  0.20166689711642918, 0.23525619516209126, 0.2607053382099483,
                  0.27251558967040057, 0.2763264711161255, 0.28955137050446594,
                  0.304932265919255, 0.3148256144472792, 0.3184290527960464,
                  0.3189877406768403, 0.3197614763698102, 0.31802907835300215,
                  0.3114557011059248, 0.30330226332746846, 0.29961565923753236,
                  0.29560989965635254, 0.2835002128518717, 0.26518826583003713,
                  0.24682603235335937, 0.2363263173065147]
    expected_vt = [48.69930069470063, 48.66062455243542, 48.56248385320539,
                   48.44739540039693, 48.356899577993346, 48.31320986052848,
                   48.29877856909803, 48.246923892235586, 48.18066351580376,
                   48.129080487159236, 48.10305953666385, 48.09738838964179,
                   48.0835369263978, 48.06725256843622, 48.06377329183692,
                   48.07175665423122, 48.077120078348294, 48.0837555662707,
                   48.10752410001497, 48.15080133501752, 48.20057520767725,
                   48.2313940231353, 48.24279792962095, 48.290008958569004,
                   48.37031551281947, 48.46084104743039, 48.52606969578476,
                   48.543592477045266, 48.59528711815222, 48.69721705884503,
                   48.819573417539814, 48.91926424085318, 48.957619266840894,
                   48.99642723715217, 49.1017339636088, 49.240685999482,
                   49.36596387753322, 49.433382644490976, 49.45690221208473,
                   49.54776603236271, 49.684120202021894, 49.818354086686135,
                   49.905457551134475, 49.92769380108198, 49.99070984289968,
                   50.104826201294834, 50.22641719846013, 50.31456550613657,
                   50.3460642993979, 50.37662372632785, 50.45311406211657,
                   50.540158070014336, 50.60542656331001, 50.63538420940169,
                   50.644978423640666, 50.6778277931332, 50.71420910246362,
                   50.73400412818807, 50.73765907879664, 50.73735335711106,
                   50.73359017832743, 50.714984397497645, 50.675871356244734,
                   50.632384330688886, 50.613148582127394, 50.59235412049779,
                   50.52954943236978, 50.43347149716516, 50.33502859461642,
                   50.27769938918693, 50.257007364620655, 50.17378187627705,
                   50.03933232531328, 49.89597686850462, 49.79709515630362,
                   49.77109953316968, 49.695726282509895, 49.552541340875706,
                   49.389530772579924, 49.26346680198405, 49.21652750723896,
                   49.16990329194413, 49.04773733444067, 48.89613325612863,
                   48.768712595039574, 48.70377032426922]

    @testset "Braginskii electron IMEX timestepping" verbose=use_verbose begin
        println("Braginskii electron IMEX timestepping tests")

        @testset "Split 3" begin
            test_input["output"]["base_directory"] = test_output_directory
            run_test(test_input, expected_p, expected_q, expected_vt)
        end
        @long @testset "Check other timestep - $type" for
                type ∈ ("KennedyCarpenterARK437",)

            timestep_check_input = deepcopy(test_input)
            timestep_check_input["output"]["base_directory"] = test_output_directory
            timestep_check_input["output"]["run_name"] = type
            timestep_check_input["timestepping"]["type"] = type
            run_test(timestep_check_input, expected_p, expected_q, expected_vt,
                     rtol=2.e-4)
        end
    end

    if global_rank[] == 0
        # Delete output directory to avoid using too much disk space
        rm(realpath(test_output_directory); recursive=true)
    end
end

end # BraginskiiElectronsIMEX


using .BraginskiiElectronsIMEX

BraginskiiElectronsIMEX.runtests()
