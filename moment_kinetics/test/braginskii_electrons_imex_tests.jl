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
test_input = OptionsDict( "composition" => OptionsDict("n_ion_species" => 1,
                                                  "n_neutral_species" => 1,
                                                  "electron_physics" => "braginskii_fluid",
                                                  "T_e" => 0.2),
                  "output" => OptionsDict("run_name" => "braginskii-electrons-imex"),
                  "evolve_moments" => OptionsDict("density" => true,
                                                  "parallel_flow" => true,
                                                  "parallel_pressure" => true,
                                                  "moments_conservation" => true),
                  "ion_species_1" => OptionsDict("initial_density" => 1.0,
                                                      "initial_temperature" => 1.0),
                  "z_IC_ion_species_1" => OptionsDict("initialization_option" => "sinusoid",
                                                           "density_amplitude" => 0.1,
                                                           "density_phase" => 0.0,
                                                           "upar_amplitude" => 1.0,
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
                                                          "initial_temperature" => 1.0),
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
                  "electron_fluid_collisions" => OptionsDict("nu_ei" => 1.0e3),
                  "reactions" => OptionsDict("charge_exchange_frequency" => 0.75,
                                             "electron_charge_exchange_frequency" => 0.0,
                                             "ionization_frequency" => 0.5,
                                             "electron_ionization_frequency" => 0.5),
                  "timestepping" => OptionsDict("type" => "KennedyCarpenterARK324",
                                                     "implicit_ion_advance" => false,
                                                     "implicit_vpa_advection" => false,
                                                     "nstep" => 10000,
                                                     "dt" => 1.0e-6,
                                                     "minimum_dt" => 1.e-7,
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
                                       "L" => 12.0,
                                       "bc" => "zero",
                                       "discretization" => "chebyshev_pseudospectral"),
                  "vz" => OptionsDict("ngrid" => 6,
                                      "nelement" => 31,
                                      "L" => 12.0,
                                      "bc" => "zero",
                                      "discretization" => "chebyshev_pseudospectral"),
                  "ion_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0,
                                                                  "vpa_dissipation_coefficient" => 1e0),
                  "neutral_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0,
                                                                      "vz_dissipation_coefficient" => 1e-1))

if global_size[] > 2 && global_size[] % 2 == 0
    # Test using distributed-memory
    test_input["z"]["nelement_local"] = test_input["z"]["nelement"] ÷ 2
end

"""
Run a test for a single set of parameters
"""
function run_test(test_input, expected_p, expected_q, expected_vt; rtol=1.e-6,
                  atol=1.e-8, args...)
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

            parallel_pressure_zrt = get_variable(run_info, "electron_parallel_pressure")
            parallel_heat_flux_zrt = get_variable(run_info, "electron_parallel_heat_flux")
            thermal_speed_zrt = get_variable(run_info, "electron_thermal_speed")

            close_run_info(run_info)

            p = parallel_pressure_zrt[:,1,:]
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

    expected_p = [0.4495142465429501, 0.4481074570400799, 0.4446926507671661,
                  0.441014931207275, 0.438437484907815, 0.43732276848674934,
                  0.43697738108348716, 0.4358505937202197, 0.4347699826447534,
                  0.4344228757609946, 0.43458945364148766, 0.43468331831747653,
                  0.4350672958077072, 0.43623460384739954, 0.4382356037697932,
                  0.4402733574139258, 0.44114461741796185, 0.44207258633469737,
                  0.44481428500607545, 0.4489027843262288, 0.453025485972639,
                  0.4554121289571335, 0.45627240079596043, 0.45973167401761067,
                  0.46533850385552156, 0.4713770638795399, 0.47559710838136604,
                  0.476715611261031, 0.479982783224088, 0.4863020639892704,
                  0.4937119975974234, 0.49963229872930703, 0.5018852836409862,
                  0.5041517184636275, 0.5102383830494562, 0.5181352744341807,
                  0.5251276978157543, 0.5288402134929471, 0.5301268445658873,
                  0.5350547772015388, 0.5423142325356332, 0.549281901092844,
                  0.5536931125467701, 0.5548036118396735, 0.5579127346493844,
                  0.5633810748426228, 0.5689227223013396, 0.5726989389591978,
                  0.5739861129301271, 0.5751978044349347, 0.5780376317680165,
                  0.5808087184921097, 0.5823707816448083, 0.5828438191685916,
                  0.5829478678498194, 0.5830501447271408, 0.582289374665072,
                  0.58041726529817, 0.578567246341523, 0.578010370567434,
                  0.5762363836977468, 0.5722374018277052, 0.5667184837394905,
                  0.5617562820044443, 0.5597530081006242, 0.5576784345338779,
                  0.5518328397278729, 0.5437120062373345, 0.5360740333848819,
                  0.5318634014437891, 0.5303801974694475, 0.5245886448318062,
                  0.5157429268125628, 0.5068931895798254, 0.5010911275747203,
                  0.4996033672347492, 0.49537346939399834, 0.48766266395170715,
                  0.4793665038362437, 0.47328254478889087, 0.47108895742014023,
                  0.46894821314506374, 0.46351866701062155, 0.4571467077365343,
                  0.4521184688553548, 0.4496789134802193]
    expected_q = [0.6827368245785391, 0.6759374884122982, 0.6546231347742282,
                  0.6207644653684102, 0.5852345446664419, 0.5642914912701764,
                  0.5566913793739807, 0.5259485115758797, 0.4756895656827874,
                  0.42125917218523906, 0.38314112093591407, 0.3730358831782404,
                  0.3435094715658696, 0.28639442460467884, 0.21942310954090613,
                  0.16588345312219946, 0.14549470709849813, 0.12496588212159782,
                  0.06973409804318272, -0.002277979990819454, -0.06656125842474958,
                  -0.10096979713184408, -0.11294886299698863, -0.15913505586466467,
                  -0.22827559683009083, -0.29632348386800467, -0.34057238788817457,
                  -0.35188477117080214, -0.38401198256646085, -0.4424894117515319,
                  -0.5054075147722747, -0.551578280611729, -0.5682229018769172,
                  -0.5844589615289878, -0.6255457794535301, -0.6733324179176754,
                  -0.7102476286038639, -0.727670964209917, -0.7333371144326686,
                  -0.75319805658154, -0.7766559507703321, -0.7916773920489434,
                  -0.7966681631677981, -0.7972852592194414, -0.7974916160291323,
                  -0.7914438097067681, -0.7742385679874255, -0.7531088527477023,
                  -0.743408463634901, -0.7327425055315714, -0.6996714269670532,
                  -0.6472876583694323, -0.5920318841772106, -0.5592838355889932,
                  -0.5473758460409419, -0.49906014931826975, -0.41979928692481905,
                  -0.33399875775641275, -0.2742157838833754, -0.2584365112978667,
                  -0.2125423348311034, -0.12493880095585069, -0.024812382514926727,
                  0.05267036484341804, 0.08148808313922458, 0.11008153462508363,
                  0.18476737546020716, 0.2767152191735648, 0.35303031409372465,
                  0.39146322002938433, 0.4044299132160631, 0.45235213418321946,
                  0.5176780467364177, 0.5739929333551202, 0.6061126338223533,
                  0.6137322801586892, 0.6340064254152137, 0.6655411066848723,
                  0.6911220612965332, 0.703784794184395, 0.7069433100521626,
                  0.7092485734332703, 0.7113024723236737, 0.7055269402944423,
                  0.6929268653405701, 0.6834684333208771]
    expected_vt = [60.50325000883391, 60.46153663577712, 60.351970477892515,
                   60.21534903090227, 60.09983676011894, 60.04068455588239,
                   60.02054763915641, 59.945199883240946, 59.83957009420658,
                   59.744666895939794, 59.68818695137363, 59.674444759340595,
                   59.637060182503255, 59.57560056194215, 59.52004170741952,
                   59.48730585934008, 59.477406533256826, 59.46882409647505,
                   59.452421734214404, 59.4451406974879, 59.451629381397865,
                   59.460022972831844, 59.46374326848477, 59.481933384371196,
                   59.52060694116945, 59.57229292756352, 59.61340531331237,
                   59.624904024931666, 59.65981241052454, 59.73238919549682,
                   59.82488748334155, 59.903801944653935, 59.93490325573575,
                   59.96675729988948, 60.055010088419515, 60.17514400987675,
                   60.286701981156945, 60.34792956674607, 60.36948000195064,
                   60.453649700899376, 60.5826703964192, 60.71293998487627,
                   60.799305288874706, 60.821599434956894, 60.885357316117144,
                   61.003189111620806, 61.13270692111263, 61.229825132684674,
                   61.2653418670328, 61.300279019223346, 61.390205327453316,
                   61.49842367304756, 61.586171197195874, 61.629589978240276,
                   61.644110540583746, 61.69714162299434, 61.76739640893565,
                   61.82509834367504, 61.85602039180023, 61.86303962006188,
                   61.88089212318481, 61.90494338120651, 61.91706876619464,
                   61.91528365713896, 61.912062394252956, 61.90744583447494,
                   61.888358463758614, 61.849363124441425, 61.80151776974985,
                   61.77096690371078, 61.75952954046432, 61.71157712980619,
                   61.62836673446321, 61.53297203742497, 61.46358620440665,
                   61.444884246595834, 61.38960287198621, 61.28033655867083,
                   61.14919803185266, 61.04274318967074, 61.00193919888,
                   60.96076073093901, 60.84966221188132, 60.704809373196035,
                   60.57626945515781, 60.50802161608869]

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
                     rtol=2.e-4, atol=1.e-10)
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
