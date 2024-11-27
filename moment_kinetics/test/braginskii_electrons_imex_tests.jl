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

    expected_p = [0.4495141154262702, 0.44810731623424865, 0.4446924860674211,
                  0.44101474518047623, 0.43843729172365287, 0.43732257545500436,
                  0.4369771896163651, 0.43585040736153463, 0.43476981731494,
                  0.43442274081030574, 0.43458934725870224, 0.43468321808979,
                  0.4350672174585142, 0.4362345707586601, 0.4382356226789203,
                  0.44027341593371017, 0.44114469018060415, 0.4420726748472857,
                  0.4448144055192906, 0.44890293982902774, 0.45302566125041793,
                  0.455412309803846, 0.45627258226390244, 0.45973185720339665,
                  0.46533867318637956, 0.47137720587404724, 0.4755972251834676,
                  0.4767157199899437, 0.479982869902296, 0.48630210267423224,
                  0.4937119745209616, 0.499632228117403, 0.5018851956608419,
                  0.5041516147429022, 0.5102382404520471, 0.5181350935202635,
                  0.5251274962417639, 0.528840009619943, 0.5301266410196889,
                  0.5350545811435541, 0.5423140663994042, 0.5492817851159084,
                  0.5536930382468159, 0.5548035497737663, 0.5579127105543162,
                  0.5633811248630228, 0.5689228569091898, 0.5726991310447442,
                  0.5739863254025046, 0.5751980350089889, 0.5780379005475736,
                  0.5808090106835145, 0.5823710691200007, 0.5828440926355674,
                  0.5829481350770388, 0.5830503832825662, 0.5822895468763042,
                  0.5804173591751022, 0.5785672859272517, 0.5780103965926895,
                  0.5762363699105814, 0.572237324021336, 0.5667183507722322,
                  0.5617561268531523, 0.559752846313555, 0.5576782694282233,
                  0.5518326820337154, 0.543711872849146, 0.5360739378273777,
                  0.5318633271279509, 0.530380132078173, 0.5245886109720006,
                  0.5157429370998954, 0.5068932335599383, 0.5010911857600961,
                  0.49960342886504683, 0.4953735336353052, 0.4876627291240921,
                  0.4793665506564848, 0.4732825693615421, 0.4710889716397071,
                  0.46894821577973017, 0.4635186380600756, 0.45714663551681506,
                  0.4521183598666343, 0.4496787838015138]
    expected_q = [0.6827418582524897, 0.6759389228503324, 0.6546242399054403,
                  0.6207644558519532, 0.5852347231333254, 0.564287918118347,
                  0.556689983477969, 0.5259471969259888, 0.4756862865761334,
                  0.4212550001972507, 0.3831394924057264, 0.37303276222778314,
                  0.343503487668719, 0.28638988969921425, 0.2194175020090074,
                  0.1658803337183493, 0.14548865444145112, 0.1249612385669573,
                  0.06972848884953677, -0.0022814642388705414, -0.06656443212989245,
                  -0.10097001502687641, -0.11294895521598257, -0.15913617106073358,
                  -0.22827644490625998, -0.2963224914648622, -0.3405683906933329,
                  -0.3518831487295314, -0.3840082790031256, -0.44248719673145465,
                  -0.5054051728451866, -0.551574215204323, -0.5682200554009741,
                  -0.5844574843383253, -0.6255436186448009, -0.6733338973995404,
                  -0.7102497022619719, -0.7276721907765155, -0.7333397406079434,
                  -0.7531999175389356, -0.7766606922910467, -0.7916849353418282,
                  -0.7966732798546298, -0.7972935019748215, -0.7974998065347064,
                  -0.7914511420487961, -0.7742462932837018, -0.7531170349745917,
                  -0.7434154933841022, -0.732749549391789, -0.6996770392831854,
                  -0.6472925144162879, -0.5920338583314992, -0.5592837663247994,
                  -0.5473759749629427, -0.49905814442640656, -0.419796532987056,
                  -0.3339940482100009, -0.27421323483476967, -0.2584305361791927,
                  -0.21253786937371122, -0.12493437607128674, -0.024809724391701034,
                  0.05267445297594558, 0.08149394898045487, 0.11008483580643007,
                  0.18477109572661538, 0.27671771664185124, 0.353033924201842,
                  0.3914632816914362, 0.4044296067758364, 0.4523545483111222,
                  0.5176797539993653, 0.5739945138569089, 0.6061138463121605,
                  0.6137350794737616, 0.6340122577148257, 0.6655474631299673,
                  0.6911272362100365, 0.7037896304781194, 0.7069493638375064,
                  0.7092542139170925, 0.7113077755769187, 0.7055330024527745,
                  0.692930113700812, 0.6834744521342498]
    expected_vt = [60.503243776596165, 60.46153023170041, 60.35196357745784,
                   60.21534169442452, 60.099829334123406, 60.040677146084676,
                   60.02054030308586, 59.945192607070005, 59.83956332270126,
                   59.74466073286034, 59.6881815170369, 59.67443939896856,
                   59.63705527455284, 59.5755966280989, 59.52003881563317,
                   59.487303746230786, 59.47740469832934, 59.46882266803735,
                   59.45242085810586, 59.44514055782754, 59.45162966582038,
                   59.460023392631314, 59.4637436896616, 59.48193396567828,
                   59.520607349859056, 59.57229303072903, 59.6134051525281,
                   59.62490374169246, 59.65981194145, 59.73238824637004,
                   59.82488582955655, 59.90379982182214, 59.93490096692693,
                   59.96675491891175, 60.05500747114224, 60.17514126096532,
                   60.28669920214769, 60.34792696001425, 60.36947746684328,
                   60.453647521646644, 60.582668964615706, 60.71293949062846,
                   60.79930544278526, 60.82159978880084, 60.88535831265987,
                   61.003191250737984, 61.13271036566074, 61.22982940090206,
                   61.26534648299963, 61.300283935289386, 61.390210936392684,
                   61.49842988504546, 61.58617772618817, 61.6295965023691,
                   61.64411705987245, 61.697148185872855, 61.767402554884846,
                   61.8251039647873, 61.85602560032788, 61.863044731750065,
                   61.880896838249456, 61.90494747307992, 61.91707206994147,
                   61.91528656108256, 61.91206502126355, 61.90744824569871,
                   61.88836065205279, 61.84936481884471, 61.80151925202637,
                   61.77096817847411, 61.759530818105134, 61.711578242982426,
                   61.628367571043206, 61.532972492870755, 61.46358631318027,
                   61.444884310802, 61.38960250589494, 61.280335702777364,
                   61.149196249162785, 61.04274069753091, 61.0019364120191,
                   60.960757611821634, 60.849658312940456, 60.704804495390036,
                   60.57626380073712, 60.50801542345454]

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
