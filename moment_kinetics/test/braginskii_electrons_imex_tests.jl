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
                         # Use higher-order "KennedyCarpenterARK437" scheme rather than
                         # "KennedyCarpenterARK324" for the default case in this test,
                         # because the higher order scheme is faster when enforcing the
                         # tight tolerances that we use here. Tight solver tolerances are
                         # used so that the results can be tested with reasonably tight
                         # tolerances without needing to match iteration counts (which
                         # could be affected by rounding errors, etc.).
                         "timestepping" => OptionsDict("type" => "KennedyCarpenterARK437",
                                                       "kinetic_ion_solver" => "full_explicit_ion_advance",
                                                       "nstep" => 10000,
                                                       "dt" => 7.071067811865475e-7,
                                                       "minimum_dt" => 7.071067811865474e-8,
                                                       "rtol" => 1.0e-11,
                                                       "atol" => 1.0e-14,
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
                  qpar_atol=2.e-6, args...)
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
            @test elementwise_isapprox(actual_p, expected_p, rtol=rtol, atol=0.0)
        end
        if expected_q == nothing
            # Error: no expected input provided
            println("data tested would be: ", actual_q)
            @test false
        else
            @test elementwise_isapprox(actual_q, expected_q, rtol=0.0, atol=qpar_atol)
        end
        if expected_vt == nothing
            # Error: no expected input provided
            println("data tested would be: ", actual_vt)
            @test false
        else
            @test elementwise_isapprox(actual_vt, expected_vt, rtol=rtol, atol=0.0)
        end
    end
end

function runtests()
    # Create a temporary directory for test output
    test_output_directory = get_MPI_tempdir()

    expected_p = [0.2912266392615819, 0.29025433984310034, 0.2879242795894934,
                  0.28548225806473115, 0.28384242576375174, 0.2831665146026283,
                  0.28296363638030586, 0.2823366897521711, 0.2818560435545019,
                  0.28192177350350867, 0.2822583749092662, 0.28238277460158656,
                  0.2828236654979084, 0.28397537848027077, 0.28576932659403154,
                  0.2875095946566044, 0.28823872862422006, 0.28900815861323625,
                  0.2912486795813647, 0.29452677666810323, 0.2977814147553455,
                  0.2996484553672727, 0.30031887657350365, 0.30300285160128704,
                  0.3073193430797012, 0.31193161657793445, 0.31513687813211466,
                  0.31598426295913096, 0.31845480898675826, 0.32321552054602143,
                  0.32877244876597045, 0.33319548212915606, 0.33487518066968125,
                  0.33656308386018763, 0.34108750584114733, 0.34694015823920066,
                  0.3521069551605689, 0.354844322667331, 0.3557920384394313,
                  0.35941717616187985, 0.36474291829688, 0.36983586978198124,
                  0.37304870523370237, 0.37385586661162806, 0.37611166127481654,
                  0.3800616750498591, 0.38403292172193515, 0.3867106101171749,
                  0.3876154906109739, 0.3884623805519838, 0.39042067991590823,
                  0.3922634642590541, 0.3932126368037694, 0.393443458681066,
                  0.39347751170947465, 0.39338056919954756, 0.39253779331488525,
                  0.39085137153218863, 0.38927266454143705, 0.38880516389341757,
                  0.3873310901398761, 0.38406289700323226, 0.3796240522962586,
                  0.3756763737398465, 0.37409128322752316, 0.37245421008517576,
                  0.36786251955039573, 0.3615273349978123, 0.3556082477818965,
                  0.35236000523009325, 0.3512182086376803, 0.34677149340933866,
                  0.34001536757416884, 0.3332995292327041, 0.3289209220377829,
                  0.3278014123659849, 0.3246260247818751, 0.31886782836137484,
                  0.3127209794825403, 0.30824996369248003, 0.3066465903792448,
                  0.30508670304077556, 0.30115425144651664, 0.29659122715149966,
                  0.29304172559674013, 0.29134085003866206]
    expected_q = [0.23556826923235596, 0.2288797766664259, 0.21041116238456214,
                  0.18549245084781352, 0.1626219201422328, 0.15017868685441035,
                  0.1458162456876596, 0.12885601503784916, 0.10305577133873899,
                  0.07714819243582582, 0.05998214248641777, 0.055544990502692515,
                  0.04282846870749182, 0.01915058669706081, -0.007332921607335636,
                  -0.02769202946560893, -0.035283415721614204, -0.04284327260070175,
                  -0.06281091997672486, -0.08813260536031556, -0.11016357594575653,
                  -0.12176308441555106, -0.1257726062289506, -0.1411008577426437,
                  -0.1636855820770096, -0.18552052141998032, -0.19951405956802407,
                  -0.20306661730037573, -0.2130929704046349, -0.23109815855579335,
                  -0.2500552520690091, -0.2636158151349759, -0.268412568602741,
                  -0.2730353882345693, -0.2844327851651806, -0.2969302167940793,
                  -0.30567633889851736, -0.30935363220923506, -0.3104606141551865,
                  -0.3138487066565867, -0.3161079342833782, -0.3146323233472668,
                  -0.31143848659084417, -0.3103111991123877, -0.3063695130000694,
                  -0.29610797074083817, -0.2798689974197797, -0.2637840314630642,
                  -0.256959258590788, -0.24970925520558615, -0.22836003518647713,
                  -0.19666284393997444, -0.16490286167057094, -0.14664780208467387,
                  -0.1400967815924841, -0.11393718546115707, -0.07229522166883755,
                  -0.02876357704403918, 0.0006893474578849874, 0.00834770098017756,
                  0.03033886249615677, 0.07115533338885446, 0.11585389615167233,
                  0.14889687382302802, 0.16081345508408768, 0.17242339898794337,
                  0.20167334048420674, 0.2352662878247843, 0.26071690396404323,
                  0.2725278098522258, 0.2763388829863348, 0.289562883607562,
                  0.30494352817593645, 0.3148361224416993, 0.3184371471998132,
                  0.3189956780890423, 0.31976826683863097, 0.31803327794963066,
                  0.31145690651138874, 0.30330213265226985, 0.2996141494839335,
                  0.29560753786650734, 0.28349703217117483, 0.265184092043082,
                  0.24682117995410455, 0.2363213577230064]
    expected_vt = [48.6992868526641, 48.66061151828625, 48.562472749119,
                   48.447386590870806, 48.35689214185663, 48.31320288737565,
                   48.29877167655277, 48.246917255621504, 48.18065639743303,
                   48.12907208406565, 48.10305008519001, 48.09737860085546,
                   48.08352585895935, 48.06723944601686, 48.06375753887273,
                   48.07173886211495, 48.07710175827975, 48.083736776797075,
                   48.10750394035914, 48.15078049524789, 48.2005548271185,
                   48.23137401970733, 48.24277819914424, 48.28999073879341,
                   48.37030021545349, 48.460829577792815, 48.526061475669344,
                   48.543585047501594, 48.595282117266834, 48.69721704693387,
                   48.81957915246863, 48.91927408797529, 48.9576306807959,
                   48.99644006244413, 49.101750081571524, 49.240705451992845,
                   49.365985339465844, 49.43340450803446, 49.45692420763894,
                   49.547788154586335, 49.68414133910146, 49.8183730789756,
                   49.905474733027326, 49.92771042907716, 49.99072482092697,
                   50.10483804117641, 50.226425429950744, 50.314571337911886,
                   50.346069312688655, 50.376628012585485, 50.45311695531929,
                   50.54016009674209, 50.605428974548346, 50.63538718425807,
                   50.644981677404, 50.677832316151594, 50.71421623646161,
                   50.73401425585734, 50.73767117504187, 50.73736594876943,
                   50.73360398839693, 50.71500032904826, 50.67588851616629,
                   50.632401347623436, 50.61316532492046, 50.59237053079966,
                   50.52956408310521, 50.43348269604562, 50.33503576389008,
                   50.27770400767619, 50.25701107772187, 50.17378212319549,
                   50.039327077197896, 49.89596658563543, 49.797081906296036,
                   49.771085589662256, 49.69571049233119, 49.5525228842394,
                   49.38951061127272, 49.26344621978354, 49.21650697142358,
                   49.16988299867822, 49.04771792593905, 48.896115879997794,
                   48.76869738844039, 48.70375640464302]

    @testset "Braginskii electron IMEX timestepping" verbose=use_verbose begin
        println("Braginskii electron IMEX timestepping tests")

        @testset "Split 3" begin
            test_input["output"]["base_directory"] = test_output_directory
            run_test(test_input, expected_p, expected_q, expected_vt)
        end
        @long @testset "Check other timestep - $type" for
                type ∈ ("KennedyCarpenterARK324",)

            timestep_check_input = deepcopy(test_input)
            timestep_check_input["output"]["base_directory"] = test_output_directory
            timestep_check_input["output"]["run_name"] = type
            timestep_check_input["timestepping"]["type"] = type
            run_test(timestep_check_input, expected_p, expected_q, expected_vt)
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
