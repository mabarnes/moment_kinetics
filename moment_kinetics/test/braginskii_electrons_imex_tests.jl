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
                                             "electron_ionization_frequency" => 0.0),
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

    expected_p = [0.4472667753573, 0.44586573052015466, 0.4424647096166432,
                  0.43880148354600546, 0.4362338056099866, 0.4351231286737388,
                  0.43477895664414823, 0.43365593088225596, 0.43257823403449747,
                  0.43223077674966537, 0.4323953669714128, 0.43248849111426135,
                  0.43286984145693197, 0.4340302551115443, 0.43602047637959573,
                  0.4380477467623677, 0.4389146087536753, 0.4398379346877904,
                  0.44256609548429504, 0.4466347628125969, 0.45073775485714207,
                  0.45311308823748697, 0.45396929815757175, 0.45741231101077917,
                  0.462992995661822, 0.46900364137058925, 0.47320430814053555,
                  0.4743176908892343, 0.47756994729526264, 0.4838605291598065,
                  0.4912370470466483, 0.4971308287885936, 0.4993737621289088,
                  0.5016301082844635, 0.5076897886177998, 0.5155519153447844,
                  0.5225137910929818, 0.5262101842133429, 0.5274912429744992,
                  0.5323979131182897, 0.5396262894584034, 0.5465644289618494,
                  0.5509571249788077, 0.5520629883964039, 0.5551591883386835,
                  0.5606050354082762, 0.5661242922789107, 0.5698855786467936,
                  0.5711677515774394, 0.5723747879575733, 0.5752039811770493,
                  0.5779653686021129, 0.5795228328585527, 0.5799950142571424,
                  0.5800990332324834, 0.5802025129877998, 0.5794475559292418,
                  0.5775860811587035, 0.5757457393730918, 0.5751917019939247,
                  0.5734266125158594, 0.5694471729004168, 0.563954552931989,
                  0.5590156030954267, 0.5570216393839076, 0.5549566689161276,
                  0.5491379571481412, 0.5410541224525021, 0.5334506536106367,
                  0.5292589393149492, 0.5277823837805637, 0.5220167160265328,
                  0.5132103110360944, 0.5043996594930746, 0.49862308603756,
                  0.4971418463597247, 0.49293044374252115, 0.485253207324287,
                  0.47699292326869625, 0.47093510094532054, 0.46875089498826766,
                  0.46661928268970854, 0.461212786847941, 0.4548676297024186,
                  0.44986029090974355, 0.4474307681332574]
    expected_q = [0.6773526451128238, 0.670583999021797, 0.6493886936653975,
                  0.6157359500217003, 0.5804376967961619, 0.5596337683940502,
                  0.5520884618821148, 0.5215589214269033, 0.47165961574381243,
                  0.41762978109691684, 0.3797992913311255, 0.3697685979640904,
                  0.3404660459654843, 0.2837914706113863, 0.217344682639778,
                  0.1642308040656865, 0.14400193349152263, 0.12363882636780353,
                  0.06884919066760159, -0.0025785696544894806, -0.06633807940552049,
                  -0.10046307561383809, -0.11234396149687136, -0.1581512531134767,
                  -0.22672491652613394, -0.29421227846226833, -0.33809300344022425,
                  -0.34931457578099034, -0.38117689954727624, -0.4391741381555335,
                  -0.501575739061081, -0.5473683674328982, -0.5638771699478984,
                  -0.5799806191665012, -0.6207340444297192, -0.6681331503508241,
                  -0.7047507032435498, -0.7220311587726211, -0.7276524782441415,
                  -0.7473515305771995, -0.7706226333743329, -0.7855209416550488,
                  -0.7904681155606789, -0.7910821989001742, -0.7912803709037391,
                  -0.7852740210044945, -0.7681916679409239, -0.7472160866320815,
                  -0.7375866366230286, -0.7270004378513353, -0.6941685680972457,
                  -0.642170986350678, -0.5873165464655995, -0.5548068899004526,
                  -0.5429860662994594, -0.4950235043954368, -0.4163406850257616,
                  -0.33116882409660287, -0.271824124909181, -0.2561572815629586,
                  -0.21060197644705128, -0.12364331044604336, -0.024257155821442774,
                  0.05264922083668048, 0.08125254896573005, 0.10963254026115248,
                  0.18375652132254605, 0.2750046993849566, 0.35073171063740033,
                  0.3888620927820992, 0.40172591212930353, 0.4492718653540388,
                  0.5140694693311022, 0.5699172475856528, 0.6017607551833434,
                  0.609314631225841, 0.629411904044992, 0.6606553098210337,
                  0.6859771459149894, 0.6984902863851211, 0.7016072172421283,
                  0.7038742159983571, 0.705858015204439, 0.700059410583365,
                  0.6874958260779616, 0.6780812480898903]
    expected_vt = [60.351816207604635, 60.31012015049677, 60.200604221763534,
                   60.064056388407685, 59.9486166098168, 59.88950555572966,
                   59.86938336673384, 59.794093675971, 59.688555354831614,
                   59.59374628595299, 59.53732958585759, 59.52360365226546,
                   59.486265902734885, 59.424892586011715, 59.369427971140304,
                   59.33676186929905, 59.32688776618443, 59.318330034971396,
                   59.301990065786086, 59.29478194397539, 59.30132730953455,
                   59.309748023136486, 59.31347718556533, 59.33169891964568,
                   59.37041176928683, 59.422126924870454, 59.46325312228893,
                   59.47475465628287, 59.50966972516561, 59.58225272283746,
                   59.674748539452345, 59.753654804363926, 59.784751684376296,
                   59.81660062329952, 59.90483646250452, 60.02494223227007,
                   60.13646982486386, 60.19767945902454, 60.21922333352952,
                   60.30336655092559, 60.432344078530356, 60.56256699334704,
                   60.64889991624143, 60.67118554864238, 60.73491855012813,
                   60.852702387568364, 60.982164600875514, 61.07923838490804,
                   61.11473832613493, 61.14965858786135, 61.23953948372659,
                   61.34769829901661, 61.43539250374229, 61.47878240685701,
                   61.49329290414419, 61.54628496978551, 61.61648033100149,
                   61.67412230312648, 61.70500460236538, 61.71201348070227,
                   61.72983625925911, 61.75383223671623, 61.7658956739661,
                   61.76406330619319, 61.76082451138627, 61.75619056494155,
                   61.73705790442811, 61.69800694295683, 61.65011541510284,
                   61.6195413583548, 61.60809623511118, 61.560115308047024,
                   61.476866628666976, 61.38144082319236, 61.312038431820156,
                   61.29333291605387, 61.238042443579566, 61.12876555363245,
                   60.99762498726125, 60.8911766729031, 60.85037690232061,
                   60.809203653504845, 60.698124222165994, 60.55330673128157,
                   60.424808096655475, 60.35658592031778]

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
