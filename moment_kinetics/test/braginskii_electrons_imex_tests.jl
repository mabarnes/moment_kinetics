module BraginskiiElectronsIMEX

# Regression test using wall boundary conditions, with recycling fraction less than 1 and
# a plasma source. Runs for a while and then checks phi profile against saved reference
# output.

include("setup.jl")

using Base.Filesystem: tempname
using MPI

using moment_kinetics.coordinates: define_coordinate
using moment_kinetics.input_structs: grid_input, advection_input
using moment_kinetics.interpolation: interpolate_to_grid_z
using moment_kinetics.load_data: open_readonly_output_file
using moment_kinetics.load_data: load_electron_moments_data

# default inputs for tests
test_input = Dict("n_ion_species" => 1,
                  "n_neutral_species" => 1,
                  "electron_physics" => "braginskii_fluid",
                  "run_name" => "braginskii-electrons-imex",
                  "evolve_moments_density" => true,
                  "evolve_moments_parallel_flow" => true,
                  "evolve_moments_parallel_pressure" => true,
                  "evolve_moments_conservation" => true,
                  "T_e" => 0.2,
                  "nu_ei" => 1.0e3,
                  "initial_density1" => 1.0,
                  "initial_temperature1" => 1.0,
                  "z_IC_option1" => "sinusoid",
                  "z_IC_density_amplitude1" => 0.1,
                  "z_IC_density_phase1" => 0.0,
                  "z_IC_upar_amplitude1" => 1.0,
                  "z_IC_upar_phase1" => 0.0,
                  "z_IC_temperature_amplitude1" => 0.1,
                  "z_IC_temperature_phase1" => 1.0,
                  "vpa_IC_option1" => "gaussian",
                  "vpa_IC_density_amplitude1" => 1.0,
                  "vpa_IC_density_phase1" => 0.0,
                  "vpa_IC_upar_amplitude1" => 0.0,
                  "vpa_IC_upar_phase1" => 0.0,
                  "vpa_IC_temperature_amplitude1" => 0.0,
                  "vpa_IC_temperature_phase1" => 0.0,
                  "initial_density2" => 1.0,
                  "initial_temperature2" => 1.0,
                  "z_IC_option2" => "sinusoid",
                  "z_IC_density_amplitude2" => 0.001,
                  "z_IC_density_phase2" => 0.0,
                  "z_IC_upar_amplitude2" => 0.0,
                  "z_IC_upar_phase2" => 0.0,
                  "z_IC_temperature_amplitude2" => 0.0,
                  "z_IC_temperature_phase2" => 0.0,
                  "vpa_IC_option2" => "gaussian",
                  "vpa_IC_density_amplitude2" => 1.0,
                  "vpa_IC_density_phase2" => 0.0,
                  "vpa_IC_upar_amplitude2" => 0.0,
                  "vpa_IC_upar_phase2" => 0.0,
                  "vpa_IC_temperature_amplitude2" => 0.0,
                  "vpa_IC_temperature_phase2" => 0.0,
                  "charge_exchange_frequency" => 0.75,
                  "ionization_frequency" => 0.5,
                  "constant_ionization_rate" => false,
                  "timestepping" => Dict{String,Any}("type" => "KennedyCarpenterARK324",
                                                     "implicit_ion_advance" => false,
                                                     "implicit_vpa_advection" => false,
                                                     "nstep" => 10000,
                                                     "dt" => 1.0e-6,
                                                     "minimum_dt" => 1.e-7,
                                                     "rtol" => 1.0e-7,
                                                     "nwrite" => 10000,
                                                     "high_precision_error_sum" => true),
                  "nonlinear_solver" => Dict{String,Any}("nonlinear_max_iterations" => 100),
                  "r_ngrid" => 1,
                  "r_nelement" => 1,
                  "z_ngrid" => 17,
                  "z_nelement" => 16,
                  "z_bc" => "periodic",
                  "z_discretization" => "chebyshev_pseudospectral",
                  "vpa_ngrid" => 6,
                  "vpa_nelement" => 31,
                  "vpa_L" => 12.0,
                  "vpa_bc" => "zero",
                  "vpa_discretization" => "chebyshev_pseudospectral",
                  "vz_ngrid" => 6,
                  "vz_nelement" => 31,
                  "vz_L" => 12.0,
                  "vz_bc" => "zero",
                  "vz_discretization" => "chebyshev_pseudospectral",
                  "ion_numerical_dissipation" => Dict{String,Any}("force_minimum_pdf_value" => 0.0,
                                                                  "vpa_dissipation_coefficient" => 1e0),
                  "neutral_numerical_dissipation" => Dict{String,Any}("force_minimum_pdf_value" => 0.0,
                                                                      "vz_dissipation_coefficient" => 1e-1))

if global_size[] > 2 && global_size[] % 2 == 0
    # Test using distributed-memory
    test_input["z_nelement_local"] = test_input["z_nelement"] ÷ 2
end

"""
Run a test for a single set of parameters
"""
function run_test(test_input, expected_p, expected_q, expected_vt; rtol=4.e-14,
                  atol=1.e-15, args...)
    # by passing keyword arguments to run_test, args becomes a Tuple of Pairs which can be
    # used to update the default inputs

    # Make a copy to make sure nothing modifies the input Dicts defined in this test
    # script.
    test_input = deepcopy(test_input)

    # Convert keyword arguments to a unique name
    name = test_input["run_name"]
    if length(args) > 0
        name = string(name, "_", (string(k, "-", v, "_") for (k, v) in args)...)

        # Remove trailing "_"
        name = chop(name)
    end

    # Provide some progress info
    println("    - testing ", name)

    # Convert dict from symbol keys to String keys
    modified_inputs = Dict(String(k) => v for (k, v) in args)

    # Update default inputs with values to be changed
    input = merge(test_input, modified_inputs)

    input["run_name"] = name

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

            path = joinpath(realpath(input["base_directory"]), name, name)

            # open the netcdf file and give it the handle 'fid'
            fid = open_readonly_output_file(path,"moments")

            # load fields data
            parallel_pressure_zrt, parallel_heat_flux_zrt, thermal_speed_zrt =
            load_electron_moments_data(fid)

            close(fid)
            
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
            @test isapprox(actual_q, expected_q, rtol=rtol, atol=atol)
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

    expected_p = [0.44951398349003646, 0.4481071794001275, 0.4446923360697697,
                  0.44101458684278777, 0.4384371320097695, 0.43732241798810445,
                  0.4369770330480599, 0.4358502579878266, 0.434769684040917,
                  0.43442263207284637, 0.43458925629798073, 0.4346831326574436,
                  0.4350671485602161, 0.4362345316295716, 0.438235619583819,
                  0.4402734411528, 0.4411447249385275, 0.4420727185406772,
                  0.44481447378612293, 0.44890303238070045, 0.4530257687705389,
                  0.4554124231936834, 0.4562726972649091, 0.4597319740656588,
                  0.46533878727932276, 0.4713773067908409, 0.47559731298259184,
                  0.4767158034722211, 0.47998294113890483, 0.4863021470825949,
                  0.4937119852803594, 0.49963221191774776, 0.5018851701590787,
                  0.5041515792976728, 0.5102381811386025, 0.5181350099985299,
                  0.525127401588624, 0.52883991125208, 0.530126542118299,
                  0.5350544857912003, 0.5423139848496283, 0.5492817298572252,
                  0.5536930071759895, 0.5548035254843448, 0.5579127055178238,
                  0.5633811620967428, 0.5689229399038384, 0.5726992489776701,
                  0.5739864539763129, 0.5751981737538447, 0.5780380624660489,
                  0.5808091879153607, 0.582371244425747, 0.5828442626520689,
                  0.5829483019641976, 0.5830505324419235, 0.582289662647816,
                  0.5804174302674363, 0.5785673270305791, 0.5780104294159015,
                  0.5762363804591635, 0.5722372963746002, 0.5667182906200348,
                  0.5617560490141181, 0.5597527656925925, 0.5576781862353368,
                  0.5518325952127137, 0.5437117958155238, 0.5360738744744328,
                  0.5318632735767702, 0.5303800815638172, 0.52458857332585,
                  0.5157429166779243, 0.5068932260184091, 0.5010911813300937,
                  0.4996034253232407, 0.49537353092296554, 0.4876627194305651,
                  0.47936652714814293, 0.4732825296867743, 0.47108892532041974,
                  0.46894816235702474, 0.4635185649133237, 0.4571465360817948,
                  0.452118238600268, 0.4496786524823512]
    expected_q = [0.6831704519267778, 0.6763596731108003, 0.6550248548395122,
                  0.621137479330641, 0.5855826766086369, 0.564624348243424,
                  0.5570221429154738, 0.5262612511440813, 0.4759763560524193,
                  0.4215209814677685, 0.3833879518808367, 0.3732765291862611,
                  0.34373639264749195, 0.2865988207461233, 0.21960124384883264,
                  0.16604253330270563, 0.14564326706615302, 0.1251073951350269,
                  0.06985203286011953, -0.002189422724091073, -0.06650025444931523,
                  -0.10092291876877173, -0.11290749491826756, -0.15911660726647084,
                  -0.22829407549560776, -0.296381042779731, -0.34065325931034834,
                  -0.35197568475314056, -0.38412394718970916, -0.44264467058363494,
                  -0.5056139329952417, -0.5518262734055381, -0.5684878570815051,
                  -0.5847404886752257, -0.6258728785916116, -0.6737205371368551,
                  -0.7106893697823674, -0.728139347093567, -0.733817023138346,
                  -0.7537144673017869, -0.7772295129285097, -0.7923028306406594,
                  -0.7973216863277829, -0.7979485411906018, -0.7981733058556373,
                  -0.7921566532586084, -0.7749750759560777, -0.7538532886753724,
                  -0.7441527309138952, -0.7334870232950073, -0.7004061657361658,
                  -0.6479952272743823, -0.5926982319505321, -0.5599217222341915,
                  -0.5480028121415452, -0.49964097922347867, -0.4203001696008753,
                  -0.3344059204499378, -0.2745563922655548, -0.258755893042823,
                  -0.21281014297392614, -0.1251037616154198, -0.02486081101616467,
                  0.052711670524221245, 0.08156346618184351, 0.11018860125597137,
                  0.184956394584215, 0.2769990401549583, 0.3533862259663847,
                  0.39185061791993997, 0.4048279100244915, 0.4527903930933007,
                  0.5181604219781556, 0.5745067111528425, 0.6066373424047252,
                  0.6142607159164151, 0.6345436396519946, 0.6660807860559502,
                  0.6916551627488451, 0.7043061657863463, 0.7074613359074287,
                  0.7097605448059333, 0.7117980175789959, 0.7059983443925062,
                  0.693373867220681, 0.6839039812030164]
    expected_vt = [60.50323643979975, 60.46152282311776, 60.35195588066853,
                   60.21533395043499, 60.09982168515988, 60.04066964102238,
                   60.02053283714272, 59.94518544878992, 59.83955672767372,
                   59.74465499035528, 59.688176306690224, 59.674434375999695,
                   59.63705080297302, 59.575593022878486, 59.5200362867479,
                   59.487302095351524, 59.477403317059846, 59.46882152619191,
                   59.45242049432979, 59.44514092412334, 59.45163053025197,
                   59.460024506000224, 59.463744883587445, 59.481935299569656,
                   59.52060886409094, 59.572294471701106, 59.61340646988501,
                   59.624904994314015, 59.6598130677176, 59.73238906214771,
                   59.82488626148378, 59.9037999413929, 59.934901010623335,
                   59.966754837133514, 60.05500712816102, 60.17514068478408,
                   60.28669865440711, 60.34792638918577, 60.36947690125584,
                   60.45364714009672, 60.58266887898263, 60.71293986868975,
                   60.79930625245654, 60.82160071089043, 60.8853594969313,
                   61.00319313188073, 61.132712888454186, 61.229832513831234,
                   61.265349742432086, 61.30028735647212, 61.39021480819434,
                   61.49843415833545, 61.586182165745534, 61.62960105016664,
                   61.644121615044114, 61.69715263940207, 61.76740693027959,
                   61.825107969209625, 61.8560293732873, 61.86304841080026,
                   61.880900287439, 61.90495043959934, 61.91707450705104,
                   61.91528849056989, 61.912066851775705, 61.907449920498564,
                   61.88836178502154, 61.84936547495241, 61.801519377780714,
                   61.77096806875504, 61.75953059854668, 61.711577658100396,
                   61.62836639460284, 61.53297073619124, 61.46358406675648,
                   61.44488197626709, 61.38959988569385, 61.280332330036074,
                   61.14919211755058, 61.04273591939215, 61.00193140312018,
                   60.96075236762326, 60.84965248313868, 60.70479794465818,
                   60.57625670314503, 60.50800809804206]

    @testset "Braginskii electron IMEX timestepping" verbose=use_verbose begin
        println("Braginskii electron IMEX timestepping tests")

        @testset "Split 3" begin
            test_input["base_directory"] = test_output_directory
            run_test(test_input, expected_p, expected_q, expected_vt)
        end
        @long @testset "Check other timestep - $type" for
                type ∈ ("KennedyCarpenterARK437",)

            timestep_check_input = deepcopy(test_input)
            timestep_check_input["base_directory"] = test_output_directory
            timestep_check_input["run_name"] = type
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
