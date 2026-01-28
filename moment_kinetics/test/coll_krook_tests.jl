module coll_krook_tests

# Test generated from TOML input files

include("setup.jl")

using Base.Filesystem: tempname
using MPI

using moment_kinetics.interpolation: interpolate_to_grid_z
using moment_kinetics.load_data: get_run_info_no_setup, close_run_info,
                                 postproc_load_variable
using moment_kinetics.utils: merge_dict_with_kwargs!

# default inputs for tests
dkions_n10 = OptionsDict(
 "output" => OptionsDict(
     "display_timing_info" => false
    ),
 "r" => OptionsDict(
     "ngrid" => 1,
     "nelement" => 1
    ),
 "evolve_moments" => OptionsDict(
     "pressure" => true,
     "density" => true,
     "moments_conservation" => true,
     "parallel_flow" => true
    ),
 "ion_species_1" => OptionsDict(
     "initial_temperature" => 0.3333333333333333,
     "initial_density" => 40.0
    ),
 "krook_collisions" => OptionsDict(
     "use_krook" => true,
     "frequency_option" => "reference_parameters"
    ),
 "vpa" => OptionsDict(
     "ngrid" => 6,
     "discretization" => "chebyshev_pseudospectral",
     "nelement" => 31,
     "L" => 14.0,
     "element_spacing_option" => "coarse_tails8.660254037844386",
     "bc" => "zero"
    ),
 "z" => OptionsDict(
     "ngrid" => 5,
     "discretization" => "chebyshev_pseudospectral",
     "nelement" => 20,
     "L" => 1.4,
     "bc" => "wall"
    ),
 "vpa_IC_ion_species_1" => OptionsDict(
     "initialization_option" => "gaussian",
     "density_amplitude" => 1.0,
     "temperature_amplitude" => 0.0,
     "density_phase" => 0.0,
     "upar_amplitude" => 0.0,
     "temperature_phase" => 0.0,
     "upar_phase" => 0.0
    ),
 "composition" => OptionsDict(
     "T_e" => 0.8,
     "electron_physics" => "boltzmann_electron_response",
     "n_ion_species" => 1,
     "n_neutral_species" => 0
    ),
 "ion_source_2" => OptionsDict(
     "source_type" => "density_midpoint_control",
     "source_T" => 0.1,
     "active" => true,
     "PI_density_controller_I" => 5.0,
     "source_strength" => 40.0,
     "z_profile" => "wall_exp_decay",
     "PI_density_controller_P" => 7.0,
     "PI_density_target_amplitude" => 10.0,
     "z_width" => 0.2
    ),
 "z_IC_ion_species_1" => OptionsDict(
     "initialization_option" => "gaussian",
     "density_amplitude" => 0.001,
     "temperature_amplitude" => 0.0,
     "density_phase" => 0.0,
     "upar_amplitude" => 1.4142135623730951,
     "temperature_phase" => 0.0,
     "upar_phase" => 0.0
    ),
 "ion_source_1" => OptionsDict(
     "PI_temperature_controller_I" => 500.0,
     "source_type" => "temperature_midpoint_control",
     "source_T" => 1.5,
     "active" => true,
     "PI_temperature_target_amplitude" => 0.3333333333333333,
     "source_strength" => 14.0,
     "z_profile" => "super_gaussian_4",
     "PI_temperature_controller_P" => 500.0,
     "z_width" => 0.38
    ),
 "timestepping" => OptionsDict(
     "nstep" => 4000,
     "steady_state_residual" => true,
     "dt" => 0.0005,
     "nwrite" => 2000,
     "type" => "SSPRK4",
     "nwrite_dfns" => 2000,
     "print_nT_live" => true
    )
)

coll_krook_n10 = recursive_merge(dkions_n10,
                               OptionsDict(
 "vpa" => OptionsDict(
     "ngrid" => 1,
     "nelement" => 1,
    ),
 "composition" => OptionsDict(
     "ion_physics" => "coll_krook_ions"
    ),
 "z" => OptionsDict(
     "ngrid" => 6,
     "nelement" => 40,
    ),
 "timestepping" => OptionsDict(
     "nstep" => 10000,
     "dt" => 0.00005,
     "nwrite" => 1000,
     "nwrite_dfns" => 1000,
    )
))

dkions_n1 = recursive_merge(dkions_n10,
                               OptionsDict(
 "ion_source_2" => OptionsDict(
     "source_strength" => 4.0,
     "PI_density_target_amplitude" => 1.0
    ),
 "vpa" => OptionsDict(
     "nelement" => 41
    ),
 "ion_source_1" => OptionsDict(
     "source_strength" => 1.4
    ),
 "timestepping" => OptionsDict(
     "nstep" => 400,
     "nwrite" => 200,
     "nwrite_dfns" => 200
    ),
 "z" => OptionsDict(
     "nelement" => 32
    ),
 "ion_species_1" => OptionsDict(
     "initial_density" => 4.0
    )
))

coll_krook_n1 = recursive_merge(dkions_n1,
                               OptionsDict(
 "vpa" => OptionsDict(
     "ngrid" => 1,
     "nelement" => 1,
    ),
 "z" => OptionsDict(
     "ngrid" => 6,
     "nelement" => 40,
    ),
 "composition" => OptionsDict(
     "ion_physics" => "coll_krook_ions"
    ),
 "timestepping" => OptionsDict(
     "nstep" => 10000,
     "dt" => 0.000005,
     "nwrite" => 1000,
     "nwrite_dfns" => 1000,
    )
))

dkions_n100 = recursive_merge(dkions_n10,
                               OptionsDict(
 "ion_source_2" => OptionsDict(
     "PI_density_controller_I" => 500.0,
     "source_strength" => 200.0,
     "PI_density_controller_P" => 700.0,
     "PI_density_target_amplitude" => 100.0
    ),
 "vpa" => OptionsDict(
     "nelement" => 41
    ),
 "ion_source_1" => OptionsDict(
     "source_strength" => 140.0
    ),
 "timestepping" => OptionsDict(
     "nstep" => 80,
     "dt" => 0.0002,
     "nwrite" => 40,
     "nwrite_dfns" => 40
    ),
 "z" => OptionsDict(
     "nelement" => 32
    ),
 "ion_species_1" => OptionsDict(
     "initial_density" => 130.0
    )
))

coll_krook_n100 = recursive_merge(dkions_n100,
                               OptionsDict(
 "vpa" => OptionsDict(
     "ngrid" => 1,
     "nelement" => 1,
    ),
 "z" => OptionsDict(
     "ngrid" => 6,
     "nelement" => 40,
    ),
 "composition" => OptionsDict(
     "ion_physics" => "coll_krook_ions"
    ),
 "timestepping" => OptionsDict(
     "nstep" => 10000,
     "dt" => 0.00005,
     "nwrite" => 1000,
     "nwrite_dfns" => 1000,
    )
))


# Here choose the names for each test
dkions_n10 = recursive_merge(dkions_n10,
                               OptionsDict("output" => OptionsDict("run_name" => "dkions_n10.0_for_test_generation")))
coll_krook_n10 = recursive_merge(coll_krook_n10,
                               OptionsDict("output" => OptionsDict("run_name" => "coll_krook_n10.0")))
dkions_n1 = recursive_merge(dkions_n1,
                               OptionsDict("output" => OptionsDict("run_name" => "dkions_n1.0_for_test_generation")))
coll_krook_n1 = recursive_merge(coll_krook_n1,
                               OptionsDict("output" => OptionsDict("run_name" => "coll_krook_n1.0")))
dkions_n100 = recursive_merge(dkions_n100,
                               OptionsDict("output" => OptionsDict("run_name" => "dkions_n100.0_for_test_generation")))
coll_krook_n100 = recursive_merge(coll_krook_n100,
                               OptionsDict("output" => OptionsDict("run_name" => "coll_krook_n100.0")))

"""
Run a test for a single set of parameters
"""
function run_test_with_restart(dkions_test_input, coll_krook_test_input, expected_phi; rtol=4.e-14, atol=1.e-15, args...)
    # by passing keyword arguments to run_test, args becomes a Tuple of Pairs which can be
    # used to update the default inputs

    # Make a copy to make sure nothing modifies the input Dicts defined in this test
    # script.
    dkions_input = deepcopy(dkions_test_input)
    coll_krook_input = deepcopy(coll_krook_test_input)

     # Convert keyword arguments to a unique name
    dkions_name = dkions_input["output"]["run_name"]
    if length(args) > 0
        dkions_name = string(dkions_name, "_", (string(k, "-", v, "_") for (k, v) in args)...)  
        # Remove trailing "_"
        dkions_name = chop(dkions_name)
    end

    # Convert keyword arguments to a unique name
    coll_krook_name = coll_krook_input["output"]["run_name"]
    if length(args) > 0
        coll_krook_name = string(coll_krook_name, "_", (string(k, "-", v, "_") for (k, v) in args)...)

        # Remove trailing "_"
        coll_krook_name = chop(coll_krook_name)
    end

    # Provide some progress info
    println("    - testing ", coll_krook_name)

    # Update default inputs with values to be changed
    merge_dict_with_kwargs!(dkions_input; args...)
    dkions_input["output"]["run_name"] = dkions_name
    # Suppress console output while running
    phi = undef
    quietoutput() do
        # run simulation
        run_moment_kinetics(dkions_input)

        # now run the coll_krook simulation restarting from the dkions output
        name_of_restart_file = dkions_name * ".dfns.h5"
        run_moment_kinetics(coll_krook_input, restart = joinpath(
            realpath(dkions_input["output"]["base_directory"]),
            dkions_name, name_of_restart_file))
    end

    if global_rank[] == 0
        quietoutput() do
            # Load and analyse output
            #########################

            path = joinpath(realpath(coll_krook_input["output"]["base_directory"]), coll_krook_name)

            # open the output file(s)
            run_info = get_run_info_no_setup(path)

            # load fields data
            phi_zrt = postproc_load_variable(run_info, "phi")

            close_run_info(run_info)
            
            phi = phi_zrt[:,1,:]
        end

        # Regression test
        actual_phi = phi[begin:3:end, end]
        if expected_phi == nothing
            # Error: no expected input provided
            println("data tested would be: ", actual_phi)
            @test false
        else
            @test isapprox(actual_phi, expected_phi, rtol=rtol, atol=atol)
        end
    end
end

function runtests()
    # Create a temporary directory for test output
    test_output_directory = get_MPI_tempdir()

    @testset "coll_krook tests" verbose=use_verbose begin
        println("coll_krook tests")
        @testset "coll_krook_test_n1.0" begin
            dkions_n1["output"]["base_directory"] = test_output_directory
            run_test_with_restart(dkions_n1, coll_krook_n1,
                        [0.8842634458213864, 0.8812488806345314, 0.8775345487526505,
                        0.8678231008557887, 0.8614227542056043, 0.8508347037275611,
                        0.8389856602349453, 0.8302725938601654, 0.8129997378028317,
                        0.8028695314917359, 0.7870833940836338, 0.7705899431699565,
                        0.7591803946312081,0.7379700633047853, 0.726331638329607,
                        0.7092235667751957, 0.6923990402253128, 0.6813777169807153,
                        0.662028908635982, 0.6520279628107796, 0.6380083015271908,
                        0.6250085630020819, 0.6168311896451611, 0.6031150184652299,
                        0.5963727882101739, 0.5872868370230311, 0.5792966500257297,
                        0.5745746384335557, 0.5673008591887192, 0.5640647091545209,
                        0.5601578458489653, 0.5573632756786828, 0.5561047975301222,
                        0.5550659786604735, 0.5552091583991401, 0.5563360037022237,
                        0.5585416221149674, 0.5606599919234262, 0.5658384693326338,
                        0.5693604125452796, 0.575551093671854, 0.5829262795582846,
                        0.5885462953765741, 0.600131277682945, 0.6071687084813014,
                        0.6185612724219882, 0.6310195213237633, 0.639990092387538,
                        0.6576462166116419, 0.6678781816759793, 0.6837377580679918,
                        0.7002890603564625, 0.7117040839608457, 0.7329264845258185,
                        0.7445646615695124, 0.7616646712082016, 0.7784627965031139,
                        0.7894304312257568, 0.8086668025566477, 0.818509027296723,
                        0.8322116670256634, 0.844734218687321, 0.8524850249735572,
                        0.8651201722969568, 0.8710895291778953, 0.8785123160850224,
                        0.8833752895006004]
)
        end
        @testset "coll_krook_test_n10" begin
            dkions_n10["output"]["base_directory"] = test_output_directory
            run_test_with_restart(dkions_n10, coll_krook_n10,
                        [0.8496194750592662, 1.117208202360778, 1.15808284123184,
                        1.181228672587099, 1.1858720882479592, 1.180766837802604,
                        1.1718120996871564, 1.163073844436221, 1.1463971348710094,
                        1.1363615782197185, 1.1228969552091963, 1.1102732902302292,
                        1.1031710292738226, 1.0921919493464862, 1.086528136708633,
                        1.0806210452944625, 1.0795526528081483, 1.0817892953459864,
                        1.093078111275722, 1.1027768412431473, 1.1212415722095908,
                        1.142905092957713, 1.1586211492368919, 1.18787375710512,
                        1.2034786528898493, 1.2253423763809521, 1.2451982236294739,
                        1.2571927919381531, 1.2760696538729688, 1.2845589131668373,
                        1.2947706868823985, 1.302108895809166, 1.3054259358518778,
                        1.3081359099744665, 1.3077625947818985, 1.3048214734140486,
                        1.2990011233329009, 1.2934610215193787, 1.2799045241538487,
                        1.2706834963766143, 1.2547006690072844, 1.2361031931988886,
                        1.2222811148300876, 1.1946858622408199, 1.1788881197239494,
                        1.1551491064007602, 1.1323913923739493, 1.1182552564745045,
                        1.0970061387862602, 1.088571401925482, 1.0811410998369613,
                        1.0793934623200159, 1.0813007365438545, 1.0896704771698578,
                        1.095590277268393, 1.1044860026762056, 1.1161378658341596,
                        1.124543381760358, 1.1420891868205525, 1.1513958710927046,
                        1.165380307491002, 1.1761995076495935, 1.18288204943767,
                        1.1839244881122206, 1.1790954803218752, 1.1498224639945986,
                        1.0772770250511523]
)
        end
        @testset "coll_krook_test_n100" begin
            dkions_n100["output"]["base_directory"] = test_output_directory
            run_test_with_restart(dkions_n100, coll_krook_n100,
                    [2.997246485025625, 3.2790410903028278, 3.328887681120636,
                    3.353869420841202, 3.3475769716280865, 3.3274536234023815,
                    3.30813459933686, 3.2937661668888385, 3.2687985269571267,
                    3.256564050489506, 3.2443112901955304, 3.2397947841804786,
                    3.2426167600353226, 3.2599572360266182, 3.276049393121782,
                    3.3065082422218945, 3.3428870495368743, 3.3691375354153474,
                    3.4182844427925847, 3.444307407353598, 3.480813918794553,
                    3.5137882814914243, 3.533796433820305, 3.5656003981734212,
                    3.5803652513915716, 3.5990993145966232, 3.6144076368494895,
                    3.622965066038144, 3.63547888677125, 3.6407405986883767,
                    3.646799164371213, 3.65095754894705, 3.652797603005248,
                    3.6542650143549076, 3.6540705146099173, 3.6524586664948284,
                    3.6492190375499485, 3.6460336382690155, 3.6378850677965495,
                    3.632020053580849, 3.62122886899136, 3.60758032560886,
                    3.596597517283854, 3.572208499343478, 3.5564484470830053,
                    3.529610054298926, 3.4987069175219645, 3.475666981387859,
                    3.42967715894536, 3.4031434990502216, 3.363415552782067,
                    3.3251421870875526, 3.301705955625504, 3.2663917875584847,
                    3.2529815200144627, 3.2414918182428245, 3.2408442132034083,
                    3.2453817909098195, 3.263250415285893, 3.275934476136049,
                    3.297280402643294, 3.3171525574913066, 3.331346844466132,
                    3.3525060224861694, 3.353662199995851, 3.3194371587624314,
                    3.220967865752475]
)
        end
    end

    if global_rank[] == 0
        # Delete output directory to avoid using too much disk space
        rm(realpath(test_output_directory); recursive=true)
    end
end

end

using .coll_krook_tests

coll_krook_tests.runtests()
