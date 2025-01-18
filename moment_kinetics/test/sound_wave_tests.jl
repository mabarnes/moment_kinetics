module SoundWaveTests

include("setup.jl")

using Base.Filesystem: tempname
#using Plots: plot, plot!, gui

using moment_kinetics.array_allocation: allocate_float
using moment_kinetics.input_structs: netcdf
using moment_kinetics.file_io: io_has_implementation
using moment_kinetics.load_data: open_readonly_output_file
using moment_kinetics.load_data: load_fields_data, load_time_data
using moment_kinetics.load_data: load_species_data, load_coordinate_data
using moment_kinetics.type_definitions
using moment_kinetics.analysis: analyze_fields_data
using moment_kinetics.analysis: fit_delta_phi_mode
using moment_kinetics.utils: merge_dict_with_kwargs!

const analytical_rtol = 3.e-2
const regression_rtol = 1.e-14
const regression_range = 5:10

# Use "netcdf" to test the NetCDF I/O if it is available (or if we are forcing optional
# dependencies to be used, e.g. for CI tests), otherwise fall back to "hdf5".
const binary_format = (force_optional_dependencies || io_has_implementation(netcdf)) ?
                      "netcdf" : "hdf5"

# default inputs for tests
test_input_finite_difference = OptionsDict("composition" => OptionsDict("n_ion_species" => 1,
                                                                        "n_neutral_species" => 1,
                                                                        "electron_physics" => "boltzmann_electron_response",
                                                                        "T_e" => 1.0),
                                           "ion_species_1" => OptionsDict("initial_density" => 0.5,
                                                                          "initial_temperature" => 1.0),
                                           "z_IC_ion_species_1" => OptionsDict("initialization_option" => "sinusoid",
                                                                               "density_amplitude" => 0.001,
                                                                               "density_phase" => 0.0,
                                                                               "upar_amplitude" => 0.0,
                                                                               "upar_phase" => 0.0,
                                                                               "temperature_amplitude" => 0.0,
                                                                               "temperature_phase" => 0.0),
                                           "neutral_species_1" => OptionsDict("initial_density" => 0.5,
                                                                              "initial_temperature" => 1.0),
                                           "z_IC_neutral_species_1" => OptionsDict("initialization_option" => "sinusoid",
                                                                                   "density_amplitude" => 0.001,
                                                                                   "density_phase" => 0.0,
                                                                                   "upar_amplitude" => 0.0,
                                                                                   "upar_phase" => 0.0,
                                                                                   "temperature_amplitude" => 0.0,
                                                                                   "temperature_phase" => 0.0),                                                                        
                                           "output" => OptionsDict("run_name" => "finite_difference",
                                                                   "binary_format" => binary_format),
                                           "evolve_moments" => OptionsDict("density" => false,
                                                                           "parallel_flow" => false,
                                                                           "parallel_pressure" => false,
                                                                           "moments_conservation" => true),
                                           "reactions" => OptionsDict("charge_exchange_frequency" => 2*π*0.1,
                                                                      "ionization_frequency" => 0.0),
                                           "timestepping" => OptionsDict("nstep" => 1500,
                                                                         "dt" => 0.002,
                                                                         "nwrite" => 20,
                                                                         "split_operators" => false),
                                           "r" => OptionsDict("ngrid" => 1,
                                                              "nelement" => 1,
                                                              "bc" => "periodic",
                                                              "discretization" => "finite_difference"),
                                           "z" => OptionsDict("ngrid" => 100,
                                                              "nelement" => 1,
                                                              "bc" => "periodic",
                                                              "discretization" => "finite_difference"),
                                           "vperp" => OptionsDict("ngrid" => 1,
                                                                  "nelement" => 1,
                                                                  "L" => 1.0,
                                                                  "discretization" => "finite_difference"),
                                           "vpa" => OptionsDict("ngrid" => 180,
                                                                "nelement" => 1,
                                                                "L" => 8.0,
                                                                "bc" => "periodic",
                                                                "discretization" => "finite_difference"),
                                           "vz" => OptionsDict("ngrid" => 180,
                                                               "nelement" => 1,
                                                               "L" => 8.0,
                                                               "bc" => "periodic",
                                                               "discretization" => "finite_difference"),
                                          )

test_input_finite_difference_split_1_moment =
    recursive_merge(test_input_finite_difference,
                    OptionsDict("output" => OptionsDict("run_name" => "finite_difference_split_1_moment"),
                                "evolve_moments" => OptionsDict("density" => true))
                   )

test_input_finite_difference_split_2_moments =
    recursive_merge(test_input_finite_difference_split_1_moment,
                    OptionsDict("output" => OptionsDict("run_name" => "finite_difference_split_2_moments"),
                                "evolve_moments" => OptionsDict("parallel_flow" => true),
                                "vpa" => OptionsDict("ngrid" => 270, "L" => 12.0),
                                "vz" => OptionsDict("ngrid" => 270, "L" => 12.0))
                   )

test_input_finite_difference_split_3_moments =
    recursive_merge(test_input_finite_difference_split_2_moments,
                    OptionsDict("output" => OptionsDict("run_name" => "finite_difference_split_3_moments"),
                                "evolve_moments" => OptionsDict("parallel_pressure" => true),
                                "vpa" => OptionsDict("ngrid" => 270, "L" => 12.0),
                                "vz" => OptionsDict("ngrid" => 270, "L" => 12.0))
                   )

test_input_chebyshev = recursive_merge(test_input_finite_difference,
                                       OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral"),
                                                   "z" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                      "ngrid" => 9,
                                                                      "nelement" => 2),
                                                   "vpa" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                        "ngrid" => 17,
                                                                        "nelement" => 8),
                                                   "vz" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                       "ngrid" => 17,
                                                                       "nelement" => 8),
                                                  ))

test_input_chebyshev_split_1_moment =
    recursive_merge(test_input_chebyshev,
                    OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_split_1_moment"),
                                "evolve_moments" => OptionsDict("density" => true)))

test_input_chebyshev_split_2_moments =
    recursive_merge(test_input_chebyshev_split_1_moment,
                    OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_split_2_moments"),
                                "evolve_moments" => OptionsDict("parallel_flow" => true)))

test_input_chebyshev_split_3_moments =
    recursive_merge(test_input_chebyshev_split_2_moments,
                    OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_split_3_moments"),
                                "evolve_moments" => OptionsDict("parallel_pressure" => true)))


"""
Run a sound-wave test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, analytic_frequency, analytic_growth_rate,
                  regression_phi, itime_min=50; args...)
    # by passing keyword arguments to run_test, args becomes a Dict which can be used to
    # update the default inputs

    # Make a copy to make sure nothing modifies the input Dicts defined in this test
    # script.
    input = deepcopy(test_input)

    # Convert keyword arguments to a unique name
    function stringify_arg(key, value)
        if isa(value, AbstractDict)
            return string(string(key)[1], (stringify_arg(k, v) for (k, v) in value)...)
        else
            return string(string(key)[1], value)
        end
    end
    name = input["output"]["run_name"]
    shortname = name
    if length(args) > 0
        name = string(name, "_", (string(k, "-", v, "_") for (k, v) in args)...)
        shortname = string(shortname, "_", (stringify_arg(k, v) for (k, v) in args)...)

        # Remove trailing "_"
        name = chop(name)
    end

    # Provide some progress info
    println("    - testing ", name)

    # Update default inputs with values to be changed
    merge_dict_with_kwargs!(input; args...)
    input["output"]["run_name"] = shortname

    # Suppress console output while running
    phi_fit = undef
    phi = undef
    quietoutput() do
        # run simulation
        run_moment_kinetics(input)
    end

    if global_rank[] == 0
        quietoutput() do

            # Load and analyse output
            #########################

            path = joinpath(realpath(input["output"]["base_directory"]), shortname, shortname)

            # open the netcdf file and give it the handle 'fid'
            fid = open_readonly_output_file(path,"moments")

            # load space-time coordinate data
            z, z_spectral = load_coordinate_data(fid, "z"; ignore_MPI=true)
            r, r_spectral = load_coordinate_data(fid, "r"; ignore_MPI=true)
            n_ion_species, n_neutral_species = load_species_data(fid)
            ntime, time = load_time_data(fid)
            
            # load fields data
            phi_zrt, Er_zrt, Ez_zrt = load_fields_data(fid)

            close(fid)
            
            ir0 = 1 
            
            phi = phi_zrt[:,ir0,:]

            # analyze the fields data
            phi_fldline_avg, delta_phi = analyze_fields_data(phi, ntime, z)

            # use a fit to calculate the damping rate and growth rate of the perturbed
            # electrostatic potential
            itime_max = ntime
            iz0 = cld(z.n, 3)
            shifted_time = mk_zeros(mk_float, ntime)
            @. shifted_time = time - time[itime_min]
            @views phi_fit = fit_delta_phi_mode(shifted_time[itime_min:itime_max], z.grid,
                                                delta_phi[:, itime_min:itime_max])
            ## The following plot code (copied from post_processing.jl) may be helpful for
            ## debugging tests. Uncomment to use, and also uncomment
            ## `using Plots: plot, plot!, gui at the top of the file.
            #L = z.grid[end] - z.grid[begin]
            #fitted_delta_phi =
            #    @. (phi_fit.amplitude0 * cos(2.0 * π * (z.grid[iz0] + phi_fit.offset0) / L)
            #        * exp(phi_fit.growth_rate * shifted_time)
            #        * cos(phi_fit.frequency * shifted_time + phi_fit.phase))
            #@views plot(time, abs.(delta_phi[iz0,:]), xlabel="t*z.L/vti", ylabel="δϕ", yaxis=:log)
            #plot!(time, abs.(fitted_delta_phi))
            #gui()
        end

        # Check the fit errors are not too large, otherwise we are testing junk
        @test phi_fit.amplitude_fit_error < 1.e-1
        @test phi_fit.offset_fit_error < 5.e-6
        @test phi_fit.cosine_fit_error < 5.e-8

        # analytic_frequency and analytic_growth rate are the analytically expected values
        # (from F. Parra's calculation).
        @test isapprox(phi_fit.frequency, analytic_frequency, rtol=analytical_rtol)
        @test isapprox(phi_fit.growth_rate, analytic_growth_rate, rtol=analytical_rtol)

        # Test some values of phi for a regression test, which can use with tighter
        # tolerances than the analytic test.
        @test isapprox(phi[regression_range], regression_phi, rtol=regression_rtol)
    end
end


# run_test_set_* functions call run_test for various parameters, and record the expected
# values to be used for regression_frequency and regression_growth_rate for each
# particular case

function run_test_set_finite_difference()
    #n_i=n_n, T_e=1
    @long run_test(test_input_finite_difference, 2*π*1.4467, -2*π*0.6020,
                   [-0.6941155980262039, -0.6940977032813103, -0.6940759796933667,
                    -0.6940505149819431, -0.6940214119659553, -0.6939887881451088];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*0.0))
    run_test(test_input_finite_difference, 2*π*1.4240, -2*π*0.6379,
             [-0.6941155980262039, -0.6940977032813103, -0.6940759796933667,
              -0.6940505149819431, -0.6940214119659553, -0.6939887881451088])
    @long run_test(test_input_finite_difference, 2*π*0.0, -2*π*0.3235,
                   [-0.6941155980262039, -0.6940977032813103, -0.6940759796933667,
                    -0.6940505149819431, -0.6940214119659553, -0.6939887881451088];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*1.8))
    @long run_test(test_input_finite_difference, 2*π*0.0, -2*π*0.2963,
                   [-0.6941155980262039, -0.6940977032813103, -0.6940759796933667,
                    -0.6940505149819431, -0.6940214119659553, -0.6939887881451088];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i>>n_n T_e=1
    @long run_test(test_input_finite_difference, 2*π*1.4467, -2*π*0.6020,
                   [-0.001068422466592656, -0.001050527721698838, -0.0010288041337549816,
                    -0.0010033394223323698, -0.0009742364063442995,
                    -0.000941612585497573];
                   ion_species_1 = OptionsDict("initial_density" => 0.9999), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.0001))
    @long run_test(test_input_finite_difference, 2*π*1.4467, -2*π*0.6020,
                   [-0.001068422466592656, -0.001050527721698838, -0.0010288041337549816,
                    -0.0010033394223323698, -0.0009742364063442995,
                    -0.000941612585497573]; 
                   ion_species_1 = OptionsDict("initial_density" => 0.9999), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.0001),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i<<n_n T_e=1
    @long run_test(test_input_finite_difference, 2*π*1.3954, -2*π*0.6815,
                   [-9.211308789442441, -9.211290894697548, -9.211269171109604,
                    -9.21124370639818, -9.211214603382192, -9.211181979561346];
                   ion_species_1 = OptionsDict("initial_density" => 0.0001), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.9999))
    @long run_test(test_input_finite_difference, 2*π*0.0, -2*π*0.5112,
                   [-9.211308789442441, -9.211290894697548, -9.211269171109604,
                    -9.21124370639818, -9.211214603382192, -9.211181979561346];
                   ion_species_1 = OptionsDict("initial_density" => 0.0001), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.9999),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i=n_n T_e=0.5
    @long run_test(test_input_finite_difference, 2*π*1.2671, -2*π*0.8033,
                   [-0.34705779901310196, -0.34704885164065513, -0.3470379898466833,
                    -0.3470252574909716, -0.3470107059829777, -0.3469943940725544], 30;
                   composition = OptionsDict("T_e" => 0.5), 
                   timestepping = OptionsDict("nstep" => 1300),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*0.0))
    @long run_test(test_input_finite_difference, 2*π*0.0, -2*π*0.2727,
                   [-0.34705779901310196, -0.34704885164065513, -0.3470379898466833,
                    -0.3470252574909716, -0.3470107059829777, -0.3469943940725544];
                   composition = OptionsDict("T_e" => 0.5),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i=n_n T_e=4
    @long run_test(test_input_finite_difference, 2*π*1.9919, -2*π*0.2491,
                   [-2.7764623921048157, -2.776390813125241, -2.7763039187734666,
                    -2.7762020599277726, -2.7760856478638214, -2.775955152580435];
                   composition = OptionsDict("T_e" => 4.0))
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_finite_difference_split_1_moment()
    #n_i=n_n, T_e=1
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.4467, -2*π*0.6020,
             [-0.6941155980262039, -0.6940977032813103, -0.6940759796933667,
              -0.6940505149819431, -0.6940214119659553, -0.6939887881451088];
             reactions=OptionsDict("charge_exchange_frequency"=>2*π*0.0))
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.4240, -2*π*0.6379,
             [-0.6941155980262039, -0.6940977032813103, -0.6940759796933667,
              -0.6940505149819431, -0.6940214119659553, -0.6939887881451088])
    run_test(test_input_finite_difference_split_1_moment, 2*π*0.0, -2*π*0.3235,
             [-0.6941155980262039, -0.6940977032813103, -0.6940759796933667,
              -0.6940505149819431, -0.6940214119659553, -0.6939887881451088];
             reactions=OptionsDict("charge_exchange_frequency"=>2*π*1.8))
    run_test(test_input_finite_difference_split_1_moment, 2*π*0.0, -2*π*0.2963,
             [-0.6941155980262039, -0.6940977032813103, -0.6940759796933667,
              -0.6940505149819431, -0.6940214119659553, -0.6939887881451088];
             reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i>>n_n T_e=1
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.4467, -2*π*0.6020,
             [-0.0010684224665919893, -0.0010505277216983934, -0.0010288041337547594,
              -0.0010033394223312585, -0.0009742364063434105, -0.0009416125854969064];
             ion_species_1 = OptionsDict("initial_density" => 0.9999), 
             neutral_species_1 = OptionsDict("initial_density" => 0.0001))
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.4467, -2*π*0.6020,
             [-0.0010684224665919893, -0.0010505277216983934, -0.0010288041337547594,
              -0.0010033394223312585, -0.0009742364063434105, -0.0009416125854969064];
             ion_species_1 = OptionsDict("initial_density" => 0.9999), 
             neutral_species_1 = OptionsDict("initial_density" => 0.0001),
             reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i<<n_n T_e=1
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.3954, -2*π*0.6815,
             [-9.211308789442441, -9.211290894697548, -9.211269171109604,
              -9.21124370639818, -9.211214603382192, -9.211181979561346];
             ion_species_1 = OptionsDict("initial_density" => 0.0001), 
             neutral_species_1 = OptionsDict("initial_density" => 0.9999))
    run_test(test_input_finite_difference_split_1_moment, 2*π*0.0, -2*π*0.5112,
             [-9.211308789442441, -9.211290894697548, -9.211269171109604,
              -9.21124370639818, -9.211214603382192, -9.211181979561346];
             ion_species_1 = OptionsDict("initial_density" => 0.0001), 
             neutral_species_1 = OptionsDict("initial_density" => 0.9999),
             reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i=n_n T_e=0.5
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.2671, -2*π*0.8033,
             [-0.34705779901310196, -0.34704885164065513, -0.3470379898466833,
              -0.3470252574909716, -0.3470107059829777, -0.3469943940725544], 30;
             composition = OptionsDict("T_e" => 0.5),
             timestepping = OptionsDict("nstep" => 1300),
             reactions=OptionsDict("charge_exchange_frequency"=>2*π*0.0))
    run_test(test_input_finite_difference_split_1_moment, 2*π*0.0, -2*π*0.2727,
             [-0.34705779901310196, -0.34704885164065513, -0.3470379898466833,
              -0.3470252574909716, -0.3470107059829777, -0.3469943940725544];
             composition = OptionsDict("T_e" => 0.5),
             reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i=n_n T_e=4
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.9919, -2*π*0.2491,
             [-2.7764623921048157, -2.776390813125241, -2.7763039187734666,
              -2.7762020599277726, -2.7760856478638214, -2.775955152580435];
              composition = OptionsDict("T_e" => 4.0))
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_finite_difference_split_2_moments()
    #n_i=n_n, T_e=1
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.4467, -2*π*0.6020,
             [-0.6941155980262039, -0.6940977032813103, -0.6940759796933667,
              -0.6940505149819431, -0.6940214119659553, -0.6939887881451088];
             reactions=OptionsDict("charge_exchange_frequency"=>2*π*0.0))
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.4240, -2*π*0.6379,
             [-0.6941155980262039, -0.6940977032813103, -0.6940759796933667,
              -0.6940505149819431, -0.6940214119659553, -0.6939887881451088])
    run_test(test_input_finite_difference_split_2_moments, 2*π*0.0, -2*π*0.3235,
             [-0.6941155980262039, -0.6940977032813103, -0.6940759796933667,
              -0.6940505149819431, -0.6940214119659553, -0.6939887881451088];
             reactions=OptionsDict("charge_exchange_frequency"=>2*π*1.8))
    run_test(test_input_finite_difference_split_2_moments, 2*π*0.0, -2*π*0.2963,
             [-0.6941155980262039, -0.6940977032813103, -0.6940759796933667,
              -0.6940505149819431, -0.6940214119659553, -0.6939887881451088];
             reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i>>n_n T_e=1
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.4467, -2*π*0.6020,
             [-0.0010684224665919893, -0.0010505277216983934, -0.0010288041337547594,
              -0.0010033394223312585, -0.0009742364063434105, -0.0009416125854969064];
             ion_species_1 = OptionsDict("initial_density" => 0.9999), 
             neutral_species_1 = OptionsDict("initial_density" => 0.0001))
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.4467, -2*π*0.6020,
             [-0.0010684224665919893, -0.0010505277216983934, -0.0010288041337547594,
              -0.0010033394223312585, -0.0009742364063434105, -0.0009416125854969064];
             ion_species_1 = OptionsDict("initial_density" => 0.9999), 
             neutral_species_1 = OptionsDict("initial_density" => 0.0001),
             reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i<<n_n T_e=1
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.3954, -2*π*0.6815,
             [-9.211308789442441, -9.211290894697548, -9.211269171109604,
              -9.21124370639818, -9.211214603382192, -9.211181979561346];
             ion_species_1 = OptionsDict("initial_density" => 0.0001), 
             neutral_species_1 = OptionsDict("initial_density" => 0.9999))
    run_test(test_input_finite_difference_split_2_moments, 2*π*0.0, -2*π*0.5112,
             [-9.211308789442441, -9.211290894697548, -9.211269171109604,
              -9.21124370639818, -9.211214603382192, -9.211181979561346];
             ion_species_1 = OptionsDict("initial_density" => 0.0001), 
             neutral_species_1 = OptionsDict("initial_density" => 0.9999),
             reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i=n_n T_e=0.5
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.2671, -2*π*0.8033,
             [-0.34706673733456106, -0.3470627566790802, -0.3470579059173919,
              -0.347052193699157, -0.34704563020982493, -0.3470382271523149], 30;
             composition = OptionsDict("T_e" => 0.5),
             timestepping = OptionsDict("nstep" => 1300), z = OptionsDict("ngrid" => 150),
             reactions=OptionsDict("charge_exchange_frequency"=>2*π*0.0))
    run_test(test_input_finite_difference_split_2_moments, 2*π*0.0, -2*π*0.2727,
             [-0.34705779901310196, -0.34704885164065513, -0.3470379898466833,
              -0.3470252574909716, -0.3470107059829777, -0.3469943940725544];
             composition = OptionsDict("T_e" => 0.5),
             reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i=n_n T_e=4
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.9919, -2*π*0.2491,
             [-2.7764623921048157, -2.776390813125241, -2.7763039187734666,
              -2.7762020599277726, -2.7760856478638214, -2.775955152580435];
              composition = OptionsDict("T_e" => 4.0))
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_finite_difference_split_3_moments()
    #n_i=n_n, T_e=1
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*1.4467,
                   -2*π*0.6020, [-0.6941155980262039, -0.6940977032813103,
                                 -0.6940759796933667, -0.6940505149819431,
                                 -0.6940214119659553, -0.6939887881451088];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*0.0))
    run_test(test_input_finite_difference_split_3_moments, 2*π*1.4240, -2*π*0.6379,
             [-0.6941155980262039, -0.6940977032813103, -0.6940759796933667,
              -0.6940505149819431, -0.6940214119659553, -0.6939887881451088])
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*0.0, -2*π*0.3235,
                   [-0.6941155980262039, -0.6940977032813103, -0.6940759796933667,
                    -0.6940505149819431, -0.6940214119659553, -0.6939887881451088];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*1.8))
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*0.0, -2*π*0.2963,
                   [-0.6941155980262039, -0.6940977032813103, -0.6940759796933667,
                    -0.6940505149819431, -0.6940214119659553, -0.6939887881451088];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i>>n_n T_e=1
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*1.4467,
                   -2*π*0.6020, [-0.0010684224665919893, -0.0010505277216983934,
                                 -0.0010288041337547594, -0.0010033394223312585,
                                 -0.0009742364063434105, -0.0009416125854969064];
                   ion_species_1 = OptionsDict("initial_density" => 0.9999), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.0001))
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*1.4467,
                   -2*π*0.6020, [-0.0010684224665919893, -0.0010505277216983934,
                                 -0.0010288041337547594, -0.0010033394223312585,
                                 -0.0009742364063434105, -0.0009416125854969064];
                   ion_species_1 = OptionsDict("initial_density" => 0.9999), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.0001),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i<<n_n T_e=1
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*1.3954,
                   -2*π*0.6815, [-9.211308789442441, -9.211290894697548,
                                 -9.211269171109604, -9.21124370639818,
                                 -9.211214603382192, -9.211181979561346];
                   ion_species_1 = OptionsDict("initial_density" => 0.0001), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.9999))
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*0.0, -2*π*0.5112,
                   [-9.211308789442441, -9.211290894697548, -9.211269171109604,
                    -9.21124370639818, -9.211214603382192, -9.211181979561346];
                   ion_species_1 = OptionsDict("initial_density" => 0.0001), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.9999),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i=n_n T_e=0.5
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*1.2671,
                   -2*π*0.8033, [-0.34705779901310196, -0.34704885164065513,
                                 -0.3470379898466833, -0.3470252574909716,
                                 -0.3470107059829777, -0.3469943940725544], 30;
                   composition = OptionsDict("T_e" => 0.5),
                   timestepping = OptionsDict("nstep" => 1300),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*0.0))
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*0.0, -2*π*0.2727,
                   [-0.34705779901310196, -0.34704885164065513, -0.3470379898466833,
                    -0.3470252574909716, -0.3470107059829777, -0.3469943940725544];
                   composition = OptionsDict("T_e" => 0.5),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i=n_n T_e=4
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*1.9919,
                   -2*π*0.2491, [-2.7764623921048157, -2.776390813125241,
                                 -2.7763039187734666, -2.7762020599277726,
                                 -2.7760856478638214, -2.775955152580435];
                                 composition = OptionsDict("T_e" => 4.0))
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_chebyshev()
    #n_i=n_n, T_e=1
    @long run_test(test_input_chebyshev, 2*π*1.4467, -2*π*0.6020,
                   [-0.6931471805599453, -0.6925817758085663, -0.6922515628093567,
                    -0.6921548130694323, -0.6921476802268619, -0.6921548130694323];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*0.0))
    run_test(test_input_chebyshev, 2*π*1.4240, -2*π*0.6379,
             [-0.6931471805599453, -0.6925817758085663, -0.6922515628093567,
              -0.6921548130694323, -0.6921476802268619, -0.6921548130694323])
    @long run_test(test_input_chebyshev, 2*π*0.0, -2*π*0.3235,
                   [-0.6931471805599453, -0.6925817758085663, -0.6922515628093567,
                    -0.6921548130694323, -0.6921476802268619, -0.6921548130694323];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*1.8))
    @long run_test(test_input_chebyshev, 2*π*0.0, -2*π*0.2963,
                   [-0.6931471805599453, -0.6925817758085663, -0.6922515628093567,
                    -0.6921548130694323, -0.6921476802268619, -0.6921548130694323];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i>>n_n T_e=1
    @long run_test(test_input_chebyshev, 2*π*1.4467, -2*π*0.6020,
                   [-0.00010000500033334732, 0.0004653997510461739, 0.0007956127502558478,
                    0.0008923624901804128, 0.0008994953327500175, 0.0008923624901804128];
                   ion_species_1 = OptionsDict("initial_density" => 0.9999), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.0001))
    @long run_test(test_input_chebyshev, 2*π*1.4467, -2*π*0.6020,
                   [-0.00010000500033334732, 0.0004653997510461739, 0.0007956127502558478,
                    0.0008923624901804128, 0.0008994953327500175, 0.0008923624901804128];
                   ion_species_1 = OptionsDict("initial_density" => 0.9999), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.0001),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i<<n_n T_e=1
    @long run_test(test_input_chebyshev, 2*π*1.3954, -2*π*0.6815,
                   [-9.210340371976182, -9.209774967224805, -9.209444754225593,
                    -9.209348004485669, -9.2093408716431, -9.209348004485669];
                   ion_species_1 = OptionsDict("initial_density" => 0.0001), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.9999))
    @long run_test(test_input_chebyshev, 2*π*0.0, -2*π*0.5112,
                   [-9.210340371976182, -9.209774967224805, -9.209444754225593,
                    -9.209348004485669, -9.2093408716431, -9.209348004485669];
                   ion_species_1 = OptionsDict("initial_density" => 0.0001), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.9999),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i=n_n T_e=0.5
    @long run_test(test_input_chebyshev, 2*π*1.2671, -2*π*0.8033,
                   [-0.34657359027997264, -0.34629088790428314, -0.34612578140467837,
                    -0.34607740653471614, -0.34607384011343095, -0.34607740653471614],
                   30; composition = OptionsDict("T_e" => 0.5),
                   timestepping = OptionsDict("nstep" => 1300),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*0.0))
    @long run_test(test_input_chebyshev, 2*π*0.0, -2*π*0.2727,
                   [-0.34657359027997264, -0.34629088790428314, -0.34612578140467837,
                    -0.34607740653471614, -0.34607384011343095, -0.34607740653471614];
                   composition = OptionsDict("T_e" => 0.5),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i=n_n T_e=4
    @long run_test(test_input_chebyshev, 2*π*1.9919, -2*π*0.2491,
                   [-2.772588722239781, -2.770327103234265, -2.769006251237427,
                    -2.768619252277729, -2.7685907209074476, -2.768619252277729];
                   composition = OptionsDict("T_e" => 4.0))
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_chebyshev_split_1_moment()
    #n_i=n_n, T_e=1
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*1.4467, -2*π*0.6020,
                   [-0.6931471805599453, -0.6925817758085663, -0.6922515628093567,
                    -0.6921548130694323, -0.6921476802268619, -0.6921548130694323];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*0.0))
    run_test(test_input_chebyshev_split_1_moment, 2*π*1.4240, -2*π*0.6379,
             [-0.6931471805599453, -0.6925817758085663, -0.6922515628093567,
              -0.6921548130694323, -0.6921476802268619, -0.6921548130694323])
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*0.0, -2*π*0.3235,
                   [-0.6931471805599453, -0.6925817758085663, -0.6922515628093567,
                    -0.6921548130694323, -0.6921476802268619, -0.6921548130694323];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*1.8))
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*0.0, -2*π*0.2963,
                   [-0.6931471805599453, -0.6925817758085663, -0.6922515628093567,
                    -0.6921548130694323, -0.6921476802268619, -0.6921548130694323];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i>>n_n T_e=1
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*1.4467, -2*π*0.6020,
                   [-0.00010000500033334732, 0.00046539975104573, 0.0007956127502551822,
                    0.0008923624901797472, 0.0008994953327500175,
                    0.0008923624901797472];
                   ion_species_1 = OptionsDict("initial_density" => 0.9999), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.0001))
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*1.4467, -2*π*0.6020,
                   [-0.00010000500033334732, 0.00046539975104573, 0.0007956127502551822,
                    0.0008923624901797472, 0.0008994953327500175,
                    0.0008923624901797472];
                   ion_species_1 = OptionsDict("initial_density" => 0.9999), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.0001),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i<<n_n T_e=1
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*1.3954, -2*π*0.6815,
                   [-9.210340371976182, -9.209774967224805, -9.209444754225593,
                    -9.209348004485669, -9.2093408716431, -9.209348004485669];
                   ion_species_1 = OptionsDict("initial_density" => 0.0001), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.9999))
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*0.0, -2*π*0.5112,
                   [-9.210340371976182, -9.209774967224805, -9.209444754225593,
                    -9.209348004485669, -9.2093408716431, -9.209348004485669];
                   ion_species_1 = OptionsDict("initial_density" => 0.0001), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.9999),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i=n_n T_e=0.5
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*1.2671, -2*π*0.8033,
                   [-0.34657359027997264, -0.34629088790428314, -0.34612578140467837,
                    -0.34607740653471614, -0.34607384011343095, -0.34607740653471614],
                   30; composition = OptionsDict("T_e" => 0.5),
                   timestepping = OptionsDict("nstep" => 1300),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*0.0))
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*0.0, -2*π*0.2727,
                   [-0.34657359027997264, -0.34629088790428314, -0.34612578140467837,
                    -0.34607740653471614, -0.34607384011343095, -0.34607740653471614];
                   composition = OptionsDict("T_e" => 0.5),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i=n_n T_e=4
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*1.9919, -2*π*0.2491,
                   [-2.772588722239781, -2.770327103234265, -2.769006251237427,
                    -2.768619252277729, -2.7685907209074476, -2.768619252277729];
                   composition = OptionsDict("T_e" => 4.0))
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_chebyshev_split_2_moments()
    #n_i=n_n, T_e=1
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*1.4467, -2*π*0.6020,
                   [-0.6931471805599453, -0.6925817758085663, -0.6922515628093567,
                    -0.6921548130694323, -0.6921476802268619, -0.6921548130694323];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*0.0))
    run_test(test_input_chebyshev_split_2_moments, 2*π*1.4240, -2*π*0.6379,
             [-0.6931471805599453, -0.6925817758085663, -0.6922515628093567,
              -0.6921548130694323, -0.6921476802268619, -0.6921548130694323])
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*0.0, -2*π*0.3235,
                   [-0.6931471805599453, -0.6925817758085663, -0.6922515628093567,
                    -0.6921548130694323, -0.6921476802268619, -0.6921548130694323];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*1.8))
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*0.0, -2*π*0.2963,
                   [-0.6931471805599453, -0.6925817758085663, -0.6922515628093567,
                    -0.6921548130694323, -0.6921476802268619, -0.6921548130694323];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i>>n_n T_e=1
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*1.4467, -2*π*0.6020,
                   [-0.00010000500033334732, 0.00046539975104573, 0.0007956127502551822,
                    0.0008923624901797472, 0.0008994953327500175,
                    0.0008923624901797472]; 
                   ion_species_1 = OptionsDict("initial_density" => 0.9999), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.0001))
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*1.4467, -2*π*0.6020,
                   [-0.00010000500033334732, 0.00046539975104573, 0.0007956127502551822,
                    0.0008923624901797472, 0.0008994953327500175,
                    0.0008923624901797472];
                   ion_species_1 = OptionsDict("initial_density" => 0.9999), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.0001),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i<<n_n T_e=1
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*1.3954, -2*π*0.6815,
                   [-9.210340371976182, -9.209774967224805, -9.209444754225593,
                    -9.209348004485669, -9.2093408716431, -9.209348004485669];
                   ion_species_1 = OptionsDict("initial_density" => 0.0001), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.9999))
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*0.0, -2*π*0.5112,
                   [-9.210340371976182, -9.209774967224805, -9.209444754225593,
                    -9.209348004485669, -9.2093408716431, -9.209348004485669];
                   ion_species_1 = OptionsDict("initial_density" => 0.0001), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.9999),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i=n_n T_e=0.5
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*1.2671, -2*π*0.8033,
                   [-0.34657359027997264, -0.34629088790428314, -0.34612578140467837,
                    -0.34607740653471614, -0.34607384011343095, -0.34607740653471614],
                   40; composition = OptionsDict("T_e" => 0.5),
                   timestepping = OptionsDict("nstep" => 1300, "nwrite" => 10),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*0.0))
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*0.0, -2*π*0.2727,
                   [-0.34657359027997264, -0.34629088790428314, -0.34612578140467837,
                    -0.34607740653471614, -0.34607384011343095, -0.34607740653471614];
                   composition = OptionsDict("T_e" => 0.5),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i=n_n T_e=4
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*1.9919, -2*π*0.2491,
                   [-2.772588722239781, -2.770327103234265, -2.769006251237427,
                    -2.768619252277729, -2.7685907209074476, -2.768619252277729];
                   composition = OptionsDict("T_e" => 4.0))
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_chebyshev_split_3_moments()
    #n_i=n_n, T_e=1
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*1.4467, -2*π*0.6020,
                   [-0.6931471805599453, -0.6925817758085663, -0.6922515628093567,
                    -0.6921548130694323, -0.6921476802268619, -0.6921548130694323];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*0.0))
    run_test(test_input_chebyshev_split_3_moments, 2*π*1.4240, -2*π*0.6379,
             [-0.6931471805599453, -0.6925817758085663, -0.6922515628093567,
              -0.6921548130694323, -0.6921476802268619, -0.6921548130694323])
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*0.0, -2*π*0.3235,
                   [-0.6931471805599453, -0.6925817758085663, -0.6922515628093567,
                    -0.6921548130694323, -0.6921476802268619, -0.6921548130694323];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*1.8))
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*0.0, -2*π*0.2963,
                   [-0.6931471805599453, -0.6925817758085663, -0.6922515628093567,
                    -0.6921548130694323, -0.6921476802268619, -0.6921548130694323];
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i>>n_n T_e=1
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*1.4467, -2*π*0.6020,
                   [-0.00010000500033334732, 0.00046539975104573, 0.0007956127502551822,
                    0.0008923624901797472, 0.0008994953327500175,
                    0.0008923624901797472];
                   ion_species_1 = OptionsDict("initial_density" => 0.9999), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.0001))
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*1.4467, -2*π*0.6020,
                   [-0.00010000500033334732, 0.00046539975104573, 0.0007956127502551822,
                    0.0008923624901797472, 0.0008994953327500175,
                    0.0008923624901797472];
                   ion_species_1 = OptionsDict("initial_density" => 0.9999), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.0001),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i<<n_n T_e=1
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*1.3954, -2*π*0.6815,
                   [-9.210340371976182, -9.209774967224805, -9.209444754225593,
                    -9.209348004485669, -9.2093408716431, -9.209348004485669];
                   ion_species_1 = OptionsDict("initial_density" => 0.0001), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.9999))
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*0.0, -2*π*0.5112,
                   [-9.210340371976182, -9.209774967224805, -9.209444754225593,
                    -9.209348004485669, -9.2093408716431, -9.209348004485669];
                   ion_species_1 = OptionsDict("initial_density" => 0.0001), 
                   neutral_species_1 = OptionsDict("initial_density" => 0.9999),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i=n_n T_e=0.5
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*1.2671, -2*π*0.8033,
                   [-0.34657359027997264, -0.34629088790428314, -0.34612578140467837,
                    -0.34607740653471614, -0.34607384011343095, -0.34607740653471614],
                   80; composition = OptionsDict("T_e" => 0.5),
                   timestepping = OptionsDict("nstep" => 1300, "nwrite" => 5),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*0.0))
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*0.0, -2*π*0.2727,
                   [-0.34657359027997264, -0.34629088790428314, -0.34612578140467837,
                    -0.34607740653471614, -0.34607384011343095, -0.34607740653471614];
                   composition = OptionsDict("T_e" => 0.5),
                   reactions=OptionsDict("charge_exchange_frequency"=>2*π*2.0))

    # n_i=n_n T_e=4
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*1.9919, -2*π*0.2491,
                   [-2.772588722239781, -2.770327103234265, -2.769006251237427,
                    -2.768619252277729, -2.7685907209074476, -2.768619252277729];
                   composition = OptionsDict("T_e" => 4.0))
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function runtests()
    # Create a temporary directory for test output
    test_output_directory = get_MPI_tempdir()

    @testset "sound wave" verbose=use_verbose begin
        println("sound wave tests")

        @testset "finite difference" begin
            test_input_finite_difference["output"]["base_directory"] = test_output_directory
            run_test_set_finite_difference()

            test_input_finite_difference_split_1_moment["output"]["base_directory"] = test_output_directory
            @long run_test_set_finite_difference_split_1_moment()

            test_input_finite_difference_split_2_moments["output"]["base_directory"] = test_output_directory
            @long run_test_set_finite_difference_split_2_moments()

            test_input_finite_difference_split_3_moments["output"]["base_directory"] = test_output_directory
            run_test_set_finite_difference_split_3_moments()
        end

        @testset "Chebyshev" begin
            test_input_chebyshev["output"]["base_directory"] = test_output_directory
            run_test_set_chebyshev()

            test_input_chebyshev_split_1_moment["output"]["base_directory"] = test_output_directory
            run_test_set_chebyshev_split_1_moment()

            test_input_chebyshev_split_2_moments["output"]["base_directory"] = test_output_directory
            run_test_set_chebyshev_split_2_moments()

            test_input_chebyshev_split_3_moments["output"]["base_directory"] = test_output_directory
            run_test_set_chebyshev_split_3_moments()
        end
    end

    if global_rank[] == 0
        # Delete output directory to avoid using too much disk space
        rm(realpath(test_output_directory); recursive=true)
    end
end

end # SoundWaveTests


using .SoundWaveTests

SoundWaveTests.runtests()
