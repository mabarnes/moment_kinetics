include("setup.jl")

using Base.Filesystem: tempname
using TimerOutputs
#using Plots: plot, plot!, gui

using moment_kinetics.array_allocation: allocate_float
using moment_kinetics.load_data: open_netcdf_file
using moment_kinetics.load_data: load_coordinate_data, load_fields_data
using moment_kinetics.analysis: analyze_fields_data
using moment_kinetics.post_processing: fit_delta_phi_mode

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

# default inputs for tests
test_input_finite_difference = Dict("n_ion_species" => 1,
                                    "n_neutral_species" => 1,
                                    "boltzmann_electron_response" => true,
                                    "run_name" => "finite_difference",
                                    "base_directory" => test_output_directory,
                                    "evolve_moments_density" => false,
                                    "evolve_moments_parallel_flow" => false,
                                    "evolve_moments_parallel_pressure" => false,
                                    "evolve_moments_conservation" => true,
                                    "T_e" => 1.0,
                                    "initial_density1" => 0.5,
                                    "initial_temperature1" => 1.0,
                                    "initial_density2" => 0.5,
                                    "initial_temperature2" => 1.0,
                                    "z_IC_amplitude1" => 0.001,
                                    "charge_exchange_frequency" => 0.0,
                                    "nstep" => 1500,
                                    "dt" => 0.002,
                                    "nwrite" => 20,
                                    "use_semi_lagrange" => false,
                                    "n_rk_stages" => 4,
                                    "split_operators" => false,
                                    "z_ngrid" => 200,
                                    "z_nelement" => 1,
                                    "z_discretization" => "finite_difference",
                                    "vpa_ngrid" => 200,
                                    "vpa_nelement" => 1,
                                    "vpa_L" => 8.0,
                                    "vpa_bc" => "periodic",
                                    "vpa_discretization" => "finite_difference")

test_input_finite_difference_split_1_moment =
    merge(test_input_finite_difference,
          Dict("run_name" => "finite_difference_split_1_moment",
               "evolve_moments_density" => true))

test_input_finite_difference_split_2_moments =
    merge(test_input_finite_difference_split_1_moment,
          Dict("run_name" => "finite_difference_split_2_moments",
               "evolve_moments_parallel_flow" => true))

test_input_finite_difference_split_3_moments =
    merge(test_input_finite_difference_split_2_moments,
          Dict("run_name" => "finite_difference_split_3_moments",
               "evolve_moments_parallel_pressure" => true))

test_input_chebyshev = merge(test_input_finite_difference,
                             Dict("run_name" => "chebyshev_pseudospectral",
                                  "z_discretization" => "chebyshev_pseudospectral",
                                  "z_ngrid" => 9,
                                  "z_nelement" => 2,
                                  "vpa_discretization" => "chebyshev_pseudospectral",
                                  "vpa_ngrid" => 17,
                                  "vpa_nelement" => 8))

test_input_chebyshev_split_1_moment =
    merge(test_input_chebyshev,
          Dict("run_name" => "chebyshev_pseudospectral_split_1_moment",
               "evolve_moments_density" => true))

test_input_chebyshev_split_2_moments =
    merge(test_input_chebyshev_split_1_moment,
          Dict("run_name" => "chebyshev_pseudospectral_split_2_moments",
               "evolve_moments_parallel_flow" => true))

test_input_chebyshev_split_3_moments =
    merge(test_input_chebyshev_split_2_moments,
          Dict("run_name" => "chebyshev_pseudospectral_split_3_moments",
               "evolve_moments_parallel_pressure" => true))


# Not actually used in the tests, but needed for first argument of run_moment_kinetics
to = TimerOutput()

"""
Run a sound-wave test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, analytic_frequency, analytic_growth_rate,
                  regression_frequency, regression_growth_rate, itime_min=50;
                  args...)
    # by passing keyword arguments to run_test, args becomes a Dict which can be used to
    # update the default inputs

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
    phi_fit = undef
    quietoutput() do
        # run simulation
        run_moment_kinetics(to, input)

        # Load and analyse output
        #########################

        path = joinpath(realpath(input["base_directory"]), name, name)

        # open the netcdf file and give it the handle 'fid'
        fid = open_netcdf_file(path)

        # load space-time coordinate data
        nz, z, z_wgts, Lz, nvpa, vpa, vpa_wgts, ntime, time = load_coordinate_data(fid)

        # load fields data
        phi = load_fields_data(fid)

        # analyze the fields data
        phi_fldline_avg, delta_phi = analyze_fields_data(phi, ntime, nz, z_wgts, Lz)

        # use a fit to calculate the damping rate and growth rate of the perturbed
        # electrostatic potential
        itime_max = ntime
        iz0 = cld(nz, 3)
        shifted_time = allocate_float(ntime)
        @. shifted_time = time - time[itime_min]
        @views phi_fit = fit_delta_phi_mode(shifted_time[itime_min:itime_max], z,
                                            delta_phi[:, itime_min:itime_max])
        ## The following plot code (copied from post_processing.jl) may be helpful for
        ## debugging tests. Uncomment to use, and also uncomment
        ## `using Plots: plot, plot!, gui at the top of the file.
        #L = z[end] - z[begin]
        #fitted_delta_phi =
        #    @. (phi_fit.amplitude0 * cos(2.0 * π * (z[iz0] + phi_fit.offset0) / L)
        #        * exp(phi_fit.growth_rate * shifted_time)
        #        * cos(phi_fit.frequency * shifted_time + phi_fit.phase))
        #@views plot(time, abs.(delta_phi[iz0,:]), xlabel="t*Lz/vti", ylabel="δϕ", yaxis=:log)
        #plot!(time, abs.(fitted_delta_phi))
        #gui()
    end

    # Check the fit errors are not too large, otherwise we are testing junk
    @test phi_fit.fit_error < 2.e-2
    @test phi_fit.offset_fit_error < 5.e-6
    @test phi_fit.cosine_fit_error < 5.e-8

    # analytic_frequency and analytic_growth rate are the analytically expected values
    # (from F. Parra's calculation).
    @test isapprox(phi_fit.frequency, analytic_frequency, rtol=3.e-2)
    @test isapprox(phi_fit.growth_rate, analytic_growth_rate, rtol=3.e-2)

    # regression_frequency and regression_growth_rate are saved numerical values, which
    # are tested with tighter tolerances than the analytic values.
    @test isapprox(phi_fit.frequency, regression_frequency, rtol=1.e-9)
    @test isapprox(phi_fit.growth_rate, regression_growth_rate, rtol=1.e-9)
end


# run_test_set_* functions call run_test for various parameters, and record the expected
# values to be used for regression_frequency and regression_growth_rate for each
# particular case

function run_test_set_finite_difference()
    #n_i=n_n, T_e=1
    run_test(test_input_finite_difference, 2*π*1.4467, -2*π*0.6020, 9.091646784462293,
             -3.7772056653373385)
    run_test(test_input_finite_difference, 2*π*1.4240, -2*π*0.6379, 8.954132663749439,
             -4.000929213530583; charge_exchange_frequency=2*π*0.1)
    run_test(test_input_finite_difference, 2*π*0.0, -2*π*0.3235, 0.0,
             -2.059288429238561; charge_exchange_frequency=2*π*1.8)
    run_test(test_input_finite_difference, 2*π*0.0, -2*π*0.2963, 0.0,
             -1.8818455584985407; charge_exchange_frequency=2*π*2.0)

    # n_i>>n_n T_e=1
    run_test(test_input_finite_difference, 2*π*1.4467, -2*π*0.6020, 9.091621526115807,
             -3.777246851378533; initial_density1=0.9999, initial_density2=0.0001,
             charge_exchange_frequency=2*π*0.1)
    run_test(test_input_finite_difference, 2*π*1.4467, -2*π*0.6020, 9.088756546426614,
             -3.786593990110537; initial_density1=0.9999, initial_density2=0.0001,
             charge_exchange_frequency=2*π*2.0)

    # n_i<<n_n T_e=1
    run_test(test_input_finite_difference, 2*π*1.3954, -2*π*0.6815, 8.786954560354536,
             -4.274831939571534; initial_density1=0.0001, initial_density2=0.9999,
             charge_exchange_frequency=2*π*0.1)
    run_test(test_input_finite_difference, 2*π*0.0, -2*π*0.5112, 0.0,
             -3.211984675439382; initial_density1=0.0001, initial_density2=0.9999,
             charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=0.5
    run_test(test_input_finite_difference, 2*π*1.2671, -2*π*0.8033, 7.966986900120581,
             -5.027477871263039, 30; T_e=0.5, nstep=1300,
             charge_exchange_frequency=2*π*0.0)
    run_test(test_input_finite_difference, 2*π*0.0, -2*π*0.2727, 0.0,
             -1.711457812071076; T_e=0.5, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=4
    run_test(test_input_finite_difference, 2*π*1.9919, -2*π*0.2491, 12.51565738918366,
             -1.5654422948824103; T_e=4.0, charge_exchange_frequency=2*π*0.1)
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_finite_difference_split_1_moment()
    #n_i=n_n, T_e=1
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.4467, -2*π*0.6020,
             9.091493524327964, -3.777665867614049)
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.4240, -2*π*0.6379,
             8.953508560280834, -4.001644687690077; charge_exchange_frequency=2*π*0.1)
    run_test(test_input_finite_difference_split_1_moment, 2*π*0.0, -2*π*0.3235, 0.0,
             -2.05926190520108; charge_exchange_frequency=2*π*1.8)
    run_test(test_input_finite_difference_split_1_moment, 2*π*0.0, -2*π*0.2963, 0.0,
             -1.8818262558642804; charge_exchange_frequency=2*π*2.0)

    # n_i>>n_n T_e=1
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.4467, -2*π*0.6020,
             9.09146802461751, -3.777706892295241; initial_density1=0.9999,
             initial_density2=0.0001, charge_exchange_frequency=2*π*0.1)
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.4467, -2*π*0.6020,
             9.088600096940231, -3.7870589478251206; initial_density1=0.9999,
             initial_density2=0.0001, charge_exchange_frequency=2*π*2.0)

    # n_i<<n_n T_e=1
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.3954, -2*π*0.6815,
             8.785064220517988, -4.275580111640141; initial_density1=0.0001,
             initial_density2=0.9999, charge_exchange_frequency=2*π*0.1)
    run_test(test_input_finite_difference_split_1_moment, 2*π*0.0, -2*π*0.5112, 0.0,
             -3.2119286548981716; initial_density1=0.0001, initial_density2=0.9999,
             charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=0.5
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.2671, -2*π*0.8033,
             7.966196877373141, -5.02965525662319, 30; T_e=0.5, nstep=1300,
             charge_exchange_frequency=2*π*0.0)
    run_test(test_input_finite_difference_split_1_moment, 2*π*0.0, -2*π*0.2727, 0.0,
             -1.7114385998755213; T_e=0.5, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=4
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.9919, -2*π*0.2491,
             12.515654502100627, -1.5654340021890731; T_e=4.0,
             charge_exchange_frequency=2*π*0.1)
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_finite_difference_split_2_moments()
    #n_i=n_n, T_e=1
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.4467, -2*π*0.6020,
             9.097763659968544, -3.7580080201111734)
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.4240, -2*π*0.6379,
             8.978198943638452, -3.975337092289034; charge_exchange_frequency=2*π*0.1)
    run_test(test_input_finite_difference_split_2_moments, 2*π*0.0, -2*π*0.3235, 0.0,
             -2.0539272618322517; charge_exchange_frequency=2*π*1.8)
    run_test(test_input_finite_difference_split_2_moments, 2*π*0.0, -2*π*0.2963, 0.0,
             -1.8772110217069822; charge_exchange_frequency=2*π*2.0)

    # n_i>>n_n T_e=1
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.4467, -2*π*0.6020,
             9.097740524923225, -3.7580476472383055; initial_density1=0.9999,
             initial_density2=0.0001, charge_exchange_frequency=2*π*0.1)
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.4467, -2*π*0.6020,
             9.094955859467234, -3.7672675386429764; initial_density1=0.9999,
             initial_density2=0.0001, charge_exchange_frequency=2*π*2.0)

    # n_i<<n_n T_e=1
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.3954, -2*π*0.6815,
             8.853475538840721, -4.250291948484979; initial_density1=0.0001,
             initial_density2=0.9999, charge_exchange_frequency=2*π*0.1)
    run_test(test_input_finite_difference_split_2_moments, 2*π*0.0, -2*π*0.5112, 0.0,
             -3.2023178398409193; initial_density1=0.0001, initial_density2=0.9999,
             charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=0.5
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.2671, -2*π*0.8033,
             7.983874107794549, -4.970861124146282, 30; T_e=0.5, nstep=1300,
             charge_exchange_frequency=2*π*0.0)
    run_test(test_input_finite_difference_split_2_moments, 2*π*0.0, -2*π*0.2727, 0.0,
             -1.7073623296055411; T_e=0.5, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=4
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.9919, -2*π*0.2491,
             12.516541634192802, -1.5656433826739014; T_e=4.0,
             charge_exchange_frequency=2*π*0.1)
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_finite_difference_split_3_moments()
    #n_i=n_n, T_e=1
    run_test(test_input_finite_difference_split_3_moments, 2*π*1.4467, -2*π*0.6020,
             9.093629275008885, -3.7729070085467162)
    run_test(test_input_finite_difference_split_3_moments, 2*π*1.4240, -2*π*0.6379,
             8.960453644497454, -3.9998305928623856; charge_exchange_frequency=2*π*0.1)
    run_test(test_input_finite_difference_split_3_moments, 2*π*0.0, -2*π*0.3235, 0.0,
             -2.0521682863990653; charge_exchange_frequency=2*π*1.8)
    run_test(test_input_finite_difference_split_3_moments, 2*π*0.0, -2*π*0.2963, 0.0,
             -1.8756529591893045; charge_exchange_frequency=2*π*2.0)

    # n_i>>n_n T_e=1
    run_test(test_input_finite_difference_split_3_moments, 2*π*1.4467, -2*π*0.6020,
             9.0936044015915, -3.7729484978962944; initial_density1=0.9999,
             initial_density2=0.0001, charge_exchange_frequency=2*π*0.1)
    run_test(test_input_finite_difference_split_3_moments, 2*π*1.4467, -2*π*0.6020,
             9.09073992320698, -3.7823446011594464; initial_density1=0.9999,
             initial_density2=0.0001, charge_exchange_frequency=2*π*2.0)

    # n_i<<n_n T_e=1
    run_test(test_input_finite_difference_split_3_moments, 2*π*1.3954, -2*π*0.6815,
             8.80311956904467, -4.28207267141589; initial_density1=0.0001,
             initial_density2=0.9999, charge_exchange_frequency=2*π*0.1)
    run_test(test_input_finite_difference_split_3_moments, 2*π*0.0, -2*π*0.5112, 0.0,
             -3.198045285969458; initial_density1=0.0001, initial_density2=0.9999,
             charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=0.5
    run_test(test_input_finite_difference_split_3_moments, 2*π*1.2671, -2*π*0.8033,
             7.9709416406027405, -5.017098763695644, 30; T_e=0.5, nstep=1300,
             charge_exchange_frequency=2*π*0.0)
    run_test(test_input_finite_difference_split_3_moments, 2*π*0.0, -2*π*0.2727, 0.0,
             -1.7058801097683656; T_e=0.5, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=4
    run_test(test_input_finite_difference_split_3_moments, 2*π*1.9919, -2*π*0.2491,
             12.517160609534336, -1.5665029660914167; T_e=4.0,
             charge_exchange_frequency=2*π*0.1)
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_chebyshev()
    #n_i=n_n, T_e=1
    run_test(test_input_chebyshev, 2*π*1.4467, -2*π*0.6020, 9.085815883255613,
             -3.794758980708628)
    run_test(test_input_chebyshev, 2*π*1.4240, -2*π*0.6379, 8.933662137891284,
             -4.025210168129686; charge_exchange_frequency=2*π*0.1)
    run_test(test_input_chebyshev, 2*π*0.0, -2*π*0.3235, 0.0, -2.058256214667329;
             charge_exchange_frequency=2*π*1.8)
    run_test(test_input_chebyshev, 2*π*0.0, -2*π*0.2963, 0.0, -1.8811503415036912;
             charge_exchange_frequency=2*π*2.0)

    # n_i>>n_n T_e=1
    run_test(test_input_chebyshev, 2*π*1.4467, -2*π*0.6020, 9.085788801178547,
             -3.7948015219732847; initial_density1=0.9999, initial_density2=0.0001,
             charge_exchange_frequency=2*π*0.1)
    run_test(test_input_chebyshev, 2*π*1.4467, -2*π*0.6020, 9.082828578103666,
             -3.8043350154911963; initial_density1=0.9999, initial_density2=0.0001,
             charge_exchange_frequency=2*π*2.0)

    # n_i<<n_n T_e=1
    run_test(test_input_chebyshev, 2*π*1.3954, -2*π*0.6815, 8.731402331355747,
             -4.298957441316324; initial_density1=0.0001, initial_density2=0.9999,
             charge_exchange_frequency=2*π*0.1)
    run_test(test_input_chebyshev, 2*π*0.0, -2*π*0.5112, 0.0, -3.209548708667724;
             initial_density1=0.0001, initial_density2=0.9999,
             charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=0.5
    run_test(test_input_chebyshev, 2*π*1.2671, -2*π*0.8033, 7.961282395577671,
             -5.0569493468462605, 30; T_e=0.5, nstep=1300,
             charge_exchange_frequency=2*π*0.0)
    run_test(test_input_chebyshev, 2*π*0.0, -2*π*0.2727, 0.0, -1.7108849176496357;
             T_e=0.5, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=4
    run_test(test_input_chebyshev, 2*π*1.9919, -2*π*0.2491, 12.515690550010879,
             -1.5653809098333944; T_e=4.0, charge_exchange_frequency=2*π*0.1)
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_chebyshev_split_1_moment()
    #n_i=n_n, T_e=1
    run_test(test_input_chebyshev_split_1_moment, 2*π*1.4467, -2*π*0.6020,
             9.085990812952684, -3.79588298709171)
    run_test(test_input_chebyshev_split_1_moment, 2*π*1.4240, -2*π*0.6379,
             8.934738270890447, -4.025718553464908; charge_exchange_frequency=2*π*0.1)
    run_test(test_input_chebyshev_split_1_moment, 2*π*0.0, -2*π*0.3235, 0.0,
             -2.0582986398310155; charge_exchange_frequency=2*π*1.8)
    run_test(test_input_chebyshev_split_1_moment, 2*π*0.0, -2*π*0.2963, 0.0,
             -1.8811803891784995; charge_exchange_frequency=2*π*2.0)

    # n_i>>n_n T_e=1
    run_test(test_input_chebyshev_split_1_moment, 2*π*1.4467, -2*π*0.6020,
             9.085963694047994, -3.795925359965073; initial_density1=0.9999,
             initial_density2=0.0001, charge_exchange_frequency=2*π*0.1)
    run_test(test_input_chebyshev_split_1_moment, 2*π*1.4467, -2*π*0.6020,
             9.08300151030528, -3.8054690741046944; initial_density1=0.9999,
             initial_density2=0.0001, charge_exchange_frequency=2*π*2.0)

    # n_i<<n_n T_e=1
    run_test(test_input_chebyshev_split_1_moment, 2*π*1.3954, -2*π*0.6815,
             8.737202843165628, -4.29936911741645; initial_density1=0.0001,
             initial_density2=0.9999, charge_exchange_frequency=2*π*0.1)
    run_test(test_input_chebyshev_split_1_moment, 2*π*0.0, -2*π*0.5112, 0.0,
             -3.209889224717887; initial_density1=0.0001, initial_density2=0.9999,
             charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=0.5
    run_test(test_input_chebyshev_split_1_moment, 2*π*1.2671, -2*π*0.8033,
             7.959176531864484, -5.0468118597306315, 30; T_e=0.5, nstep=1300,
             charge_exchange_frequency=2*π*0.0)
    run_test(test_input_chebyshev_split_1_moment, 2*π*0.0, -2*π*0.2727, 0.0,
             -1.7109497111415288; T_e=0.5, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=4
    run_test(test_input_chebyshev_split_1_moment, 2*π*1.9919, -2*π*0.2491,
             12.515693871624226, -1.5653870459310333; T_e=4.0,
             charge_exchange_frequency=2*π*0.1)
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_chebyshev_split_2_moments()
    #n_i=n_n, T_e=1
    run_test(test_input_chebyshev_split_2_moments, 2*π*1.4467, -2*π*0.6020,
             9.085692213794985, -3.796847219028767)
    run_test(test_input_chebyshev_split_2_moments, 2*π*1.4240, -2*π*0.6379,
             8.93547132091605, -4.028739848302094; charge_exchange_frequency=2*π*0.1)
    run_test(test_input_chebyshev_split_2_moments, 2*π*0.0, -2*π*0.3235, 0.0,
             -2.052034499180987; charge_exchange_frequency=2*π*1.8)
    run_test(test_input_chebyshev_split_2_moments, 2*π*0.0, -2*π*0.2963, 0.0,
             -1.8759661453977758; charge_exchange_frequency=2*π*2.0)

    # n_i>>n_n T_e=1
    run_test(test_input_chebyshev_split_2_moments, 2*π*1.4467, -2*π*0.6020,
             9.08566529769197, -3.7968898948273133; initial_density1=0.9999,
             initial_density2=0.0001, charge_exchange_frequency=2*π*0.1)
    run_test(test_input_chebyshev_split_2_moments, 2*π*1.4467, -2*π*0.6020,
             9.08267018403319, -3.8065237687674482; initial_density1=0.9999,
             initial_density2=0.0001, charge_exchange_frequency=2*π*2.0)

    # n_i<<n_n T_e=1
    run_test(test_input_chebyshev_split_2_moments, 2*π*1.3954, -2*π*0.6815,
             8.737881434679556, -4.305186452089822; initial_density1=0.0001,
             initial_density2=0.9999, charge_exchange_frequency=2*π*0.1)
    run_test(test_input_chebyshev_split_2_moments, 2*π*0.0, -2*π*0.5112, 0.0,
             -3.197485614384494; initial_density1=0.0001, initial_density2=0.9999,
             charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=0.5
    run_test(test_input_chebyshev_split_2_moments, 2*π*1.2671, -2*π*0.8033,
             7.957949041949206, -5.050516393765112, 30; T_e=0.5, nstep=1300,
             charge_exchange_frequency=2*π*0.0)
    run_test(test_input_chebyshev_split_2_moments, 2*π*0.0, -2*π*0.2727, 0.0,
             -1.70626927062398; T_e=0.5, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=4
    run_test(test_input_chebyshev_split_2_moments, 2*π*1.9919, -2*π*0.2491,
             12.516557919918846, -1.565597154553777; T_e=4.0,
             charge_exchange_frequency=2*π*0.1)
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_chebyshev_split_3_moments()
    #n_i=n_n, T_e=1
    run_test(test_input_chebyshev_split_3_moments, 2*π*1.4467, -2*π*0.6020,
             9.086786508104689, -3.7975280087731282)
    run_test(test_input_chebyshev_split_3_moments, 2*π*1.4240, -2*π*0.6379,
             8.934015460526556, -4.034338449186001; charge_exchange_frequency=2*π*0.1)
    run_test(test_input_chebyshev_split_3_moments, 2*π*0.0, -2*π*0.3235, 0.0,
             -2.0508568608063786; charge_exchange_frequency=2*π*1.8)
    run_test(test_input_chebyshev_split_3_moments, 2*π*0.0, -2*π*0.2963, 0.0,
             -1.874775713946675; charge_exchange_frequency=2*π*2.0)

    # n_i>>n_n T_e=1
    run_test(test_input_chebyshev_split_3_moments, 2*π*1.4467, -2*π*0.6020,
             9.086759275944901, -3.7975714393101967; initial_density1=0.9999,
             initial_density2=0.0001, charge_exchange_frequency=2*π*0.1)
    run_test(test_input_chebyshev_split_3_moments, 2*π*1.4467, -2*π*0.6020,
             9.083766358338591, -3.807232848653037; initial_density1=0.9999,
             initial_density2=0.0001, charge_exchange_frequency=2*π*2.0)

    # n_i<<n_n T_e=1
    run_test(test_input_chebyshev_split_3_moments, 2*π*1.3954, -2*π*0.6815,
             8.728901333140364, -4.317875620754033; initial_density1=0.0001,
             initial_density2=0.9999, charge_exchange_frequency=2*π*0.1)
    run_test(test_input_chebyshev_split_3_moments, 2*π*0.0, -2*π*0.5112, 0.0,
             -3.19502939956025; initial_density1=0.0001, initial_density2=0.9999,
             charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=0.5
    run_test(test_input_chebyshev_split_3_moments, 2*π*1.2671, -2*π*0.8033,
             7.956374919280151, -5.062100731226223, 30; T_e=0.5, nstep=1300,
             charge_exchange_frequency=2*π*0.0)
    run_test(test_input_chebyshev_split_3_moments, 2*π*0.0, -2*π*0.2727, 0.0,
             -1.7051785811012337; T_e=0.5, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=4
    run_test(test_input_chebyshev_split_3_moments, 2*π*1.9919, -2*π*0.2491,
             12.5171565748106, -1.5664675067239247; T_e=4.0,
             charge_exchange_frequency=2*π*0.1)
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end


@testset "sound wave" begin
    println("sound wave tests")

    # finite difference
    run_test_set_finite_difference()
    run_test_set_finite_difference_split_1_moment()
    run_test_set_finite_difference_split_2_moments()
    run_test_set_finite_difference_split_3_moments()

    # Chebyshev pseudospectral
    run_test_set_chebyshev()
    run_test_set_chebyshev_split_1_moment()
    run_test_set_chebyshev_split_2_moments()
    run_test_set_chebyshev_split_3_moments()
end
