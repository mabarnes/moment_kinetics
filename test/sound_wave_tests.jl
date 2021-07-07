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

test_input_chebyshev = merge(test_input_finite_difference,
                             Dict("run_name" => "chebyshev_pseudospectral",
                                  "z_discretization" => "chebyshev_pseudospectral",
                                  "z_ngrid" => 9,
                                  "z_nelement" => 2,
                                  "vpa_discretization" => "chebyshev_pseudospectral",
                                  "vpa_ngrid" => 17,
                                  "vpa_nelement" => 8))


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
    @test phi_fit.offset_fit_error < 1.e-6
    @test phi_fit.cosine_fit_error < 5.e-8

    # analytic_frequency and analytic_growth rate are the analytically expected values
    # (from F. Parra's calculation).
    @test isapprox(phi_fit.frequency, analytic_frequency, rtol=3.e-2)
    @test isapprox(phi_fit.growth_rate, analytic_growth_rate, rtol=3.e-2)

    # regression_frequency and regression_growth_rate are saved numerical values, which
    # are tested with tighter tolerances than the analytic values.
    @test isapprox(phi_fit.frequency, regression_frequency, rtol=5.e-10)
    @test isapprox(phi_fit.growth_rate, regression_growth_rate, rtol=5.e-10)
end

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


@testset "sound wave" begin
    println("sound wave tests")

    # finite difference
    run_test_set_finite_difference()

    # Chebyshev pseudospectral
    run_test_set_chebyshev()
end
