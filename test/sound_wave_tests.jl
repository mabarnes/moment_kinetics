include("setup.jl")

using Base.Filesystem: tempname
using TimerOutputs
#using Plots: plot, plot!, gui

using moment_kinetics.array_allocation: allocate_float
using moment_kinetics.load_data: open_netcdf_file
using moment_kinetics.load_data: load_coordinate_data, load_fields_data
using moment_kinetics.analysis: analyze_fields_data
using moment_kinetics.post_processing: fit_delta_phi_mode

const analytical_rtol = 3.e-2
const regression_rtol = 2.e-8

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
                                    "z_IC_density_amplitude1" => 0.001,
                                    "z_IC_upar_amplitude1" => 0.0,
                                    "z_IC_temperature_amplitude1" => 0.0,
                                    "z_IC_density_amplitude2" => 0.001,
                                    "z_IC_upar_amplitude2" => 0.0,
                                    "z_IC_temperature_amplitude2" => 0.0,
                                    "charge_exchange_frequency" => 2*π*0.1,
                                    "nstep" => 1500,
                                    "dt" => 0.002,
                                    "nwrite" => 20,
                                    "use_semi_lagrange" => false,
                                    "n_rk_stages" => 4,
                                    "split_operators" => false,
                                    "z_ngrid" => 100,
                                    "z_nelement" => 1,
                                    "z_discretization" => "finite_difference",
                                    "vpa_ngrid" => 180,
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
    @test isapprox(phi_fit.frequency, analytic_frequency, rtol=analytical_rtol)
    @test isapprox(phi_fit.growth_rate, analytic_growth_rate, rtol=analytical_rtol)

    # regression_frequency and regression_growth_rate are saved numerical values, which
    # are tested with tighter tolerances than the analytic values.
    @test isapprox(phi_fit.frequency, regression_frequency, rtol=regression_rtol)
    @test isapprox(phi_fit.growth_rate, regression_growth_rate, rtol=regression_rtol)
end


# run_test_set_* functions call run_test for various parameters, and record the expected
# values to be used for regression_frequency and regression_growth_rate for each
# particular case

function run_test_set_finite_difference()
    #n_i=n_n, T_e=1
    @long run_test(test_input_finite_difference, 2*π*1.4467, -2*π*0.6020,
                   9.101375221513, -3.7456468069748703;
                   charge_exchange_frequency=2*π*0.0)
    run_test(test_input_finite_difference, 2*π*1.4240, -2*π*0.6379, 8.994196879504377,
             -3.952746606360892,)
    @long run_test(test_input_finite_difference, 2*π*0.0, -2*π*0.3235, 0.0,
                   -2.0609440303447593,; charge_exchange_frequency=2*π*1.8)
    @long run_test(test_input_finite_difference, 2*π*0.0, -2*π*0.2963, 0.0,
                   -1.8829039272407633; charge_exchange_frequency=2*π*2.0)

    # n_i>>n_n T_e=1
    @long run_test(test_input_finite_difference, 2*π*1.4467, -2*π*0.6020,
                   9.101353480970252, -3.7456848823948112,; initial_density1=0.9999,
                   initial_density2=0.0001)
    @long run_test(test_input_finite_difference, 2*π*1.4467, -2*π*0.6020,
                   9.098661097303278, -3.7547026888371073; initial_density1=0.9999,
                   initial_density2=0.0001, charge_exchange_frequency=2*π*2.0)

    # n_i<<n_n T_e=1
    @long run_test(test_input_finite_difference, 2*π*1.3954, -2*π*0.6815,
                   8.906458967951309, -4.222513273783303; initial_density1=0.0001,
                   initial_density2=0.9999)
    @long run_test(test_input_finite_difference, 2*π*0.0, -2*π*0.5112, 0.0,
                   -3.216839550416299; initial_density1=0.0001, initial_density2=0.9999,
                   charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=0.5
    @long run_test(test_input_finite_difference, 2*π*1.2671, -2*π*0.8033,
                   8.004611001882369, -4.9080592330657575, 30; T_e=0.5, nstep=1300,
                   charge_exchange_frequency=2*π*0.0)
    @long run_test(test_input_finite_difference, 2*π*0.0, -2*π*0.2727, 0.0,
                   -1.7124950743598237; T_e=0.5, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=4
    @long run_test(test_input_finite_difference, 2*π*1.9919, -2*π*0.2491,
                   12.515603454575245, -1.5656284378719276; T_e=4.0)
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_finite_difference_split_1_moment()
    #n_i=n_n, T_e=1
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.4467, -2*π*0.6020,
             9.100194899020384, -3.7492377949160085; charge_exchange_frequency=2*π*0.0)
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.4240, -2*π*0.6379,
             8.989588662618768, -3.9582104033204746)
    run_test(test_input_finite_difference_split_1_moment, 2*π*0.0, -2*π*0.3235, 0.0,
             -2.060729141021252; charge_exchange_frequency=2*π*1.8)
    run_test(test_input_finite_difference_split_1_moment, 2*π*0.0, -2*π*0.2963, 0.0,
             -1.882747577022001; charge_exchange_frequency=2*π*2.0)

    # n_i>>n_n T_e=1
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.4467, -2*π*0.6020,
             9.100172680098513, -3.749276142899367; initial_density1=0.9999,
             initial_density2=0.0001)
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.4467, -2*π*0.6020,
             9.097458046489898, -3.7583307292981813; initial_density1=0.9999,
             initial_density2=0.0001, charge_exchange_frequency=2*π*2.0)

    # n_i<<n_n T_e=1
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.3954, -2*π*0.6815,
             8.893135769361905, -4.228505411863995; initial_density1=0.0001,
             initial_density2=0.9999)
    run_test(test_input_finite_difference_split_1_moment, 2*π*0.0, -2*π*0.5112, 0.0,
             -3.216385993093391; initial_density1=0.0001, initial_density2=0.9999,
             charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=0.5
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.2671, -2*π*0.8033,
             7.9991477497107315, -4.923599808100078, 30; T_e=0.5, nstep=1300,
             charge_exchange_frequency=2*π*0.0)
    run_test(test_input_finite_difference_split_1_moment, 2*π*0.0, -2*π*0.2727, 0.0,
             -1.7123394163548804; T_e=0.5, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=4
    run_test(test_input_finite_difference_split_1_moment, 2*π*1.9919, -2*π*0.2491,
             12.51558002809753, -1.5655611247187082; T_e=4.0)
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_finite_difference_split_2_moments()
    #n_i=n_n, T_e=1
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.4467, -2*π*0.6020,
             9.10738136411676, -3.7265296672701025; charge_exchange_frequency=2*π*0.0)
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.4240, -2*π*0.6379,
             9.015887756307855, -3.9283935582652725)
    run_test(test_input_finite_difference_split_2_moments, 2*π*0.0, -2*π*0.3235, 0.0,
             -2.0555088875845007; charge_exchange_frequency=2*π*1.8)
    run_test(test_input_finite_difference_split_2_moments, 2*π*0.0, -2*π*0.2963, 0.0,
             -1.8782098405028747; charge_exchange_frequency=2*π*2.0)

    # n_i>>n_n T_e=1
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.4467, -2*π*0.6020,
             9.107361608711102, -3.7265664501973847; initial_density1=0.9999,
             initial_density2=0.0001)
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.4467, -2*π*0.6020,
             9.104748310108103, -3.7354605581245637; initial_density1=0.9999,
             initial_density2=0.0001, charge_exchange_frequency=2*π*2.0)

    # n_i<<n_n T_e=1
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.3954, -2*π*0.6815,
             8.962270040229141, -4.197429691120341; initial_density1=0.0001,
             initial_density2=0.9999)
    run_test(test_input_finite_difference_split_2_moments, 2*π*0.0, -2*π*0.5112, 0.0,
             -3.2069705641654074; initial_density1=0.0001, initial_density2=0.9999,
             charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=0.5
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.2671, -2*π*0.8033,
             7.9905362193818625, -4.949572701349685, 30; T_e=0.5, nstep=1300,
             z_ngrid=150, vpa_ngrid=200, charge_exchange_frequency=2*π*0.0)
    run_test(test_input_finite_difference_split_2_moments, 2*π*0.0, -2*π*0.2727, 0.0,
             -1.7083318215938026; T_e=0.5, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=4
    run_test(test_input_finite_difference_split_2_moments, 2*π*1.9919, -2*π*0.2491,
             12.516476318129225, -1.565656654881501; T_e=4.0)
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_finite_difference_split_3_moments()
    #n_i=n_n, T_e=1
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*1.4467,
                   -2*π*0.6020, 9.102871623697629, -3.744824891487915;
                   charge_exchange_frequency=2*π*0.0)
    run_test(test_input_finite_difference_split_3_moments, 2*π*1.4240, -2*π*0.6379,
             8.996507905599843, -3.957028530654479)
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*0.0, -2*π*0.3235,
                   0.0, -2.0536185410365264; charge_exchange_frequency=2*π*1.8)
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*0.0, -2*π*0.2963,
                   0.0, -1.8765760203001856; charge_exchange_frequency=2*π*2.0)

    # n_i>>n_n T_e=1
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*1.4467,
                   -2*π*0.6020, 9.10285006750063, -3.744863751007747;
                   initial_density1=0.9999, initial_density2=0.0001)
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*1.4467,
                   -2*π*0.6020, 9.100150117265162, -3.7539646624025633,;
                   initial_density1=0.9999, initial_density2=0.0001,
                   charge_exchange_frequency=2*π*2.0)

    # n_i<<n_n T_e=1
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*1.3954,
                   -2*π*0.6815, 8.9104984514782, -4.23498467619971;
                   initial_density1=0.0001, initial_density2=0.9999)
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*0.0, -2*π*0.5112,
                   0.0, -3.202311734582212; initial_density1=0.0001,
                   initial_density2=0.9999, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=0.5
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*1.2671,
                   -2*π*0.8033, 8.003030612024757, -4.915968068689116, 30; T_e=0.5,
                   nstep=1300, charge_exchange_frequency=2*π*0.0)
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*0.0, -2*π*0.2727,
                   0.0, -1.7067827875683994; T_e=0.5, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=4
    @long run_test(test_input_finite_difference_split_3_moments, 2*π*1.9919,
                   -2*π*0.2491, 12.51718037187671, -1.5664591071220006; T_e=4.0)
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_chebyshev()
    #n_i=n_n, T_e=1
    @long run_test(test_input_chebyshev, 2*π*1.4467, -2*π*0.6020, 9.085815883255613,
                   -3.794758980708628; charge_exchange_frequency=2*π*0.0)
    run_test(test_input_chebyshev, 2*π*1.4240, -2*π*0.6379, 8.933662137891284,
             -4.025210168129686)
    @long run_test(test_input_chebyshev, 2*π*0.0, -2*π*0.3235, 0.0, -2.058256214667329;
                   charge_exchange_frequency=2*π*1.8)
    @long run_test(test_input_chebyshev, 2*π*0.0, -2*π*0.2963, 0.0, -1.8811503415036912;
                   charge_exchange_frequency=2*π*2.0)

    # n_i>>n_n T_e=1
    @long run_test(test_input_chebyshev, 2*π*1.4467, -2*π*0.6020, 9.085788801178547,
                   -3.7948015219732847; initial_density1=0.9999,
                   initial_density2=0.0001)
    @long run_test(test_input_chebyshev, 2*π*1.4467, -2*π*0.6020, 9.082828578103666,
                   -3.8043350154911963; initial_density1=0.9999,
                   initial_density2=0.0001, charge_exchange_frequency=2*π*2.0)

    # n_i<<n_n T_e=1
    @long run_test(test_input_chebyshev, 2*π*1.3954, -2*π*0.6815, 8.731402331355747,
                   -4.298957441316324; initial_density1=0.0001, initial_density2=0.9999)
    @long run_test(test_input_chebyshev, 2*π*0.0, -2*π*0.5112, 0.0, -3.209548708667724;
                   initial_density1=0.0001, initial_density2=0.9999,
                   charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=0.5
    @long run_test(test_input_chebyshev, 2*π*1.2671, -2*π*0.8033, 7.961282395577671,
                   -5.0569493468462605, 30; T_e=0.5, nstep=1300,
                   charge_exchange_frequency=2*π*0.0)
    @long run_test(test_input_chebyshev, 2*π*0.0, -2*π*0.2727, 0.0, -1.7108849176496357;
                   T_e=0.5, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=4
    @long run_test(test_input_chebyshev, 2*π*1.9919, -2*π*0.2491, 12.515690550010879,
                   -1.5653809098333944; T_e=4.0)
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_chebyshev_split_1_moment()
    #n_i=n_n, T_e=1
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*1.4467, -2*π*0.6020,
                   9.085990812952684, -3.79588298709171;
                   charge_exchange_frequency=2*π*0.0)
    run_test(test_input_chebyshev_split_1_moment, 2*π*1.4240, -2*π*0.6379,
             8.934738270890447, -4.025718553464908)
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*0.0, -2*π*0.3235, 0.0,
                   -2.0582986398310155; charge_exchange_frequency=2*π*1.8)
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*0.0, -2*π*0.2963, 0.0,
                   -1.8811803891784995; charge_exchange_frequency=2*π*2.0)

    # n_i>>n_n T_e=1
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*1.4467, -2*π*0.6020,
                   9.085963694047994, -3.795925359965073; initial_density1=0.9999,
                   initial_density2=0.0001)
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*1.4467, -2*π*0.6020,
                   9.08300151030528, -3.8054690741046944; initial_density1=0.9999,
                   initial_density2=0.0001, charge_exchange_frequency=2*π*2.0)

    # n_i<<n_n T_e=1
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*1.3954, -2*π*0.6815,
                   8.737202843165628, -4.29936911741645; initial_density1=0.0001,
                   initial_density2=0.9999)
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*0.0, -2*π*0.5112, 0.0,
                   -3.209889224717887; initial_density1=0.0001, initial_density2=0.9999,
                   charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=0.5
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*1.2671, -2*π*0.8033,
                   7.959176531864484, -5.0468118597306315, 30; T_e=0.5, nstep=1300,
                   charge_exchange_frequency=2*π*0.0)
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*0.0, -2*π*0.2727, 0.0,
                   -1.7109497111415288; T_e=0.5, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=4
    @long run_test(test_input_chebyshev_split_1_moment, 2*π*1.9919, -2*π*0.2491,
                   12.515693871624226, -1.5653870459310333; T_e=4.0)
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_chebyshev_split_2_moments()
    #n_i=n_n, T_e=1
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*1.4467, -2*π*0.6020,
                   9.085692213794985, -3.796847219028767;
                   charge_exchange_frequency=2*π*0.0)
    run_test(test_input_chebyshev_split_2_moments, 2*π*1.4240, -2*π*0.6379,
             8.93547132091605, -4.028739848302094)
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*0.0, -2*π*0.3235, 0.0,
                   -2.052034499180987; charge_exchange_frequency=2*π*1.8)
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*0.0, -2*π*0.2963, 0.0,
                   -1.8759661453977758; charge_exchange_frequency=2*π*2.0)

    # n_i>>n_n T_e=1
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*1.4467, -2*π*0.6020,
                   9.08566529769197, -3.7968898948273133; initial_density1=0.9999,
                   initial_density2=0.0001)
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*1.4467, -2*π*0.6020,
                   9.08267018403319, -3.8065237687674482; initial_density1=0.9999,
                   initial_density2=0.0001, charge_exchange_frequency=2*π*2.0)

    # n_i<<n_n T_e=1
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*1.3954, -2*π*0.6815,
                   8.737881434679556, -4.305186452089822; initial_density1=0.0001,
                   initial_density2=0.9999)
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*0.0, -2*π*0.5112, 0.0,
                   -3.197485614384494; initial_density1=0.0001, initial_density2=0.9999,
                   charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=0.5
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*1.2671, -2*π*0.8033,
                   7.960270878102388, -5.0492363327785865, 40; T_e=0.5, nstep=1300,
                   nwrite=10, charge_exchange_frequency=2*π*0.0)
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*0.0, -2*π*0.2727, 0.0,
                   -1.70626927062398; T_e=0.5, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=4
    @long run_test(test_input_chebyshev_split_2_moments, 2*π*1.9919, -2*π*0.2491,
                   12.516557919918846, -1.565597154553777; T_e=4.0)
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end

function run_test_set_chebyshev_split_3_moments()
    #n_i=n_n, T_e=1
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*1.4467, -2*π*0.6020,
                   9.086786508104689, -3.7975280087731282;
                   charge_exchange_frequency=2*π*0.0)
    run_test(test_input_chebyshev_split_3_moments, 2*π*1.4240, -2*π*0.6379,
             8.934015460526556, -4.034338449186001)
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*0.0, -2*π*0.3235, 0.0,
                   -2.0508568608063786; charge_exchange_frequency=2*π*1.8)
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*0.0, -2*π*0.2963, 0.0,
                   -1.874775713946675; charge_exchange_frequency=2*π*2.0)

    # n_i>>n_n T_e=1
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*1.4467, -2*π*0.6020,
                   9.086759275944901, -3.7975714393101967; initial_density1=0.9999,
                   initial_density2=0.0001)
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*1.4467, -2*π*0.6020,
                   9.083766358338591, -3.807232848653037; initial_density1=0.9999,
                   initial_density2=0.0001, charge_exchange_frequency=2*π*2.0)

    # n_i<<n_n T_e=1
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*1.3954, -2*π*0.6815,
                   8.728901333140364, -4.317875620754033; initial_density1=0.0001,
                   initial_density2=0.9999)
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*0.0, -2*π*0.5112, 0.0,
                   -3.19502939956025; initial_density1=0.0001, initial_density2=0.9999,
                   charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=0.5
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*1.2671, -2*π*0.8033,
                   7.959846943693199, -5.056059461788515, 80; T_e=0.5, nstep=1300,
                   nwrite=5, charge_exchange_frequency=2*π*0.0)
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*0.0, -2*π*0.2727, 0.0,
                   -1.7051785811012337; T_e=0.5, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=4
    @long run_test(test_input_chebyshev_split_3_moments, 2*π*1.9919, -2*π*0.2491,
                   12.5171565748106, -1.5664675067239247; T_e=4.0)
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end


@testset "sound wave" begin
    println("sound wave tests")

    @testset "finite difference" begin
        run_test_set_finite_difference()
        @long run_test_set_finite_difference_split_1_moment()
        @long run_test_set_finite_difference_split_2_moments()
        run_test_set_finite_difference_split_3_moments()
    end

    @testset "Chebyshev" begin
        run_test_set_chebyshev()
        run_test_set_chebyshev_split_1_moment()
        run_test_set_chebyshev_split_2_moments()
        run_test_set_chebyshev_split_3_moments()
    end
end
