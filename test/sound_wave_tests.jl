include("setup.jl")

using Base.Filesystem: tempname
using TimerOutputs

using moment_kinetics.array_allocation: allocate_float
using moment_kinetics.load_data: open_netcdf_file
using moment_kinetics.load_data: load_coordinate_data, load_fields_data
using moment_kinetics.analysis: analyze_fields_data
using moment_kinetics.post_processing: compute_frequencies

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

# default inputs for tests
test_input_finite_difference = Dict("n_ion_species" => 1,
                                    "n_neutral_species" => 1,
                                    "boltzmann_electron_response" => true,
                                    "run_name" => nothing, # this needs to be set by the particular test
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

# Not actually used in the tests, but needed for first argument of run_moment_kinetics
to = TimerOutput()

# Note 'name' should not be shared by any two tests in this file
function run_test(analytic_frequency, analytic_growth_rate,
                  regression_frequency, regression_growth_rate, itime_min=50;
                  args...)
    # by passing keyword arguments to run_test, args becomes a Dict which can be used to
    # update the default inputs

    # Convert keyword arguments to a unique name
    if length(args) == 0
        name = "default"
    else
        name = string((string(k, "-", v, "_") for (k, v) in args)...)

        # Remove trailing "_"
        name = chop(name)
    end

    # Provide some progress info
    println("    - testing ", name)

    # Convert dict from symbol keys to String keys
    test_inputs = Dict(String(k) => v for (k, v) in args)

    # Update default inputs with values to be changed
    input = merge(test_input_finite_difference, test_inputs)

    input["run_name"] = name

    # Suppress console output while running
    frequency = undef
    growth_rate = undef
    fit_error = undef
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
        @views growth_rate, frequency, phase, fit_error =
            compute_frequencies(shifted_time[itime_min:itime_max], delta_phi[iz0,itime_min:itime_max])
    end

    # Check the fit error is not too large, otherwise we are testing junk
    @test fit_error < 2.e-2

    # analytic_frequency and analytic_growth rate are the analytically expected values
    # (from F. Parra's calculation).
    @test isapprox(frequency, analytic_frequency, rtol=3.e-2, atol=4.e-3)
    @test isapprox(growth_rate, analytic_growth_rate, rtol=3.e-2)

    # regression_frequency and regression_growth_rate are saved numerical values, which
    # are tested with tighter tolerances than the analytic values.
    @test isapprox(frequency, regression_frequency, rtol=2.e-8, atol=3.e-8)
    @test isapprox(growth_rate, regression_growth_rate, rtol=2.e-8)
end

@testset "sound wave" begin
    println("sound wave tests")

    # finite difference
    ###################

    #n_i=n_n, T_e=1
    run_test(2*π*1.4467, -2*π*0.6020, 9.088165950344875, -3.782644195212332)
    run_test(2*π*1.4240, -2*π*0.6379, 8.94471246539846, -4.007911334911641;
             charge_exchange_frequency=2*π*0.1)
    run_test(2*π*0.0, -2*π*0.3235, -2.4875812650256877e-8, -2.0674058532583377;
             charge_exchange_frequency=2*π*1.8)
    run_test(2*π*0.0, -2*π*0.2963, 2.537172960830937e-8, -1.8859761749410544;
             charge_exchange_frequency=2*π*2.0)

    # n_i>>n_n T_e=1
    run_test(2*π*1.4467, -2*π*0.6020, 9.088140069079614, -3.7826857636062994;
             initial_density1=0.9999, initial_density2=0.0001, charge_exchange_frequency=2*π*0.1)
    run_test(2*π*1.4467, -2*π*0.6020, 9.095615164138666, -3.7823638206119243;
             initial_density1=0.9999, initial_density2=0.0001, charge_exchange_frequency=2*π*2.0)

    # n_i<<n_n T_e=1
    run_test(2*π*1.3954, -2*π*0.6815, 8.763408530044039, -4.279469251105851;
             initial_density1=0.0001, initial_density2=0.9999, charge_exchange_frequency=2*π*0.1)
    run_test(2*π*0.0, -2*π*0.5112, 0.00010548629800712489, -3.1811557181853765;
             initial_density1=0.0001, initial_density2=0.9999, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=0.5
    # Fit is difficult for low CX frequency branch at this T_e, so skip testing it
    run_test(2*π*0.0, -2*π*0.2727, 8.543432714270637e-9, -1.7140527313085931, 75;
             T_e=0.5, nstep=2500, charge_exchange_frequency=2*π*2.0)

    # n_i=n_n T_e=4
    run_test(2*π*1.9919, -2*π*0.2491, 12.516410588614505, -1.5655471316186798;
             T_e=4.0, charge_exchange_frequency=2*π*0.1)
    # CX=2*π*2.0 case with T_e=4 is too hard to converge, so skip
end
