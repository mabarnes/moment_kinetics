module HarrisonThompson
# Test case with constant-in-space, delta-function-in-vpa source against
# analytic solution from [E. R. Harrison and W. B. Thompson. The low pressure
# plane symmetric discharge. Proc. Phys.  Soc., 74:145, 1959]

include("setup.jl")

using Base.Filesystem: tempname
using SpecialFunctions: dawson
using TimerOutputs

using moment_kinetics.load_data: open_netcdf_file
using moment_kinetics.load_data: load_coordinate_data, load_fields_data

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

# Analytic solution given by implicit equation
#   z = 1/2 ± 2/(π R_ion) * D(sqrt(-phi))
# with +'ve for z>1/2 and -'ve for z<1/2, where D() is the 'Dawson function'
#   D(x) = exp(-x^2) ∫_0^x exp(y^2) dy
#
# If the ionization rate is too large (threshold is somewhere between 0.25 and
# 0.5), there is no steady solution for phi
#
# Note the derivative of the dawson function is
Dprime(x) = -2.0 * x * dawson(x) + 1.0

maxits = 10000

function newton(f, fprime, z, args...)
    phi = -1.0e-6
    count = 0
    while abs(f(phi, z, args...)) > 1.e-14
        phi = phi - f(phi, z, args...) / fprime(phi, z, args...)
        if phi > -eps()
            # phi must be negative, but iteration might overshoot
            phi = -eps()
        end
        count += 1
        if count > maxits
            error("Reached maximum iterations for z=$z with phi=$phi")
        end
    end
    return phi
end

# want to find phi such that f(phi) is zero
fnegative(phi, z, R_ion) = - z - 2.0 / π / R_ion * dawson(sqrt(-phi))
fpositive(phi, z, R_ion) = - z + 2.0 / π / R_ion * dawson(sqrt(-phi))
# derivative of f for Newton iteration
fprimenegative(phi, z, R_ion) = 1.0 / π / R_ion / sqrt(-phi) * Dprime(sqrt(-phi))
fprimepositive(phi, z, R_ion) = - 1.0 / π / R_ion / sqrt(-phi) * Dprime(sqrt(-phi))
function findphi(z, R_ion)
    if z < - eps()
        return newton(fnegative, fprimenegative, z, R_ion)
    elseif z > eps()
        return newton(fpositive, fprimepositive, z, R_ion)
    else
        return 0.0
    end
end

# default inputs for tests
test_input_finite_difference = Dict("n_ion_species" => 1,
                                    "n_neutral_species" => 1,
                                    "boltzmann_electron_response" => true,
                                    "run_name" => "finite_difference",
                                    "base_directory" => test_output_directory,
                                    "evolve_moments_density" => false,
                                    "evolve_moments_parallel_flow" => false,
                                    "evolve_moments_parallel_pressure" => false,
                                    "evolve_moments_conservation" => false,
                                    "T_e" => 1.0,
                                    "T_wall" => 1.0,
                                    "initial_density1" => 1.0,
                                    "initial_temperature1" => 1.0,
                                    "z_IC_option1" => "gaussian",
                                    "z_IC_density_amplitude1" => 0.001,
                                    "z_IC_density_phase1" => 0.0,
                                    "z_IC_upar_amplitude1" => 0.0,
                                    "z_IC_upar_phase1" => 0.0,
                                    "z_IC_temperature_amplitude1" => 0.0,
                                    "z_IC_temperature_phase1" => 0.0,
                                    "vpa_IC_option1" => "gaussian",
                                    "vpa_IC_density_amplitude1" => 1.0,
                                    "vpa_IC_density_phase1" => 0.0,
                                    "vpa_IC_upar_amplitude1" => 0.0,
                                    "vpa_IC_upar_phase1" => 0.0,
                                    "vpa_IC_temperature_amplitude1" => 0.0,
                                    "vpa_IC_temperature_phase1" => 0.0,
                                    "charge_exchange_frequency" => 0.0,
                                    "ionization_frequency" => 0.688,
                                    "constant_ionization_rate" => true,
                                    "nstep" => 9000,
                                    "dt" => 0.0005,
                                    "nwrite" => 9000,
                                    "use_semi_lagrange" => false,
                                    "n_rk_stages" => 4,
                                    "split_operators" => false,
                                    "r_ngrid" => 1,
                                    "r_nelement" => 1,
                                    "r_bc" => "periodic",
                                    "r_discretization" => "finite_difference",
                                    "z_ngrid" => 100,
                                    "z_nelement" => 1,
                                    "z_bc" => "wall",
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
                                  "vpa_nelement" => 10))

test_input_chebyshev_matrix_multiply = merge(
    test_input_chebyshev,
    Dict("run_name" => "chebyshev_pseudospectral_matrix_multiply",
         "z_discretization" => "chebyshev_pseudospectral_matrix_multiply",
         "vpa_discretization" => "chebyshev_pseudospectral_matrix_multiply"))

# Not actually used in the tests, but needed for first argument of run_moment_kinetics
to = TimerOutput()

"""
Run a test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, analytic_rtol, analytic_atol, expected_phi,
                  regression_rtol, regression_atol; args...)
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
    phi = undef
    analytic_phi = undef
    z = undef
    quietoutput() do
        # run simulation
        run_moment_kinetics(to, input)
    end

    if global_rank[] == 0
        quietoutput() do

            # Load and analyse output
            #########################

            path = joinpath(realpath(input["base_directory"]), name, name)

            # open the netcdf file and give it the handle 'fid'
            fid = open_netcdf_file(path)

            # load space-time coordinate data
            nvpa, vpa, vpa_wgts, nz, z, z_wgts, Lz, nr, r, r_wgts, Lr, ntime, time = load_coordinate_data(fid)

            # load fields data
            phi_zrt = load_fields_data(fid)

            close(fid)

            phi = phi_zrt[:,1,:]
            
            analytic_phi = [findphi(zval, input["ionization_frequency"]) for zval ∈ z]
        end

        nz = length(z)
        # Analytic solution defines phi=0 at mid-point, so need to offset the code solution
        offset = phi[(nz+1)÷2, end]
        # Error is large on the boundary points, so test those separately
        @test isapprox(phi[2:end-1, end] .- offset, analytic_phi[2:end-1],
                       rtol=analytic_rtol, atol=analytic_atol)
        @test isapprox(phi[[1, end], end] .- offset, analytic_phi[[1, end]],
                       rtol=10.0*analytic_rtol, atol=analytic_atol)

        # Regression test
        @test isapprox(phi[:, end], expected_phi, rtol=regression_rtol, atol=regression_atol)
    end
end

function runtests()
    @testset "Harrison-Thompson" verbose=use_verbose begin
        println("Harrison-Thompson wall boundary condition tests")

        @testset_skip "FD version forms discontinuity in vpa at z=±L/2" "finite difference" begin
            run_test(test_input_finite_difference, 1.e-3, 1.e-4, zeros(100), 1.e-14, 1.e-15)
        end

        @testset "Chebyshev$suffix" for (input, suffix) ∈
                ((test_input_chebyshev, ""),
                 (test_input_chebyshev_matrix_multiply, " matrix multiply"))
            run_test(input, 3.e-2, 3.e-3,
                     [-0.8100630096349565, -0.6518856160147646, -0.4205717722276304,
                      -0.2829742494858749, -0.1833320256571621, -0.1359546420028737,
                      -0.10968680094355439, -0.10655932911687617, -0.10320743994056783,
                      -0.1065593291168768, -0.10968680094355476, -0.13595464200287433,
                      -0.18333202565716236, -0.28297424948587463, -0.42057177222762804,
                      -0.6518856160147616, -0.8100630096349531], 5.e-9, 1.e-15)
        end
    end
end

end # HarrisonThompson


using .HarrisonThompson

HarrisonThompson.runtests()
