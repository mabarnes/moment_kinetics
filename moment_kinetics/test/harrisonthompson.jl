module HarrisonThompson
# Test case with constant-in-space, delta-function-in-vpa source against
# analytic solution from [E. R. Harrison and W. B. Thompson. The low pressure
# plane symmetric discharge. Proc. Phys.  Soc., 74:145, 1959]

include("setup.jl")

using Base.Filesystem: tempname
using SpecialFunctions: dawson

using moment_kinetics.load_data: open_readonly_output_file
using moment_kinetics.load_data: load_fields_data, load_time_data
using moment_kinetics.load_data: load_species_data, load_coordinate_data

ionization_frequency = 0.688

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
                                    "n_neutral_species" => 0,
                                    "boltzmann_electron_response" => true,
                                    "run_name" => "finite_difference",
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
                                    "ionization_frequency" => 0.0,
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
                                    "vpa_bc" => "zero",
                                    "vpa_discretization" => "finite_difference",
                                    "vz_ngrid" => 200,
                                    "vz_nelement" => 1,
                                    "vz_L" => 8.0,
                                    "vz_bc" => "zero",
                                    "vz_discretization" => "finite_difference",
                                    "ion_source" => Dict("active" => true,
                                                         "source_strength" => ionization_frequency,
                                                         "source_T" => 0.25,
                                                         "z_profile" => "constant",
                                                         "r_profile" => "constant"),
                                   )

test_input_chebyshev = merge(test_input_finite_difference,
                             Dict("run_name" => "chebyshev_pseudospectral",
                                  "z_discretization" => "chebyshev_pseudospectral",
                                  "z_ngrid" => 9,
                                  "z_nelement" => 2,
                                  "vpa_discretization" => "chebyshev_pseudospectral",
                                  "vpa_ngrid" => 17,
                                  "vpa_nelement" => 10,
                                  "vz_discretization" => "chebyshev_pseudospectral",
                                  "vz_ngrid" => 17,
                                  "vz_nelement" => 10))

test_input_chebyshev_split1 = merge(test_input_chebyshev,
                                    Dict("run_name" => "chebyshev_pseudospectral_split1",
                                         "evolve_moments_density" => true,
                                         "evolve_moments_conservation" => true))

test_input_chebyshev_split2 = merge(test_input_chebyshev_split1,
                                    Dict("run_name" => "chebyshev_pseudospectral_split2",
                                         "evolve_moments_parallel_flow" => true,
                                         "numerical_dissipation" => Dict("force_minimum_pdf_value" => 0.0)))

test_input_chebyshev_split3 = merge(test_input_chebyshev_split2,
                                    Dict("run_name" => "chebyshev_pseudospectral_split3",
                                         "evolve_moments_parallel_pressure" => true))

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
    phi = nothing
    analytic_phi = nothing
    z = nothing
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

            # load space-time coordinate data
            z, z_spectral, z_chunk_size = load_coordinate_data(fid, "z")
            r, r_spectral, r_chunk_size = load_coordinate_data(fid, "r")
            ntime, time = load_time_data(fid)
            n_ion_species, n_neutral_species = load_species_data(fid)
            
            # load fields data
            phi_zrt, Er_zrt, Ez_zrt = load_fields_data(fid)

            close(fid)

            phi = phi_zrt[:,1,:]
            
            analytic_phi = [findphi(zval, ionization_frequency) for zval ∈ z.grid]
        end

        # Analytic solution defines phi=0 at mid-point, so need to offset the code solution
        offset = phi[(z.n+1)÷2, end]
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
    # Create a temporary directory for test output
    test_output_directory = get_MPI_tempdir()

    @testset "Harrison-Thompson" verbose=use_verbose begin
        println("Harrison-Thompson wall boundary condition tests")

        @testset_skip "FD version forms discontinuity in vpa at z=±L/2" "finite difference" begin
            test_input_finite_difference["base_directory"] = test_output_directory
            run_test(test_input_finite_difference, 1.e-3, 1.e-4, zeros(100), 1.e-14, 1.e-15)
        end

        @testset "Chebyshev" begin
            test_input_chebyshev["base_directory"] = test_output_directory
            run_test(test_input_chebyshev, 3.e-2, 3.e-3,
                     [-0.8270506701954182, -0.6647482038047513, -0.4359510242978734,
                      -0.2930090318306279, -0.19789542580389763, -0.14560099254974576,
                      -0.12410802135258239, -0.11657014257474364, -0.11761846656548933,
                      -0.11657014257474377, -0.12410802135258239, -0.1456009925497464,
                      -0.19789542580389616, -0.2930090318306262, -0.435951024297872,
                      -0.66474820380475, -0.8270506701954171], 5.0e-9, 1.e-15)
        end
        @testset "Chebyshev split 1" begin
            test_input_chebyshev_split1["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split1, 3.e-2, 3.e-3,
                     [-0.808956646073449, -0.6619131832543625, -0.4308291868843453,
                      -0.295820339728472, -0.19344190006125275, -0.1492514208442407,
                      -0.11977511930743077, -0.12060863604650167, -0.11342106824862994,
                      -0.12060863604649866, -0.11977511930742626, -0.14925142084423915,
                      -0.1934419000612479, -0.295820339728463, -0.4308291868843545,
                      -0.6619131832543678, -0.808956646073442], 5.0e-9, 1.e-15)
        end
        @testset "Chebyshev split 2" begin
            test_input_chebyshev_split2["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split2, 5.e-2, 3.e-3,
                     [-0.7667804422571606, -0.6128777083267765, -0.39031953439035494,
                      -0.27326504140885904, -0.15311275955907663, -0.11567486122959246,
                      -0.09226471519174786, -0.07477085120501512, -0.07206945428218994,
                      -0.07477085120545898, -0.09226471518828984, -0.11567486123016281,
                      -0.15311275955613904, -0.273265041412353, -0.3903195344134153,
                      -0.612877708320375, -0.766780442235556], 5.0e-9, 1.e-15)
        end
        # The 'split 3' test is pretty badly resolved, but don't want to increase
        # run-time!
        @testset "Chebyshev split 3" begin
            test_input_chebyshev_split3["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split3, 2.1e-1, 3.e-3,
                     [-0.5535421015240105, -0.502816770781802, -0.3755477646148533,
                      -0.24212761527100635, -0.15737450156025806, -0.11242832417550296,
                      -0.09168434722655881, -0.08653015173768085, -0.0858195594227437,
                      -0.08653015173768933, -0.09168434722650211, -0.11242832417546023,
                      -0.15737450156026872, -0.24212761527101284, -0.3755477646149367,
                      -0.5028167707818142, -0.5535421015238932], 5.0e-9, 1.e-15)
        end
    end

    if global_rank[] == 0
        # Delete output directory to avoid using too much disk space
        rm(realpath(test_output_directory); recursive=true)
    end
end

end # HarrisonThompson


using .HarrisonThompson

HarrisonThompson.runtests()
