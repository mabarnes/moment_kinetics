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
using moment_kinetics.utils: merge_dict_with_kwargs!

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
test_input_finite_difference = OptionsDict("composition" => OptionsDict("n_ion_species" => 1,
                                                                        "n_neutral_species" => 0,
                                                                        "electron_physics" => "boltzmann_electron_response",
                                                                        "T_e" => 1.0,
                                                                        "T_wall" => 1.0),
                                           "ion_species_1" => OptionsDict("initial_density" => 1.0,
                                                                          "initial_temperature" => 1.0),
                                           "z_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                                               "density_amplitude" => 0.0,
                                                                               "density_phase" => 0.0,
                                                                               "upar_amplitude" => 0.0,
                                                                               "upar_phase" => 0.0,
                                                                               "temperature_amplitude" => 0.0,
                                                                               "temperature_phase" => 0.0),
                                           "vpa_IC_ion_species_1" => OptionsDict("initialization_option" => "gaussian",
                                                                                 "density_amplitude" => 1.0,
                                                                                 "density_phase" => 0.0,
                                                                                 "upar_amplitude" => 0.0,
                                                                                 "upar_phase" => 0.0,
                                                                                 "temperature_amplitude" => 0.0,
                                                                                 "temperature_phase" => 0.0),
                                           "output" => OptionsDict("run_name" => "finite_difference",
                                                                   "parallel_io" => false),
                                           "evolve_moments" => OptionsDict("density" => false,
                                                                           "parallel_flow" => false,
                                                                           "parallel_pressure" => false,
                                                                           "moments_conservation" => false),
                                           "reactions" => OptionsDict("charge_exchange_frequency" => 0.0,
                                                                      "ionization_frequency" => 0.0),
                                           "timestepping" => OptionsDict("nstep" => 9000,
                                                                         "dt" => 0.0005,
                                                                         "nwrite" => 9000,
                                                                         "split_operators" => false),
                                           "r" => OptionsDict("ngrid" => 1,
                                                              "nelement" => 1,
                                                              "bc" => "periodic",
                                                              "discretization" => "finite_difference"),
                                           "z" => OptionsDict("ngrid" => 100,
                                                              "nelement" => 1,
                                                              "bc" => "wall",
                                                              "discretization" => "finite_difference"),
                                           "vpa" => OptionsDict("ngrid" => 200,
                                                                "nelement" => 1,
                                                                "L" => 8.0,
                                                                "bc" => "zero",
                                                                "discretization" => "finite_difference"),
                                           "vz" => OptionsDict("ngrid" => 200,
                                                               "nelement" => 1,
                                                               "L" => 8.0,
                                                               "bc" => "zero",
                                                               "discretization" => "finite_difference"),
                                           "ion_source_1" => OptionsDict("active" => true,
                                                                         "source_strength" => ionization_frequency,
                                                                         "source_T" => 0.25,
                                                                         "z_profile" => "constant",
                                                                         "r_profile" => "constant"),
                                          )

test_input_chebyshev = recursive_merge(test_input_finite_difference,
                                       OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral"),
                                                   "z" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                       "ngrid" => 9,
                                                                       "nelement" => 2),
                                                   "vpa" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                        "ngrid" => 17,
                                                                        "nelement" => 10),
                                                   "vz" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                       "ngrid" => 17,
                                                                       "nelement" => 10),
                                                  ))

test_input_chebyshev_split1 = recursive_merge(test_input_chebyshev,
                                              OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_split1"),
                                                          "evolve_moments" => OptionsDict("density" => true,
                                                                                          "moments_conservation" => true),
                                                         ))

test_input_chebyshev_split2 = recursive_merge(test_input_chebyshev_split1,
                                              OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_split2"),
                                                          "evolve_moments" => OptionsDict("parallel_flow" => true),
                                                          "ion_numerical_dissipation" => OptionsDict("force_minimum_pdf_value" => 0.0),
                                                         ))

test_input_chebyshev_split3 = recursive_merge(test_input_chebyshev_split2,
                                              OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral_split3"),
                                                          "evolve_moments" => OptionsDict("parallel_pressure" => true),
                                                         ))

"""
Run a test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, analytic_rtol, analytic_atol, expected_phi,
                  regression_rtol, regression_atol; args...)
    # by passing keyword arguments to run_test, args becomes a Dict which can be used to
    # update the default inputs

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

            path = joinpath(realpath(input["output"]["base_directory"]), name, name)

            # open the netcdf file and give it the handle 'fid'
            fid = open_readonly_output_file(path,"moments")

            # load space-time coordinate data
            z, z_spectral, z_chunk_size = load_coordinate_data(fid, "z"; ignore_MPI=true)
            r, r_spectral, r_chunk_size = load_coordinate_data(fid, "r"; ignore_MPI=true)
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
        if expected_phi === nothing
            println("values tested would be ", phi[:,end])
        else
            @test isapprox(phi[:, end], expected_phi, rtol=regression_rtol, atol=regression_atol)
        end
    end
end

function runtests()
    # Create a temporary directory for test output
    test_output_directory = get_MPI_tempdir()

    @testset "Harrison-Thompson" verbose=use_verbose begin
        println("Harrison-Thompson wall boundary condition tests")

        @testset_skip "FD version forms discontinuity in vpa at z=±L/2" "finite difference" begin
            test_input_finite_difference["output"]["base_directory"] = test_output_directory
            run_test(test_input_finite_difference, 1.e-3, 1.e-4, zeros(100), 1.e-14, 1.e-15)
        end

        @testset "Chebyshev" begin
            test_input_chebyshev["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev, 3.e-2, 3.e-3,
                     [-0.8270506736528097, -0.6647482045160528, -0.43595102198197894,
                      -0.2930090302314022, -0.19789542449264944, -0.14560099229503182,
                      -0.12410802088624982, -0.11657014266155726, -0.1176184662051167,
                      -0.11657014266155688, -0.1241080208862487, -0.14560099229503298,
                      -0.1978954244926481, -0.2930090302313995, -0.4359510219819795,
                      -0.6647482045160534, -0.8270506736528144], 5.0e-9, 1.e-15)
        end
        @testset "Chebyshev split 1" begin
            test_input_chebyshev_split1["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split1, 3.e-2, 3.e-3,
                     [-0.8089566029756564, -0.6619132373326998, -0.4308297091398878,
                      -0.29582070389073745, -0.19344205972257697, -0.1492517215652862,
                      -0.11977564323040292, -0.12060863193204434, -0.11342083819826093,
                      -0.12060863193204283, -0.11977564323039841, -0.14925172156528543,
                      -0.1934420597225721, -0.2958207038907312, -0.43082970913989527,
                      -0.6619132373327011, -0.8089566029756454], 5.0e-9, 1.e-15)
        end
        @testset "Chebyshev split 2" begin
            test_input_chebyshev_split2["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split2, 6.e-2, 3.e-3,
                     [-0.7811451959385047, -0.6339216676848285, -0.3923659514493444,
                      -0.25872574685762867, -0.16505588951996034, -0.10734088322603245,
                      -0.0861706475981982, -0.08406228269910118, -0.08436699175046032,
                      -0.08406228269836541, -0.08617064759558782, -0.1073408832303688,
                      -0.16505588950485647, -0.2587257468721428, -0.39236595141820785,
                      -0.6339216677022307, -0.781145196037361], 5.0e-9, 1.e-15)
        end
        # The 'split 3' test is pretty badly resolved, but don't want to increase
        # run-time!
        @testset "Chebyshev split 3" begin
            test_input_chebyshev_split3["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split3, 2.5e-1, 3.e-3,
                     [-0.502249625727373, -0.4620063966241174, -0.3524200627407578,
                      -0.22448154400218082, -0.14326730277841201, -0.10324498495621048,
                      -0.08138023431925002, -0.0770230921784758, -0.07561906354506259,
                      -0.0770230921784547, -0.08138023431922943, -0.10324498495613933,
                      -0.14326730277828917, -0.2244815440020374, -0.35242006274062704,
                      -0.46200639662434545, -0.5022496257279193], 5.0e-9, 1.e-15)
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
