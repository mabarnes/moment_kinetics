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
                                           "timestepping" => OptionsDict("nstep" => 50000,
                                                                         "dt" => 0.0001,
                                                                         "nwrite" => 250,
                                                                         "nwrite_dfns" => 250,
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
                                                                "L" => 4.0,
                                                                "bc" => "zero",
                                                                "discretization" => "finite_difference"),
                                           "vz" => OptionsDict("ngrid" => 200,
                                                               "nelement" => 1,
                                                               "L" => 4.0,
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
                                                                      "element_spacing_option" => "sqrt",
                                                                      "ngrid" => 9,
                                                                      "nelement" => 4),
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
                                                          "vpa" => OptionsDict("L" => 8.0),
                                                          "vz" => OptionsDict("L" => 8.0),
                                                         ))

"""
Run a test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, analytic_atol, expected_phi, regression_atol; args...)
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
        @test elementwise_isapprox(phi[2:end-1, end] .- offset, analytic_phi[2:end-1],
                                   rtol=0.0, atol=analytic_atol)
        @test elementwise_isapprox(phi[[1, end], end] .- offset, analytic_phi[[1, end]],
                                   rtol=0.0, atol=10.0*analytic_atol)

        # Regression test
        if expected_phi === nothing
            println("values tested would be ", phi[:,end])
        else
println("check regression inputs\n", phi[:, end], ",\n", expected_phi, ";\nrtol=", 0.0,
                                       ", atol=", regression_atol)
            @test elementwise_isapprox(phi[:, end], expected_phi, rtol=0.0,
                                       atol=regression_atol)
        end
    end
    sleep(5)
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
            run_test(test_input_chebyshev, 2.0e-2,
                     [-0.8561626565785688, -0.7744382729289455, -0.6456749856471259,
                      -0.543944201476502, -0.4625030725350743, -0.40167353996089256,
                      -0.3613603845136074, -0.3362225303320931, -0.32955001894322317,
                      -0.3029719294903252, -0.2579051663096322, -0.19581536013548473,
                      -0.1583900133597972, -0.12435799431629155, -0.11877987619307435,
                      -0.1085084216732397, -0.11499436670759786, -0.10850842167308229,
                      -0.11877987619236534, -0.12435799431777786, -0.1583900133600403,
                      -0.1958153601347357, -0.25790516631044114, -0.3029719294876854,
                      -0.32955001894395114, -0.33622253033224214, -0.3613603845143989,
                      -0.40167353996137367, -0.4625030725359137, -0.543944201476647,
                      -0.6456749856485202, -0.7744382729294272, -0.8561626565791485],
                     1.e-9)
        end
        @testset "Chebyshev split 1" begin
            test_input_chebyshev_split1["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split1, 2.5e-2,
                     [-0.8487177403047265, -0.7725315155523992, -0.643152721001601,
                      -0.543061760953187, -0.4600632228880356, -0.4010906638434458,
                      -0.35887494756369265, -0.33609558700236325, -0.3275002067692151,
                      -0.30804743010807323, -0.2542810754155979, -0.20062211561624996,
                      -0.15387704455232754, -0.12896389735419303, -0.11435148163124935,
                      -0.11318034359865542, -0.11074729810530543, -0.11318034359997875,
                      -0.11435148162808277, -0.12896389735812386, -0.15387704455923534,
                      -0.20062211560925702, -0.254281075420651, -0.30804743009098434,
                      -0.32750020676399877, -0.33609558699858677, -0.3588749475602904,
                      -0.40109066384033426, -0.46006322288592644, -0.5430617609501933,
                      -0.6431527209968522, -0.7725315155488109, -0.848717740303481],
                      1.e-9)
        end
        @testset "Chebyshev split 2" begin
            test_input_chebyshev_split2["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split2, 1.0e-1,
                     [-0.8413372024581409, -0.7674048507780572, -0.6367586336561964,
                      -0.5399777196921194, -0.4588708185942274, -0.399468434613622,
                      -0.3573836538870472, -0.33537517993056165, -0.3269880054926632,
                      -0.30456945900989546, -0.253341366434325, -0.1967564003463642,
                      -0.1512724998449133, -0.12474179447475091, -0.11250658566715303,
                      -0.1098151964026339, -0.1094051464153018, -0.1098151964026354,
                      -0.11250658566715203, -0.12474179447475178, -0.1512724998449124,
                      -0.1967564003463642, -0.25334136643432326, -0.30456945900989907,
                      -0.32698800549267304, -0.33537517993057037, -0.357383653887053,
                      -0.3994684346136149, -0.45887081859423123, -0.539977719692105,
                      -0.6367586336562191, -0.7674048507780421, -0.8413372024580487],
                     1.e-9)
        end
        # The 'split 3' test is pretty badly resolved, but don't want to increase
        # run-time!
        @testset "Chebyshev split 3" begin
            test_input_chebyshev_split3["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split3, 1.0e-1,
                     [-0.7219076194687152, -0.6957483464815538, -0.6136944587710879,
                      -0.5109831031383838, -0.4388941687165365, -0.38631978931619404,
                      -0.34519743961814303, -0.3224497211285906, -0.31436108243066035,
                      -0.2926366238365049, -0.2426420331682596, -0.1888310515408306,
                      -0.14624283589996068, -0.11948877507606734, -0.10725616503383159,
                      -0.10392183681735936, -0.103744440837122, -0.10392183681734865,
                      -0.10725616503391773, -0.11948877507604155, -0.14624283589997045,
                      -0.1888310515408156, -0.2426420331683748, -0.29263662383655503,
                      -0.3143610824308306, -0.32244972112887815, -0.3451974396177454,
                      -0.3863197893167957, -0.4388941687153804, -0.5109831031374874,
                      -0.6136944587729036, -0.6957483464819445, -0.7219076194676388],
                     1.e-9)
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
