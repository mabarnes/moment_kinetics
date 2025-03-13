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
                                           "timestepping" => OptionsDict("nstep" => 25000,
                                                                         "dt" => 0.0002,
                                                                         "nwrite" => 25000,
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
                     [-0.8484562332448123, -0.7731594122896942, -0.6487532957897997,
                      -0.5397220258399424, -0.46578988754057354, -0.40327113838418693,
                      -0.35534532296389804, -0.33494154315926705, -0.329698003951445,
                      -0.30781941001846524, -0.2579358071349164, -0.19661808802345968,
                      -0.15875294730559114, -0.1252978062275556, -0.11776834295559209,
                      -0.10961445470905723, -0.11414995800977813, -0.10961445470905623,
                      -0.11776834295559234, -0.1252978062275542, -0.1587529473055918,
                      -0.19661808802345995, -0.2579358071349154, -0.30781941001846447,
                      -0.329698003951445, -0.33494154315926655, -0.35534532296389676,
                      -0.4032711383841856, -0.46578988754057266, -0.539722025839942,
                      -0.6487532957897987, -0.7731594122896924, -0.8484562332448115],
                     1.e-9)
        end
        @testset "Chebyshev split 1" begin
            test_input_chebyshev_split1["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split1, 2.5e-2,
                     [-0.8431127490951434, -0.7686953841398305, -0.6451298026492286,
                      -0.5407183270449003, -0.4599775273225123, -0.40322765384203124,
                      -0.35452512726676005, -0.3350896982157813, -0.32758998329845596,
                      -0.31111487633098417, -0.2548142056681187, -0.20076375700784066,
                      -0.15454780720609576, -0.129287046815988, -0.11399975195041435,
                      -0.11365625611173534, -0.11055259899001796, -0.11365625611173683,
                      -0.11399975195041585, -0.1292870468159895, -0.1545478072060942,
                      -0.2007637570078423, -0.2548142056681187, -0.31111487633098506,
                      -0.32758998329845823, -0.33508969821578377, -0.35452512726676294,
                      -0.40322765384203324, -0.4599775273225144, -0.5407183270449015,
                      -0.6451298026492299, -0.7686953841398305, -0.8431127490951434],
                      1.e-9)
        end
        @testset "Chebyshev split 2" begin
            test_input_chebyshev_split2["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split2, 1.0e-1,
                     [-0.8829328466990832, -0.7365169413557857, -0.5377733644954565,
                      -0.48755530264592656, -0.4348367840317052, -0.3601098595646277,
                      -0.31196240043772105, -0.30663178380280753, -0.2888329436807675,
                      -0.25964771352309535, -0.21748207691570984, -0.16450782175953407,
                      -0.11465648800185961, -0.08495645353217472, -0.07937358574665171,
                      -0.08262510970065341, -0.07915814020977485, -0.08262509485795091,
                      -0.07937358999564623, -0.08495644982438025, -0.11465649459239113,
                      -0.16450782894341184, -0.21748213843463834, -0.2596476466978498,
                      -0.28883329112855405, -0.3066320322001379, -0.31196248218973943,
                      -0.36010950560758653, -0.43483667144077187, -0.4875563931709269,
                      -0.537772517077348, -0.7365144685753009, -0.8829308955446193],
                     1.e-9)
        end
        # The 'split 3' test is pretty badly resolved, but don't want to increase
        # run-time!
        @testset "Chebyshev split 3" begin
            test_input_chebyshev_split3["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split3, 1.0e-1,
                     [-0.7219962663261933, -0.6957675775077491, -0.6136670218396234,
                      -0.5109865776549909, -0.4388919351296931, -0.38631162363321925,
                      -0.3451995460684685, -0.32245455435499354, -0.3143618682574149,
                      -0.29263611133097844, -0.24264133864878434, -0.1888321489656518,
                      -0.1462437722304863, -0.11948969885455997, -0.10725762202667029,
                      -0.10392176291388204, -0.10374409909008694, -0.10392176291388917,
                      -0.10725762202666311, -0.11948969885456598, -0.1462437722304755,
                      -0.18883214896567382, -0.24264133864880474, -0.2926361113310414,
                      -0.3143618682575076, -0.32245455435501763, -0.3451995460684878,
                      -0.38631162363321825, -0.43889193512975544, -0.5109865776549724,
                      -0.6136670218397087, -0.695767577507281, -0.7219962663249525],
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
