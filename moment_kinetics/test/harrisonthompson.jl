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

ionization_frequency = sqrt(2) * 0.688

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
fnegative(phi, z, R_ion) = - z - 2.0 / π * sqrt(2) / R_ion * dawson(sqrt(-phi))
fpositive(phi, z, R_ion) = - z + 2.0 / π * sqrt(2) / R_ion * dawson(sqrt(-phi))
# derivative of f for Newton iteration
fprimenegative(phi, z, R_ion) = 1.0 / π * sqrt(2) / R_ion / sqrt(-phi) * Dprime(sqrt(-phi))
fprimepositive(phi, z, R_ion) = - 1.0 / π * sqrt(2) / R_ion / sqrt(-phi) * Dprime(sqrt(-phi))
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
                                                                          "initial_temperature" => 0.3333333333333333),
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
                                                                           "pressure" => false,
                                                                           "moments_conservation" => false),
                                           "reactions" => OptionsDict("charge_exchange_frequency" => 0.0,
                                                                      "ionization_frequency" => 0.0),
                                           "timestepping" => OptionsDict("nstep" => 25000,
                                                                         "dt" => 0.0001414213562373095,
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
                                                                "L" => 5.656854249492381,
                                                                "bc" => "zero",
                                                                "discretization" => "finite_difference"),
                                           "vz" => OptionsDict("ngrid" => 200,
                                                               "nelement" => 1,
                                                               "L" => 5.656854249492381,
                                                               "bc" => "zero",
                                                               "discretization" => "finite_difference"),
                                           "ion_source_1" => OptionsDict("active" => true,
                                                                         "source_strength" => ionization_frequency,
                                                                         "source_T" => 0.25,
                                                                         "z_profile" => "constant",
                                                                         "r_profile" => "constant"))

test_input_chebyshev = recursive_merge(test_input_finite_difference,
                                       OptionsDict("output" => OptionsDict("run_name" => "chebyshev_pseudospectral"),
                                                   "z" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                      "element_spacing_option" => "sqrt",
                                                                      "ngrid" => 9,
                                                                      "nelement" => 4),
                                                   "vpa" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                        "ngrid" => 5,
                                                                        "nelement" => 40),
                                                   "vz" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                       "ngrid" => 5,
                                                                       "nelement" => 40),
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
                                                          "evolve_moments" => OptionsDict("pressure" => true),
                                                          "vpa" => OptionsDict("L" => 13.856406460551018),
                                                          "vz" => OptionsDict("L" => 13.856406460551018),
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
            @test elementwise_isapprox(phi[:, end], expected_phi, rtol=0.0,
                                       atol=regression_atol)
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
            run_test(test_input_chebyshev, 2.5e-2,
                     [-0.8545358193502869, -0.774407964023825, -0.6480964317972056,
                      -0.5437138090271466, -0.46526043489110663, -0.40062308723846,
                      -0.3642888155225684, -0.336194221384609, -0.3309510526846082,
                      -0.2965979559778681, -0.2619290551264873, -0.1899520777383056,
                      -0.1640083179440505, -0.11857363623485638, -0.12466820204956786,
                      -0.10270584477435529, -0.12038738056547979, -0.10270584478046563,
                      -0.12466820202523383, -0.11857363627205779, -0.16400831794981355,
                      -0.18995207771777678, -0.261929055133569, -0.29659795589271487,
                      -0.3309510526748036, -0.3361942213575321, -0.36428881552364273,
                      -0.4006230872120763, -0.4652604348890475, -0.5437138089946042,
                      -0.6480964317876803, -0.7744079639913641, -0.8545358193124721],
                     1.e-9)
        end
        @testset "Chebyshev split 1" begin
            test_input_chebyshev_split1["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split1, 2.5e-2,
                     [-0.8495373016990986, -0.7730604053595787, -0.6423237683807164,
                      -0.5439838216903743, -0.4601748431340267, -0.40107529909456163,
                      -0.3590633781966198, -0.335438070893098, -0.3271420925590884,
                      -0.30730823129176493, -0.2544312962149009, -0.19976775303478753,
                      -0.15431051081815625, -0.12791243078714176, -0.11474579819009974,
                      -0.11198431469501578, -0.11075845437638254, -0.1119843147009454,
                      -0.11474579820860024, -0.12791243078511672, -0.15431051083388295,
                      -0.19976775306436434, -0.25443129621627336, -0.30730823124375756,
                      -0.3271420925272819, -0.3354380708649482, -0.35906337818877165,
                      -0.401075299058729, -0.4601748431161183, -0.5439838216699101,
                      -0.64232376834831, -0.7730604053419816, -0.8495373016743037],
                     1.e-9)
        end
        @testset "Chebyshev split 2" begin
            test_input_chebyshev_split2["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split2, 1.0e-1,
                     [-0.8342783085259872, -0.7698792713448214, -0.641983385770519,
                      -0.5380389205867865, -0.45889924402177823, -0.3980719689550363,
                      -0.35765986685938084, -0.33397132961790477, -0.3264511138414644,
                      -0.3052038493444245, -0.2546499290646862, -0.19973933586755266,
                      -0.15432637421419862, -0.12607330139370168, -0.11448165922261166,
                      -0.11187911808392619, -0.11151513822962328, -0.11187911808392792,
                      -0.11448165922261092, -0.12607330139370018, -0.15432637421419812,
                      -0.19973933586755266, -0.2546499290646851, -0.30520384934442285,
                      -0.3264511138414635, -0.33397132961790366, -0.35765986685937895,
                      -0.3980719689550344, -0.4588992440217779, -0.5380389205867854,
                      -0.6419833857705195, -0.7698792713448228, -0.8342783085259872],
                     1.e-9)
        end
        # The 'split 3' test is pretty badly resolved, but don't want to increase
        # run-time!
        @testset "Chebyshev split 3" begin
            test_input_chebyshev_split3["output"]["base_directory"] = test_output_directory
            run_test(test_input_chebyshev_split3, 1.0e-1,
                     [-0.7232327624770025, -0.6952994484016762, -0.6131215101209814,
                      -0.511050013472119, -0.43761418396356366, -0.3857139266380638,
                      -0.3443541011175761, -0.3216605855298341, -0.31393132777391625,
                      -0.2923512897025766, -0.24310216823942493, -0.18941916637412903,
                      -0.14639650026886553, -0.1191242492224155, -0.10670767712859247,
                      -0.10384336441914938, -0.10363768779825491, -0.10384336441914938,
                      -0.10670767712859247, -0.119124249222415, -0.14639650026886605,
                      -0.18941916637412903, -0.24310216823942238, -0.29235128970257573,
                      -0.3139313277739178, -0.32166058552983506, -0.344354101117577,
                      -0.3857139266380658, -0.4376141839635654, -0.5110500134721212,
                      -0.613121510120985, -0.6952994484016789, -0.723232762477006],
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
