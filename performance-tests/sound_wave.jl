module SoundWavePerformance

include("utils.jl")
using .PerformanceTestUtils

using BenchmarkTools
using moment_kinetics
using TimerOutputs

const test_name = "sound_wave"
const benchmark_seconds = 60
const benchmark_samples = 100
const benchmark_evals = 1

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

# Useful parameters
const z_L = 1.0 # always 1 in normalized units?
const vpa_L = 8.0

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
                                    "z_IC_density_amplitude1" => 0.5,
                                    "z_IC_density_phase1" => 0.0,
                                    "z_IC_upar_amplitude1" => 0.0,
                                    "z_IC_upar_phase1" => 0.0,
                                    "z_IC_temperature_amplitude1" => 0.5,
                                    "z_IC_temperature_phase1" => π,
                                    "z_IC_density_amplitude2" => 0.5,
                                    "z_IC_density_phase2" => π,
                                    "z_IC_upar_amplitude2" => 0.0,
                                    "z_IC_upar_phase2" => 0.0,
                                    "z_IC_temperature_amplitude2" => 0.5,
                                    "z_IC_temperature_phase2" => 0.0,
                                    "charge_exchange_frequency" => 2*π*0.1,
                                    "nstep" => 100,
                                    "dt" => 0.0005,
                                    "nwrite" => 200,
                                    "use_semi_lagrange" => false,
                                    "n_rk_stages" => 4,
                                    "split_operators" => false,
                                    "z_ngrid" => 81,
                                    "z_nelement" => 1,
                                    #"z_bc" => "periodic", # only periodic option at the moment, so this is not currently used
                                    "z_discretization" => "finite_difference",
                                    "vpa_ngrid" => 241,
                                    "vpa_nelement" => 1,
                                    "vpa_L" => vpa_L,
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
                                  "z_nelement" => 10,
                                  "vpa_discretization" => "chebyshev_pseudospectral",
                                  "vpa_ngrid" => 17,
                                  "vpa_nelement" => 15))

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
Benchmark for one set of parameters

Returns
-------
[minimum time, median time, maximum time]
"""
function run_test(input)
    # Run once to check everything is compiled
    initial_input = deepcopy(input)
    initial_input["nsteps"] = 2
    run_moment_kinetics(to, input)

    result = @benchmark run_moment_kinetics($to, $input) seconds=benchmark_seconds samples=benchmark_samples evals=benchmark_evals
    println(input["run_name"])
    display(result)
    println()

    return extract_summary(result)
end

function run_tests()
    check_config()

    collected_results = Vector{Float64}(undef, 0)

    for input ∈ (test_input_finite_difference,
                 test_input_finite_difference_split_1_moment,
                 test_input_finite_difference_split_2_moments,
                 test_input_finite_difference_split_3_moments,
                 test_input_chebyshev,
                 test_input_chebyshev_split_1_moment,
                 test_input_chebyshev_split_2_moments,
                 test_input_chebyshev_split_3_moments)

        results = run_test(input)
        collected_results = vcat(collected_results, results)
    end

    upload_result(test_name, collected_results)
end

end # SoundWavePerformance

using .SoundWavePerformance

SoundWavePerformance.run_tests()
