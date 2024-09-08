module SoundWavePerformance

include("utils.jl")
using .PerformanceTestUtils
using moment_kinetics.utils: recursive_merge

const test_name = "sound_wave"

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

# Useful parameters
const z_L = 1.0 # always 1 in normalized units?
const vpa_L = 8.0

# default inputs for tests
test_input_finite_difference = OptionsDict("composition" => OptionsDict("n_ion_species" => 1,
                                                                        "n_neutral_species" => 1,
                                                                        "boltzmann_electron_response" => true,
                                                                        "T_e" => 1.0),
                                           "run_name" => "finite_difference",
                                           "base_directory" => test_output_directory,
                                           "evolve_moments" => OptionsDict("density" => false,
                                                                           "parallel_flow" => false,
                                                                           "parallel_pressure" => false,
                                                                           "moments_conservation" => true),
                                           "ion_species_1" => OptionsDict("initial_density" => 0.5,
                                                                          "initial_temperature" => 1.0),
                                           "neutral_species_1" => OptionsDict("initial_density" => 0.5,
                                                                              "initial_temperature" => 1.0),
                                           "z_IC_ion_species_1" => OptionsDict("initialization_option" => "sinusoid",
                                                                               "z_IC_density_amplitude" => 0.5,
                                                                               "z_IC_density_phase" => 0.0,
                                                                               "z_IC_upar_amplitude" => 0.0,
                                                                               "z_IC_upar_phase" => 0.0,
                                                                               "z_IC_temperature_amplitude" => 0.5,
                                                                               "z_IC_temperature_phase" => Float64(π)),
                                           "z_IC_neutral_species_1" => OptionsDict("initialization_option" => "sinusoid",
                                                                                   "density_amplitude" => 0.5,
                                                                                   "density_phase" => Float64(π),
                                                                                   "upar_amplitude" => 0.0,
                                                                                   "upar_phase" => 0.0,
                                                                                   "temperature_amplitude" => 0.5,
                                                                                   "temperature_phase" => 0.0),
                                           "reactions" => OptionsDict("charge_exchange_frequency" => 2*Float64(π)*0.1,
                                                                      "ionization_frequency" => 0.0),
                                           "timestepping" => OptionsDict( "nstep" => 100,
                                                                         "dt" => 0.0005,
                                                                         "nwrite" => 200,
                                                                         "use_semi_lagrange" => false,
                                                                         "n_rk_stages" => 4,
                                                                         "split_operators" => false),
                                           "r" => OptionsDict("ngrid" => 1,
                                                              "nelement" => 1),
                                           "z" => OptionsDict("ngrid" => 81,
                                                              "nelement" => 1,
                                                              "bc" => "periodic",
                                                              "discretization" => "finite_difference"),
                                           "vpa" => OptionsDict("ngrid" => 241,
                                                                "nelement" => 1,
                                                                "L" => vpa_L,
                                                                "bc" => "periodic",
                                                                "discretization" => "finite_difference"),
                                           "vz" => OptionsDict("ngrid" => 241,
                                                               "nelement" => 1,
                                                               "L" => vpa_L,
                                                               "bc" => "periodic",
                                                               "discretization" => "finite_difference"),
                                          )

test_input_finite_difference_split_1_moment =
    merge(test_input_finite_difference,
          OptionsDict("run_name" => "finite_difference_split_1_moment",
                      "evolve_moments_density" => true))

test_input_finite_difference_split_2_moments =
    merge(test_input_finite_difference_split_1_moment,
          OptionsDict("run_name" => "finite_difference_split_2_moments",
                      "evolve_moments_parallel_flow" => true))

test_input_finite_difference_split_3_moments =
    merge(test_input_finite_difference_split_2_moments,
          OptionsDict("run_name" => "finite_difference_split_3_moments",
                      "evolve_moments_parallel_pressure" => true))

test_input_chebyshev = recursive_merge(test_input_finite_difference,
                                       OptionsDict("run_name" => "chebyshev_pseudospectral",
                                                   "z" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                      "ngrid" => 9,
                                                                      "nelement" => 10),
                                                   "vpa" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                        "ngrid" => 17,
                                                                        "nelement" => 15),
                                                   "vz" => OptionsDict("discretization" => "chebyshev_pseudospectral",
                                                                       "ngrid" => 17,
                                                                       "nelement" => 15),
                                                  ))

test_input_chebyshev_split_1_moment =
    merge(test_input_chebyshev,
          OptionsDict("run_name" => "chebyshev_pseudospectral_split_1_moment",
                      "evolve_moments_density" => true))

test_input_chebyshev_split_2_moments =
    merge(test_input_chebyshev_split_1_moment,
          OptionsDict("run_name" => "chebyshev_pseudospectral_split_2_moments",
                      "evolve_moments_parallel_flow" => true))

test_input_chebyshev_split_3_moments =
    merge(test_input_chebyshev_split_2_moments,
          OptionsDict("run_name" => "chebyshev_pseudospectral_split_3_moments",
                      "evolve_moments_parallel_pressure" => true))

inputs_list = (test_input_finite_difference,
               test_input_finite_difference_split_1_moment,
               test_input_finite_difference_split_2_moments,
               test_input_finite_difference_split_3_moments,
               test_input_chebyshev,
               test_input_chebyshev_split_1_moment,
               test_input_chebyshev_split_2_moments,
               test_input_chebyshev_split_3_moments)

function run_tests()
    check_config()

    collected_initialization_results = Vector{Float64}(undef, 0)
    collected_results = Vector{Float64}(undef, 0)

    for input ∈ inputs_list

        (initialization_results, results) = run_test(input)
        collected_initialization_results  = vcat(collected_initialization_results,
                                                 initialization_results)
        collected_results = vcat(collected_results, results)
    end

    upload_result(test_name, collected_initialization_results, collected_results)
end

end # SoundWavePerformance

using .SoundWavePerformance

if abspath(PROGRAM_FILE) == @__FILE__
    SoundWavePerformance.run_tests()
end
