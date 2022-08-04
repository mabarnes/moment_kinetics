module SoundWaveDebug

include("setup.jl")

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
                                    "z_IC_option1" => "sinusoid",
                                    "z_IC_density_amplitude1" => 0.001,
                                    "z_IC_density_phase1" => 0.0,
                                    "z_IC_upar_amplitude1" => 0.0,
                                    "z_IC_upar_phase1" => 0.0,
                                    "z_IC_temperature_amplitude1" => 0.0,
                                    "z_IC_temperature_phase1" => 0.0,
                                    "z_IC_option2" => "sinusoid",
                                    "z_IC_density_amplitude2" => 0.001,
                                    "z_IC_density_phase2" => 0.0,
                                    "z_IC_upar_amplitude2" => 0.0,
                                    "z_IC_upar_phase2" => 0.0,
                                    "z_IC_temperature_amplitude2" => 0.0,
                                    "z_IC_temperature_phase2" => 0.0,
                                    "charge_exchange_frequency" => 2*π*0.1,
                                    "ionization_frequency" => 0.0,
                                    "nstep" => 3,
                                    "dt" => 0.002,
                                    "nwrite" => 2,
                                    "use_semi_lagrange" => false,
                                    "n_rk_stages" => 4,
                                    "split_operators" => false,
                                    "z_ngrid" => 8,
                                    "z_nelement" => 1,
                                    "z_bc" => "periodic",
                                    "z_discretization" => "finite_difference",
                                    "vpa_ngrid" => 8,
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
                                  "z_ngrid" => 3,
                                  "z_nelement" => 2,
                                  "vpa_discretization" => "chebyshev_pseudospectral",
                                  "vpa_ngrid" => 3,
                                  "vpa_nelement" => 2))

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
               "evolve_moments_parallel_pressure" => true,
               "runtime_plots" => true))


"""
Run a sound-wave test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input; args...)
    # by passing keyword arguments to run_test, args becomes a Dict which can be used to
    # update the default inputs

    # Convert keyword arguments to a unique name
    name = test_input["run_name"]
    if length(args) > 0
        name = string(name, "_", (string(k, "-", v, "_") for (k, v) in args)...)

        # Remove trailing "_"
        name = chop(name)
    end

    @testset "$name" begin
        # Provide some progress info
        println("    - bug-checking ", name)

        # Convert dict from symbol keys to String keys
        modified_inputs = Dict(String(k) => v for (k, v) in args)

        # Update default inputs with values to be changed
        input = merge(test_input, modified_inputs)

        input["run_name"] = name

        # run simulation
        run_moment_kinetics(input)
    end
end

function runtests()
    @testset "sound wave" begin
        println("sound wave tests")

        @testset "finite difference" begin
            run_test(test_input_finite_difference)
            run_test(test_input_finite_difference_split_1_moment)
            run_test(test_input_finite_difference_split_2_moments)
            run_test(test_input_finite_difference_split_3_moments)
        end

        @testset "finite difference, CX=0" begin
            run_test(test_input_finite_difference; charge_exchange_frequency=0.0)
            run_test(test_input_finite_difference_split_1_moment; charge_exchange_frequency=0.0)
            run_test(test_input_finite_difference_split_2_moments; charge_exchange_frequency=0.0)
            run_test(test_input_finite_difference_split_3_moments; charge_exchange_frequency=0.0)
        end

        @testset "Chebyshev" begin
            run_test(test_input_chebyshev)
            run_test(test_input_chebyshev_split_1_moment)
            run_test(test_input_chebyshev_split_2_moments)
            run_test(test_input_chebyshev_split_3_moments)
        end

        @testset "Chebyshev, CX=0" begin
            run_test(test_input_chebyshev; charge_exchange_frequency=0.0)
            run_test(test_input_chebyshev_split_1_moment; charge_exchange_frequency=0.0)
            run_test(test_input_chebyshev_split_2_moments; charge_exchange_frequency=0.0)
            run_test(test_input_chebyshev_split_3_moments; charge_exchange_frequency=0.0)
        end
    end
end

end # SoundWaveDebug


using .SoundWaveDebug

SoundWaveDebug.runtests()
