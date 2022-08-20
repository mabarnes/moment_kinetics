module HarrisonThompsonDebug
# Test case with constant-in-space, delta-function-in-vpa source against
# analytic solution from [E. R. Harrison and W. B. Thompson. The low pressure
# plane symmetric discharge. Proc. Phys. Soc., 74:145, 1959]

include("setup.jl")

# Create a temporary directory for test output
test_output_directory = tempname()
mkpath(test_output_directory)

# default inputs for tests
test_input_finite_difference = Dict("n_ion_species" => 2,
                                    "n_neutral_species" => 2,
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
                                    #"ionization_frequency" => 0.25,
                                    "ionization_frequency" => 0.688,
                                    "constant_ionization_rate" => true,
                                    "nstep" => 3,
                                    "dt" => 0.0005,
                                    "nwrite" => 2,
                                    "use_semi_lagrange" => false,
                                    "n_rk_stages" => 4,
                                    "split_operators" => false,
                                    "r_ngrid" => 4,
                                    "r_nelement" => 1,
                                    "r_discretization" => "finite_difference",
                                    "z_ngrid" => 8,
                                    "z_nelement" => 1,
                                    "z_bc" => "periodic",
                                    "z_discretization" => "finite_difference",
                                    "vperp_ngrid" => 4,
                                    "vperp_nelement" => 1,
                                    "vperp_discretization" => "finite_difference",
                                    "vpa_ngrid" => 8,
                                    "vpa_nelement" => 1,
                                    "vpa_L" => 8.0,
                                    "vpa_bc" => "periodic",
                                    "vpa_discretization" => "finite_difference",
                                    "vz_ngrid" => 4,
                                    "vz_nelement" => 1,
                                    "vz_discretization" => "finite_difference",
                                    "vr_ngrid" => 4,
                                    "vr_nelement" => 1,
                                    "vr_discretization" => "finite_difference",
                                    "vzeta_ngrid" => 4,
                                    "vzeta_nelement" => 1,
                                    "vzeta_discretization" => "finite_difference")

test_input_chebyshev = merge(test_input_finite_difference,
                             Dict("run_name" => "chebyshev_pseudospectral",
                                  "r_discretization" => "chebyshev_pseudospectral",
                                  "r_ngrid" => 3,
                                  "r_nelement" => 1,
                                  "z_discretization" => "chebyshev_pseudospectral",
                                  "z_ngrid" => 3,
                                  "z_nelement" => 2,
                                  "vperp_discretization" => "chebyshev_pseudospectral",
                                  "vperp_ngrid" => 3,
                                  "vperp_nelement" => 1,
                                  "vpa_discretization" => "chebyshev_pseudospectral",
                                  "vpa_ngrid" => 3,
                                  "vpa_nelement" => 2,
                                  "vz_discretization" => "chebyshev_pseudospectral",
                                  "vz_ngrid" => 3,
                                  "vz_nelement" => 2,
                                  "vr_discretization" => "chebyshev_pseudospectral",
                                  "vr_ngrid" => 3,
                                  "vr_nelement" => 1,
                                  "vzeta_discretization" => "chebyshev_pseudospectral",
                                  "vzeta_ngrid" => 3,
                                  "vzeta_nelement" => 1))

"""
Run a test for a single set of parameters
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

function runtests()
    @testset "Harrison-Thompson" begin
        println("Harrison-Thompson wall boundary condition tests")

        #@testset "finite difference" begin
        #    run_test(test_input_finite_difference)
        #end

        @testset "Chebyshev" begin
            run_test(test_input_chebyshev)
        end
    end
end

end # HarrisonThompsonDebug


using .HarrisonThompsonDebug

HarrisonThompsonDebug.runtests()
