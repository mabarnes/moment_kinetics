using moment_kinetics: setup_moment_kinetics, cleanup_moment_kinetics!
using moment_kinetics.time_advance: time_advance!
using moment_kinetics.communication
using moment_kinetics.looping: dimension_combinations
using Primes

"""
Run a test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, debug_loop_type, debug_loop_parallel_dims)

    name = test_input["run_name"]

    @testset "$name" begin
        # Provide some progress info
        block_rank[] == 0 && println("    - bug-checking $name, $debug_loop_type, $debug_loop_parallel_dims")

        # run simulation
        mk_state = setup_moment_kinetics(test_input; debug_loop_type=debug_loop_type,
                                         debug_loop_parallel_dims=debug_loop_parallel_dims)
        time_advance!(mk_state...)
        cleanup_moment_kinetics!(mk_state[end-1:end]...)
    end
end

function runtests()

    block_size[] == 1 && error("Cannot run debug checks in serial")

    @testset "$test_type" begin
        block_rank[] == 0 && println("$test_type tests")

        n_factors = length(factor(Vector, block_size[]))

        for input ∈ test_input_list, debug_loop_type ∈ dimension_combinations
            ndims = length(debug_loop_type)
            for i ∈ 1:(ndims+n_factors-1)÷n_factors
                debug_loop_parallel_dims =
                    debug_loop_type[(i-1)*n_factors+1:min(i*n_factors, ndims)]
                run_test(input, debug_loop_type, debug_loop_parallel_dims)
            end
        end
    end
end
