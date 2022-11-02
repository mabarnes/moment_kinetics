using moment_kinetics: setup_moment_kinetics, cleanup_moment_kinetics!
using moment_kinetics.time_advance: time_advance!
using moment_kinetics.communication
using moment_kinetics.looping: dimension_combinations
using Glob
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

"""
Search the source files to see if a begin_*_region() call is made for `dim_combination`

This is a bit hacky, as it searches the source files as text.
"""
function dimension_combination_is_used(dim_combination)
    search_string = "begin_"
    for d ∈ dim_combination
        search_string *= string(d, "_")
    end
    search_string *= "region"

    source_files = glob("*.jl", normpath(joinpath(@__DIR__, "..", "src")))

    return any(occursin(search_string, read(f, String)) for f ∈ source_files)
end

function runtests()

    block_size[] == 1 && error("Cannot run debug checks in serial")

    # Only need to test dimension combinations that are actually used for parallel loops
    # in some part of the code
    dimension_combinations_to_test = [c for c in dimension_combinations
                                      if dimension_combination_is_used(c)]

    @testset "$test_type" begin
        block_rank[] == 0 && println("$test_type tests")

        n_factors = length(factor(Vector, block_size[]))

        for input ∈ test_input_list, debug_loop_type ∈ dimension_combinations_to_test
            ndims = length(debug_loop_type)
            for i ∈ 1:(ndims+n_factors-1)÷n_factors
                debug_loop_parallel_dims =
                    debug_loop_type[(i-1)*n_factors+1:min(i*n_factors, ndims)]
                run_test(input, debug_loop_type, debug_loop_parallel_dims)
            end
        end
    end
end
