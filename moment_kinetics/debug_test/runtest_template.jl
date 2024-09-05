using moment_kinetics: setup_moment_kinetics, cleanup_moment_kinetics!
using moment_kinetics.time_advance: time_advance!
using moment_kinetics.communication
using moment_kinetics.looping: all_dimensions, dimension_combinations,
                               anyv_dimension_combinations
using moment_kinetics.Glob
using moment_kinetics.Primes

"""
Run a test for a single set of parameters
"""
# Note 'name' should not be shared by any two tests in this file
function run_test(test_input, debug_loop_type, debug_loop_parallel_dims; restart=false)

    name = test_input["run_name"]

    @testset "$name" begin
        # Provide some progress info
        global_rank[] == 0 && println("    - bug-checking $name, $debug_loop_type, $debug_loop_parallel_dims")

        # run simulation
        mk_state = setup_moment_kinetics(test_input; debug_loop_type=debug_loop_type,
                                         debug_loop_parallel_dims=debug_loop_parallel_dims,
                                         restart=restart)
        time_advance!(mk_state...)
        cleanup_moment_kinetics!(mk_state[end-2:end]...)
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

function runtests(; restart=false)

    global_size[] == 1 && error("Cannot run debug checks in serial")

    # Only need to test dimension combinations that are actually used for parallel loops
    # in some part of the code
    dimension_combinations_to_test = [c for c in tuple(dimension_combinations...,
                                                       anyv_dimension_combinations...)
                                      if dimension_combination_is_used(c)]

    @testset "$test_type" begin
        global_rank[] == 0 && println("$test_type tests")

        n_factors = length(factor(Vector, global_size[]))

        for input ∈ test_input_list, debug_loop_type ∈ dimension_combinations_to_test
            if :sn ∈ debug_loop_type && "n_neutral_species" ∈ keys(input) &&
                    input["n_neutral_species"] <= 0
                # Skip neutral dimension parallelisation options if the number of neutral
                # species is zero, as these would just be equivalent to running in serial
                continue
            end

            if :anyv ∈ debug_loop_type
                dims_to_test = debug_loop_type[2:end]
            else
                dims_to_test = debug_loop_type
            end
            for d ∈ all_dimensions
                nelement_name = "$(d)_nelement"
                if nelement_name ∈ keys(input)
                    nelement = input[nelement_name]
                elseif d ∈ (:vperp, :vzeta, :vr)
                    nelement = 1
                else
                    # Dummy value, here it only matters if this is 1 or greater than 1
                    nelement = 2
                end

                ngrid_name = "$(d)_ngrid"
                if ngrid_name ∈ keys(input)
                    ngrid = input[ngrid_name]
                elseif d ∈ (:vperp, :vzeta, :vr)
                    ngrid = 1
                else
                    # Dummy value, here it only matters if this is 1 or greater than 1
                    ngrid = 2
                end

                if nelement == 1 && ngrid == 1
                    # Dimension has only one point, so cannot be parallelised - no need to
                    # test
                    dims_to_test = Tuple(x for x ∈ dims_to_test if x != d)
                end
            end

            ndims = length(dims_to_test)
            if ndims == 0
                # No dimensions to test here
                continue
            end
            for i ∈ 1:(ndims+n_factors-1)÷n_factors
                debug_loop_parallel_dims =
                    dims_to_test[(i-1)*n_factors+1:min(i*n_factors, ndims)]
                run_test(input, debug_loop_type, debug_loop_parallel_dims; restart=restart)
            end
        end
    end
end
