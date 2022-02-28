"""
Provides convenience macros for shared-memory-parallel loops
"""
module looping

using ..debugging
using ..communication: _block_synchronize
using ..type_definitions: mk_int

using Combinatorics
using Primes

const all_dimensions = (:s, :r, :z, :vpa)
const dimension_combinations = Tuple(Tuple(c) for c in combinations(all_dimensions))

"""
Construct a string composed of the dimension names given in the Tuple `dims`,
separated by underscores
"""
function dims_string(dims::Tuple)
    result = string((string(d, "_") for d in dims[begin:end-1])...)
    result *= string(dims[end])
    return result
end

# Create struct to store ranges for loops over all combinations of dimensions
LoopRanges_body = quote
    parallel_dims::Tuple{Vararg{Symbol}}
    rank0::Bool
end
for dim ∈ all_dimensions
    global LoopRanges_body
    LoopRanges_body = quote
        $LoopRanges_body;
        $dim::UnitRange{mk_int}
    end
end
eval(quote
         """
         LoopRanges structs contain information on which points should be included on
         this process in loops over shared-memory arrays.

         Members
         -------
         parallel_dims::Tuple{Vararg{Symbol}}
                Indicates which dimensions are (or might be) parallelized when using
                this LoopRanges. Provided for information for developers, to make it
                easier to tell (when using a Debugger, or printing debug informatino)
                which LoopRanges instance is active in looping.loop_ranges at any point
                in the code.
         rank0::Bool
                Is this process the one with rank 0 in the 'block' which work in
                parallel on shared memory arrays.
         <d>::UnitRange{mk_int}
                Loop ranges for each dimension <d> in looping.all_dimensions.
         """
         Base.@kwdef struct LoopRanges
             $LoopRanges_body
         end
     end)

"""
Find possible divisions of sub_block_size into n factors
"""
function get_splits(sub_block_size, n)
    factors = factor(Vector, sub_block_size)
    if n == 1
        # We are splitting the last level, so must use all remaining processes
        return [[sub_block_size]]
    end
    # Iterate over all possible factors of sub_block_size (either 1 or any
    # combination of the prime factors)
    result = Vector{Vector{mk_int}}(undef, 0)
    for this_factor_parts ∈ [[1]; collect(combinations(factors))]
        this_factor = prod(this_factor_parts)
        @assert sub_block_size % this_factor == 0

        # `remaining` is the number of processes to split over the dimensions
        # still to be considered
        remaining = sub_block_size ÷ this_factor

        # Call get_splits() recursively to split `remaining` over the rest of
        # the dimensions, combining the results into a Vector of possible
        # factorizations.
        for other_factors in get_splits(remaining, n-1)
            # ignore any duplicates
            if !([this_factor; other_factors] ∈ result)
                push!(result, [this_factor; other_factors])
            end
        end
    end
    return result
end

"""
Calculate the expected load balance

'Load balance' is the ratio of the maximum and minimum numbers of points on any
process.

Arguments
---------
nprocs_list : Vector{mk_int}
    Number of processes for each dimension
sizes : Vector{mk_int}
    Size of each dimension
"""
function get_load_balance(nprocs_list, sizes)
    max_points = [ceil(s/n) for (n,s) ∈ zip(nprocs_list, sizes)]
    min_points = [floor(s/n) for (n,s) ∈ zip(nprocs_list, sizes)]
    return prod(max_points) / prod(min_points)
end

"""
Get local range of indices when splitting a loop over processes in a sub-block
"""
function get_local_range(sub_block_rank, sub_block_size, dim_size)
    # Assign either (n÷n_sub_block_procs) or (n÷n_sub_block_procs+1) points to each
    # processor, with the lower number (n÷n_sub_block_procs) on lower-number processors,
    # because the root process might have slightly more work to do in general.
    # This calculation is not at all optimized, but is not going to take long, and is
    # only done in initialization, so it is more important to be simple and robust.
    remaining = dim_size
    done = false
    n_points_for_proc = zeros(mk_int, sub_block_size)
    while !done
        for i ∈ sub_block_size:-1:1
            n_points_for_proc[i] += 1
            remaining -= 1
            if remaining == 0
                done = true
                break
            end
        end
    end

    ## An alternative way of dividing points between processes might be to minimise the
    #number of points on proc 0.
    #points_per_proc = div(dim_size, sub_block_size, RoundUp)
    #remaining = dim_size
    #n_points_for_proc = zeros(mk_int, sub_block_size)
    #for i ∈ n_procs:-1:1
    #    if remaining >= points_per_proc
    #        n_points_for_proc[i] = points_per_proc
    #        remaining -= points_per_proc
    #    else
    #        n_points_for_proc[i] = remaining
    #        remaining = 0
    #        break
    #    end
    #end
    #if remaining > 0
    #    error("not all grid points have been assigned to processes, but should have "
    #          * "been")
    #end

    # remember sub_block_rank is a 0-based index, so need to add one to get an index for
    # the n_points_for_proc Vector.
    first = 1
    for i ∈ 1:sub_block_rank
        first += n_points_for_proc[i]
    end
    last = first + n_points_for_proc[sub_block_rank + 1] - 1

    return first:last
end

"""
Find the ranges for loop variables that optimize load balance for a certain block_size
"""
function get_best_ranges(block_rank, block_size, dims, dim_sizes)
    # Find ranges for 'dims', which should be parallelized
    ranges = get_best_ranges_from_sizes(block_rank, block_size,
                                        (dim_sizes[d] for d ∈ dims))
    result = Dict(d=>r for (d,r) ∈ zip(dims, ranges))

    # Iterate over all points in ranges not being parallelized
    for d in all_dimensions
        if !(d in dims)
            result[d] = 1:dim_sizes[d]
        end
    end

    return result
end

"""
"""
function get_best_ranges_from_sizes(block_rank, block_size, dim_sizes_list)
    splits = get_splits(block_size, length(dim_sizes_list))
    load_balance = Inf
    best_split = splits[1]
    for split in splits
        this_load_balance = get_load_balance(split, dim_sizes_list)
        if this_load_balance < load_balance
            load_balance = this_load_balance
            best_split .= split
        end
    end

    # Get rank of this process in each sub-block (with sub-block sizes given by
    # best_split).
    sb_ranks = zeros(mk_int, length(dim_sizes_list))
    sub_rank = block_rank
    remaining_block_size = block_size
    # Use `Base.Iterators.reverse()` because we need to divide blocks up
    # working from right to left (slowest-varying to fastest-varying
    # dimensions)
    for (i, sb_size) in Base.Iterators.reverse(enumerate(best_split))
        remaining_block_size = remaining_block_size ÷ sb_size
        sb_ranks[i] = sub_rank ÷ remaining_block_size
        sub_rank = sub_rank % remaining_block_size
    end

    return [get_local_range(sb_rank, sb_size, dim_size)
            for (sb_rank, sb_size, dim_size) in zip(sb_ranks, best_split,
                                                    dim_sizes_list)]
end

"""
module variable that we can access by giving fully-qualified name in loop
macros
"""
const loop_ranges = Ref{LoopRanges}()
export loop_ranges

"""
module variable used to store LoopRanges that are swapped into the loop_ranges
variable in begin_*_region() functions
"""
const loop_ranges_store = Dict{Tuple{Vararg{Symbol}}, LoopRanges}()

eval(quote
         """
         Create ranges for loops with different combinations of variables

         Arguments
         ---------
         Keyword arguments `dim=n` are required for each dim in $($all_dimensions) where
         `n` is an integer giving the size of the dimension.
         """
         function setup_loop_ranges!(block_rank, block_size; dim_sizes...)
             rank0 = (block_rank == 0)

             # Use empty tuple for serial region
             if rank0
                 loop_ranges_store[()] = LoopRanges(;
                     parallel_dims=(), rank0=rank0,
                     Dict(d=>1:n for (d,n) in dim_sizes)...)
             else
                 loop_ranges_store[()] = LoopRanges(;
                     parallel_dims=(), rank0=rank0,
                     Dict(d=>1:0 for (d,_) in dim_sizes)...)
             end

             for dims ∈ dimension_combinations
                 loop_ranges_store[dims] = LoopRanges(;
                     parallel_dims=dims, rank0 = rank0,
                     get_best_ranges(block_rank, block_size, dims, dim_sizes)...)
             end

             loop_ranges[] = loop_ranges_store[()]

             return nothing
         end
     end)

export setup_loop_ranges!

# Create macros for looping over any set of dimensions
for dims ∈ dimension_combinations
    # Create an expression-function/macro combination for each level of the
    # loop
    dims_symb = Symbol(dims_string(dims))
    range_exprs =
        Tuple(:( loop_ranges[].$dim )
              for dim in dims)
    range_exprs = Tuple(Expr(:quote, r) for r in range_exprs)

    # Create a macro for the nested loop
    # Copy style from Base.Cartesian.@nloops code
    nested_macro_name = Symbol(:loop_, dims_symb)
    nested_macro_body_name = Symbol(:loop_body_, dims_symb)
    iteration_vars = Tuple(Symbol(:i, x) for x ∈ 1:length(dims))

    macro_body_expr = quote
        function $nested_macro_body_name(body, it_vars...)
            ex = Expr(:escape, body)
            # Reverse it_vars so final iteration variable is the inner loop
            for (it, range) ∈ zip(reverse(it_vars), reverse($range_exprs))
                this_range = eval(range)
                ex = quote
                    for $(esc(it)) = $this_range
                        $ex
                    end
                end
            end
            return ex
        end
    end
    eval(macro_body_expr)

    nested_macro_at_name = Symbol("@", nested_macro_name)
    macro_expr = quote
        """
        Loop over $($dims) dimensions
        """
        macro $nested_macro_name($(iteration_vars...), body)
            return $nested_macro_body_name(body, $(iteration_vars...))
        end

        export $nested_macro_at_name
    end
    eval(macro_expr)

    # Create a function for beginning regions where 'dims' are parallelized
    sync_name = Symbol(:begin_, dims_symb, :_region)
    eval(quote
             """
             Begin region in which $($dims) dimensions are parallelized by being split
             between processes.

             Calls `_block_synchronize()` to synchronize the processes operating on a
             shared-memory block, unless `no_synchronize=true` is passed as an argument.
             """
             function $sync_name(; no_synchronize::Bool=false)
                 if !no_synchronize
                     _block_synchronize()
                 end
                 loop_ranges[] = loop_ranges_store[$dims]
             end
             export $sync_name
         end)
end

"""
Run a block of code on only rank-0 of each group of processes operating on a
shared-memory block
"""
macro serial_region(blk)
    return quote
        if loop_ranges[].rank0
            $(esc(blk))
        end
    end
end
export @serial_region

"""
Begin region in which only rank-0 in each group of processes operating on a
shared-memory block operates on shared-memory arrays.

Calls `_block_synchronize()` to synchronize the processes operating on a shared-memory
block, unless `no_synchronize=true` is passed as an argument.
"""
function begin_serial_region(; no_synchronize::Bool=false)
    if !no_synchronize
        _block_synchronize()
    end
    loop_ranges[] = loop_ranges_store[()]
end
export begin_serial_region

end # looping
