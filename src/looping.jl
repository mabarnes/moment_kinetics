module looping
"""
Provides convenience macros for shared-memory-parallel loops
"""

using ..debugging
using ..communication: _block_synchronize
using ..type_definitions: mk_int

using Combinatorics
using Primes

@debug_loop_type_region using ..debugging: current_loop_region_type

# Export this to make sure it is available in every scope that imports looping
export @debug_loop_type_region

const all_dimensions = (:s, :z, :vpa)
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

"""
Construct names for the loop range variables

Arguments
---------
dims : Tuple
    Tuple of dimension names (all given as `Symbol`s)

Returns
-------
Dict{Symbol,Symbol}
    Dict whose keys are the dimension names, and values are the loop range
    names
"""
function loop_range_names(dims::Tuple)
    result = Dict{Symbol,Symbol}()
    loop_prefix = "$(dims_string(dims))_range_"
    for d ∈ dims
        result[d] = Symbol(string(loop_prefix, d))
    end
    return result
end

# Create struct to store ranges for loops over all combinations of dimensions
LoopRanges_body = quote
    rank0::Bool
end
setup_loop_ranges_arguments = [:( rank0=(block_rank==0) )]
for dims ∈ dimension_combinations
    global LoopRanges_body, setup_loop_ranges_arguments
    range_names = loop_range_names(dims)
    for dim ∈ dims
        LoopRanges_body = quote
            $LoopRanges_body;
            $(range_names[dim])::UnitRange{mk_int}
        end
        # best_ranges is a local variable created in setup_loop_ranges()
        push!(setup_loop_ranges_arguments,
              :( $(range_names[dim])=best_ranges[$dims][$(QuoteNode(dim))] ))
    end
end
eval(quote
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
    ranges = get_best_ranges_from_sizes(block_rank, block_size,
                                        (dim_sizes[d] for d ∈ dims))
    return Dict(d=>r for (d,r) ∈ zip(dims, ranges))
end
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

# module variable that we can access by giving fully-qualified name in loop
# macros
const loop_ranges = Ref{LoopRanges}()
export loop_ranges

#Create ranges for loops with different combinations of variables
#
#Arguments
#---------
#Keyword arguments `dim=n` are required for each dim in $all_dimensions where
#`n` is an integer giving the size of the dimension.
eval(quote
         function setup_loop_ranges!(block_rank, block_size; dim_sizes...)
             @debug_loop_type_region current_loop_region_type[] = "serial"
             best_ranges = Dict()
             for dims ∈ dimension_combinations
                 best_ranges[dims] = get_best_ranges(block_rank, block_size,
                                                     dims, dim_sizes)
             end

             loop_ranges[] = LoopRanges($(setup_loop_ranges_arguments...))
             return nothing
         end
     end)

export setup_loop_ranges!

# Create macros for looping over any set of dimensions
for dims ∈ dimension_combinations
    # Create an expression-function/macro combination for each level of the
    # loop
    dims_symb = Symbol(dims_string(dims))
    dims_symb_string = string(dims_symb)
    range_names = loop_range_names(dims)
    range_exprs =
        Tuple(:( loop_ranges[].$(range_names[dim]) )
              for dim in dims)
    range_exprs = Tuple(Expr(:quote, r) for r in range_exprs)
    for (dim, range_expr) ∈ zip(dims, range_exprs)
        macro_name = Symbol(dims_symb, :_loop_, dim)
        macro_at_name = Symbol("@", macro_name)
        one_level_expr = quote
            macro $macro_name(iteration_var, expr)
                this_range = $range_expr
                return quote
                    @debug_loop_type_region begin
                        if current_loop_region_type[] != $$dims_symb_string
                            error("Called loop of type $($$dims_symb_string) in region of "
                                  * "type $(current_loop_region_type[])")
                        end
                    end
                    for $(esc(iteration_var)) = $this_range
                        $(esc(expr))
                    end
                end
            end
            export $macro_at_name
        end
        eval(one_level_expr)
    end

    # Create a macro for the nested loop
    # Copy style from Base.Cartesian.@nloops code
    nested_macro_name =  Symbol(dims_symb, :_loop)
    nested_macro_body_name =  Symbol(dims_symb, :_loop_body)
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
            return quote
                @debug_loop_type_region begin
                    if current_loop_region_type[] != $$dims_symb_string
                        error("Called loop of type $($$dims_symb_string) in region of type "
                              * "$(current_loop_region_type[])")
                    end
                end
                $ex
            end
        end
    end
    eval(macro_body_expr)

    nested_macro_at_name = Symbol("@", nested_macro_name)
    macro_expr = quote
        macro $nested_macro_name($(iteration_vars...), body)
            return $nested_macro_body_name(body, $(iteration_vars...))
        end

        export $nested_macro_at_name
    end
    eval(macro_expr)

    # Create a function for beginning loops of type 'dims'
    sync_name = Symbol(:begin_, dims_symb, :_region)
    eval(quote
             function $sync_name(; no_synchronize::Bool=false)
                 @debug_loop_type_region begin
                     if current_loop_region_type[] == $dims_symb_string
                         error("Called $($sync_name)(), but already in region of type "
                               * "$(current_loop_region_type[]).")
                     end
                     current_loop_region_type[] = $dims_symb_string
                 end
                 if !no_synchronize
                     _block_synchronize()
                 end
             end
             export $sync_name
         end)
end

"""
Run a block of code on only rank-0 of each block
"""
macro serial_region(blk)
    return quote
        @debug_loop_type_region begin
            if current_loop_region_type[] != "serial"
                error("Called loop of type serial in region of type "
                      * "$(current_loop_region_type[])")
            end
            current_loop_region_type[] = "serial"
        end
        if loop_ranges[].rank0
            $(esc(blk))
        end
    end
end
export @serial_region
function begin_serial_region(; no_synchronize::Bool=false)
    @debug_loop_type_region begin
        if current_loop_region_type[] == "serial"
            error("Called begin_serial_region(), but already in region of type serial.")
        end
        current_loop_region_type[] = "serial"
    end
    if !no_synchronize
        _block_synchronize()
    end
end
export begin_serial_region

end # looping
