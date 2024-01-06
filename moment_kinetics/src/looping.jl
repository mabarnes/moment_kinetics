"""
Provides convenience macros for shared-memory-parallel loops
"""
module looping

using ..debugging
using ..communication: _block_synchronize, _anyv_subblock_synchronize, comm_block,
                       comm_anyv_subblock, anyv_subblock_rank, anyv_subblock_size,
                       anyv_isubblock_index, anyv_nsubblocks_per_block
using ..type_definitions: mk_int

using Combinatorics
using MPI
using Primes

# The ion dimensions and neutral dimensions are separated in order to restrict the
# supported parallel loop types to correct combinations. This also reduces the number
# of combinations - for some of the debugging features this helps.
const ion_dimensions = (:s, :r, :z, :vperp, :vpa)
const neutral_dimensions = (:sn, :r, :z, :vzeta, :vr, :vz)
const all_dimensions = unique((ion_dimensions..., neutral_dimensions...))
const dimension_combinations = Tuple(Tuple(c) for c in
                                     unique((combinations(ion_dimensions)...,
                                             combinations(neutral_dimensions)...)))
const anyv_dimension_combinations = ((:anyv,), (:anyv, :vperp), (:anyv, :vpa),
                                     (:anyv, :vperp, :vpa))

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
    is_anyv::Bool
    anyv_rank0::Bool
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

    # Add an 'epsilon' to the denominator to prevent the division giving `Inf`.
    # If `prod(min_points) == Inf` for every combinations of process numbers per dimension
    # then the parallelisation is not very efficient for the particular combination of
    # dimensios being considered (as some processes have no work) but including the
    # 'epsilon' should allow choosing the least-worst option (?), or at least make the
    # choice of process splitting less random than finding the minimum of a vector of
    # numbers that are all `Inf`.
    return prod(max_points) / (prod(min_points) + 1e-14)
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

    if dim_size == 0
        # No processor includes a grid point
        return 1:0
    end

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
    dim_sizes_list = (dim_sizes[d] for d ∈ dims)
    best_split = get_best_split_from_sizes(block_size, dim_sizes_list)
    ranges = get_ranges_from_split(block_rank, block_size, best_split, dim_sizes_list)
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
function get_splits_and_load_balances_from_sizes(block_size, dim_sizes_list)
    splits = get_splits(block_size, length(dim_sizes_list))
    @debug_detect_redundant_block_synchronize begin
        if any(dim_sizes_list .== 1)
            println("dim sizes ", dim_sizes_list)
            error("Some dimension has size 1. Does not make sense to use "
                  * "@debug_detect_redundant_block_synchronize as not all dimensions "
                  * "can be split over more than one process.")
        end
        # This debug mode requires all dimensions to be split
        filtered_splits = Vector{Vector{mk_int}}(undef, 0)
        for s in splits
            if !any(s .== 1)
                # Every dimension is split, so detection of redundant
                # _block_synchronize() calls can be done.
                push!(filtered_splits, s)
            end
        end
        if isempty(filtered_splits)
            error("No combinations of factors of `block_size` resulted in all "
                  * "dimensions being split. Probably need to use a number of "
                  * "processes with more prime factors.")
        end
        splits = filtered_splits
    end

    load_balances = [get_load_balance(s, dim_sizes_list) for s ∈ splits]

    return splits, load_balances
end

function get_best_split_from_sizes(block_size, dim_sizes_list)
    splits, load_balances =
        get_splits_and_load_balances_from_sizes(block_size, dim_sizes_list)

    best_index = argmin(load_balances)
    return splits[best_index]
end

"""
"""
function get_ranges_from_split(block_rank, block_size, split, dim_sizes_list)

    # Get rank of this process in each sub-block (with sub-block sizes given by split).
    sb_ranks = zeros(mk_int, length(dim_sizes_list))
    sub_rank = block_rank
    remaining_block_size = block_size
    # Use `Base.Iterators.reverse()` because we need to divide blocks up
    # working from right to left (slowest-varying to fastest-varying
    # dimensions)
    for (i, sb_size) in Base.Iterators.reverse(enumerate(split))
        remaining_block_size = remaining_block_size ÷ sb_size
        sb_ranks[i] = sub_rank ÷ remaining_block_size
        sub_rank = sub_rank % remaining_block_size
    end

    return [get_local_range(sb_rank, sb_size, dim_size)
            for (sb_rank, sb_size, dim_size) in zip(sb_ranks, split, dim_sizes_list)]
end

"""
Find the numbers of processes for each dimension that optimize load balance for 'anyv'
type loops for a certain block_size.

The 'anyv' parallelisation patterns are designed for use in the collision operator. They
all share the same parallelisation in species and spatial dimensions so that the region
type can be switched between 'anyv' types within a loop over species and spatial
dimensions (@loop_s_r_z). It is only defined for ions, not for neutrals.

Parts of the collision operator cannot conveniently be parallelised over velocity
dimensions, so this function aims to assign as much parallelism as possible to the species
and spatial dimensions.
"""
function get_best_anyv_split(block_size, dim_sizes)

    spatial_vperp_dim_sizes_list = (dim_sizes[d] for d ∈ (:s, :r, :z, :vperp))
    vperp_splits, vperp_load_balances =
        get_splits_and_load_balances_from_sizes(block_size, spatial_vperp_dim_sizes_list)

    spatial_vpa_dim_sizes_list = (dim_sizes[d] for d ∈ (:s, :r, :z, :vpa))
    vpa_splits, vpa_load_balances =
        get_splits_and_load_balances_from_sizes(block_size, spatial_vpa_dim_sizes_list)

    if vperp_splits != vpa_splits
        error("vperp_splits=$vperp_splits and vpa_splits=$vpa_splits should be identical "
              * "as the splits should only depend on the length of "
              * "spatial_*_dim_sizes_list")
    end

    # Base the choice on the worst load balance from vperp_load_balances and
    # vpa_load_balances, as each dimension will be parallelised over separately in part of
    # the collision operator, and the load balance when parallelising over both vperp and
    # vpa must be at least as good as the better of the two.
    load_balances = max.(vperp_load_balances, vpa_load_balances)

    # Number of processes assigned to the velocity dimension(s) by each split.
    # The velocity dimension is the 'fastest varying', so is the left-most entry in each
    # split.
    v_dim_nprocs = [s[end] for s ∈ vpa_splits]

    # Penalise the load_balances values so that we favour low numbers of processes for the
    # velocity dimension(s). It is an arbitrary choice to use `1.0 + v_dim_nprocs /
    # block_size[]` for this - the motivation is that the worst possible load balance for
    # a single dimension is 2 (as long as there are at least as many points in the
    # dimension as there are processes) so a simple linear `v_dim_nprocs` would pretty
    # much say 'only ever use one process for the velocity space dimensions unless there
    # are more processes than (n_species * (number of spatial points))', which seems a bit
    # too restrictive. Instead choose a penalisation factor that is about 1 for
    # v_dim_nprocs=1, and increases to 1 for v_dim_nprocs=(block_size/2). In principle we
    # could use any function that gets bigger with the number of processes...
    @. load_balances *= 1.0 + v_dim_nprocs / block_size[]

    best_index = argmin(load_balances)

    return vpa_splits[best_index]
end

"""
"""
function get_anyv_ranges(block_rank, block_size, split, anyv_dims, dim_sizes)
    if :vperp ∈ anyv_dims && :vpa ∈ anyv_dims
        vperp_vpa_split = get_best_split_from_sizes(split[end],
                                                    (dim_sizes[:vperp], dim_sizes[:vpa]))
        split = [split[1:end-1]..., vperp_vpa_split...]
    end

    dim_sizes_list = (dim_sizes[d] for d ∈ tuple(:s, :r, :z, anyv_dims[2:end]...))

    if !(:vpa ∈ anyv_dims || :vperp ∈ anyv_dims)
        # A 'serial' (in velocity space) region
        ranges = get_ranges_from_split(block_rank, block_size, split[1:end-1], dim_sizes_list)
    else
        ranges = get_ranges_from_split(block_rank, block_size, split, dim_sizes_list)
    end

    dims = tuple(:s, :r, :z, anyv_dims[2:end]...,)
    result = Dict(d=>r for (d,r) ∈ zip(dims, ranges))

    # Iterate over all points in ranges not being parallelized
    for d in all_dimensions
        if !(d in dims)
            result[d] = 1:dim_sizes[d]
        end
    end

    if !(:vpa ∈ anyv_dims || :vperp ∈ anyv_dims)
        # For a 'serial' 'anyv' region, following begin_anyv_region(), only loop over
        # velocity space on the rank-0 process of the anyv subblock
        if anyv_subblock_rank[] != 0
            result[:vpa] = 1:0
            result[:vperp] = 1:0
        end
    end

    return result
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

             block_rank >= block_size && error("block_rank ($block_rank) is >= "
                                               * "block_size ($block_size).")

             rank0 = (block_rank == 0)

             # Use empty tuple for serial region
             if rank0
                 loop_ranges_store[()] = LoopRanges(;
                     parallel_dims=(), rank0=rank0, is_anyv=false, anyv_rank0=rank0,
                     Dict(d=>1:n for (d,n) in dim_sizes)...)
             else
                 loop_ranges_store[()] = LoopRanges(;
                     parallel_dims=(), rank0=rank0, is_anyv=false, anyv_rank0=rank0,
                     Dict(d=>1:0 for (d,_) in dim_sizes)...)
             end

             for dims ∈ dimension_combinations
                 loop_ranges_store[dims] = LoopRanges(;
                     parallel_dims=dims, rank0=rank0, is_anyv=false, anyv_rank0=rank0,
                     get_best_ranges(block_rank, block_size, dims, dim_sizes)...)
             end


             # Set up looping for 'anyv' regions - used for the collision operator
             #####################################################################

             anyv_split = get_best_anyv_split(block_size, dim_sizes)

             anyv_subblock_size[] = anyv_split[end]
             number_of_anyv_blocks = prod(anyv_split[1:end-1])
             anyv_subblock_index = block_rank[] ÷ anyv_subblock_size[]
             anyv_rank_within_subblock = block_rank[] % anyv_subblock_size[]

             # Create communicator for the anyv subblock. OK to do this here as
             # communication.setup_distributed_memory_MPI() must have already been called
             # to set block_size[] and block_rank[]
             comm_anyv_subblock[] = MPI.Comm_split(comm_block[], anyv_subblock_index,
                                                   anyv_rank_within_subblock)
             anyv_subblock_rank[] = MPI.Comm_rank(comm_anyv_subblock[])
             anyv_isubblock_index[] = anyv_subblock_index
             anyv_nsubblocks_per_block[] = number_of_anyv_blocks
             anyv_rank0 = (anyv_subblock_rank[] == 0)

             for dims ∈ anyv_dimension_combinations
                 loop_ranges_store[dims] = LoopRanges(;
                     parallel_dims=dims, rank0=rank0, is_anyv=true, anyv_rank0=anyv_rank0,
                     get_anyv_ranges(block_rank, block_size, anyv_split, dims,
                                     dim_sizes)...)
             end

             #####################################################################

             loop_ranges[] = loop_ranges_store[()]

             return nothing
         end
     end)

 """
 For debugging the shared-memory parallelism, create ranges where only the loops for a
 single combinations of variables (given by `combination_to_split`) are parallelised,
 and which dimensions are parallelised can be set with the `dims_to_split...` arguments.

 Arguments
 ---------
 Keyword arguments `dim=n` are required for each dim in `all_dimensions` where `n` is
 an integer giving the size of the dimension.
 """
 function debug_setup_loop_ranges_split_one_combination!(
         block_rank, block_size, combination_to_split::NTuple{N,Symbol} where N,
         dims_to_split::Symbol...;
         dim_sizes...)

     block_rank >= block_size && error("block_rank ($block_rank) is >= block_size "
                                       * "($block_size).")

     rank0 = (block_rank == 0)

     # Use empty tuple for serial region
     if rank0
         serial_ranges = Dict(d=>1:n for (d,n) in dim_sizes)
         loop_ranges_store[()] = LoopRanges(;
             parallel_dims=(), rank0=rank0, is_anyv=false, anyv_rank0=rank0,
             serial_ranges...)
     else
         serial_ranges = Dict(d=>1:0 for (d,_) in dim_sizes)
         loop_ranges_store[()] = LoopRanges(;
             parallel_dims=(), rank0=rank0, is_anyv=false, anyv_rank0=rank0,
             serial_ranges...)
     end

     for dims ∈ dimension_combinations
         if dims == combination_to_split
             factors = factor(Vector, block_size)
             if length(factors) < length(dims_to_split)
                 error("Not enough factors ($factors) to split all of $dims_to_split")
             end
             ranges = Dict(d=>1:n for (d,n) in dim_sizes)
             remaining_block_size = block_size
             sub_rank = block_rank
             for (i,dim) ∈ enumerate(dims_to_split[1:end-1])
                 sub_block_size = factors[i]
                 remaining_block_size = remaining_block_size ÷ sub_block_size
                 sub_block_rank = sub_rank ÷ remaining_block_size
                 sub_rank = sub_rank % remaining_block_size
                 ranges[dim] = get_local_range(sub_block_rank, sub_block_size, dim_sizes[dim])
             end
             # For the 'last' dim, use the product of any remaining factors, in case
             # there were more factors than dims in dims_to_split
             dim = dims_to_split[end]
             sub_block_size = prod(factors[length(dims_to_split):end])
             remaining_block_size = remaining_block_size ÷ sub_block_size
             sub_block_rank = sub_rank ÷ remaining_block_size
             ranges[dim] = get_local_range(sub_block_rank,
                                           sub_block_size,
                                           dim_sizes[dim])
             loop_ranges_store[dims] = LoopRanges(;
                 parallel_dims=dims, rank0=rank0, is_anyv=false, anyv_rank0=rank0,
                 ranges...)
         else
             # Loop over all indices for non-parallelised dimensions (dimensions not in
             # `dims`), but only loop over parallel dimensions (dimensions in `dims`) on
             # rank0.
             this_ranges = Dict(d=>1:n for (d,n) in dim_sizes)
             if !rank0
                 for d ∈ dims
                     this_ranges[d] = 1:0
                 end
             end
             loop_ranges_store[dims] = LoopRanges(;
                 parallel_dims=dims, rank0=rank0, is_anyv=false, anyv_rank0=rank0,
                 this_ranges...)
         end
     end

     # Set up looping for 'anyv' regions - used for the collision operator
     #####################################################################

     anyv_split = [1, 1, 1, block_size]

     anyv_subblock_size[] = anyv_split[end]
     number_of_anyv_blocks = prod(anyv_split[1:end-1])
     anyv_subblock_index = block_rank[] ÷ anyv_subblock_size[]
     anyv_rank_within_subblock = block_rank[] % anyv_subblock_size[]

     # Create communicator for the anyv subblock. OK to do this here as
     # communication.setup_distributed_memory_MPI() must have already been called
     # to set block_size[] and block_rank[]
     comm_anyv_subblock[] = MPI.Comm_split(comm_block[], anyv_subblock_index,
                                           anyv_rank_within_subblock)
     anyv_subblock_rank[] = MPI.Comm_rank(comm_anyv_subblock[])
     anyv_isubblock_index[] = anyv_subblock_index
     anyv_nsubblocks_per_block[] = number_of_anyv_blocks
     anyv_rank0 = (anyv_subblock_rank[] == 0)

     for dims ∈ anyv_dimension_combinations
         if dims == combination_to_split
             factors = factor(Vector, block_size)
             if length(factors) < length(dims_to_split)
                 error("Not enough factors ($factors) to split all of $dims_to_split")
             end
             ranges = Dict(d=>1:n for (d,n) in dim_sizes)
             remaining_block_size = block_size
             sub_rank = block_rank
             for (i,dim) ∈ enumerate(dims_to_split[1:end-1])
                 sub_block_size = factors[i]
                 remaining_block_size = remaining_block_size ÷ sub_block_size
                 sub_block_rank = sub_rank ÷ remaining_block_size
                 sub_rank = sub_rank % remaining_block_size
                 ranges[dim] = get_local_range(sub_block_rank, sub_block_size, dim_sizes[dim])
             end
             # For the 'last' dim, use the product of any remaining factors, in case
             # there were more factors than dims in dims_to_split
             dim = dims_to_split[end]
             sub_block_size = prod(factors[length(dims_to_split):end])
             remaining_block_size = remaining_block_size ÷ sub_block_size
             sub_block_rank = sub_rank ÷ remaining_block_size
             ranges[dim] = get_local_range(sub_block_rank,
                                           sub_block_size,
                                           dim_sizes[dim])
             loop_ranges_store[dims] = LoopRanges(;
                 parallel_dims=dims, rank0=rank0, is_anyv=true, anyv_rank0=anyv_rank0,
                 ranges...)
         else
             this_ranges = Dict(d=>1:n for (d,n) in dim_sizes)
             if !rank0
                 for d ∈ (:vperp, :vpa)
                     this_ranges[d] = 1:0
                 end
             end
             loop_ranges_store[dims] = LoopRanges(;
                 parallel_dims=dims, rank0=rank0, is_anyv=true, anyv_rank0=anyv_rank0,
                 this_ranges...)
         end
     end

     #####################################################################

     loop_ranges[] = loop_ranges_store[()]

     return nothing
 end

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
            it_vars_vec = collect(reverse(it_vars))
            for i ∈ 1:length(it_vars_vec)
                rangei = Symbol(:range, i)
                ex = quote
                    for $(esc(it_vars_vec[i])) = $rangei
                        $ex
                    end
                end
            end
            range_exprs_vec = collect(reverse($range_exprs))
            for i ∈ 1:length(it_vars_vec)
                rangei = Symbol(:range, i)
                range_expr = eval(range_exprs_vec[i])
                ex = quote
                    $rangei = $range_expr
                    $ex
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

             Returns immediately if loop_ranges[] is already set to the parallel
             dimensions being requested. This allows the begin_*_region() calls to be
             placed where they make logical sense, with no cost if a call happens to be
             repeated (e.g. in different functions).

             Calls `_block_synchronize()` to synchronize the processes operating on a
             shared-memory block, unless `no_synchronize=true` is passed as an argument.
             """
             function $sync_name(; no_synchronize::Bool=false)
                 if !loop_ranges[].is_anyv && loop_ranges[].parallel_dims == $dims
                     return
                 end
                 if !no_synchronize
                     _block_synchronize()
                 end
                 loop_ranges[] = loop_ranges_store[$dims]
             end
             export $sync_name
         end)
end

"""
Begin region in which (:s,:r,:z) dimensions and velocity dimensions are parallelized by
being split between processes, and which velocity dimensions are parallelized can be
switched within the outer loop over (:s,:r,:z). This parallelization scheme is intended
for use in the collision operator.

Returns immediately if loop_ranges[] is already
set to the parallel
dimensions being requested. This allows the begin_*_region() calls to be
placed where they make logical sense, with no cost if a call happens to be
repeated (e.g. in different functions).

Calls `_block_synchronize()` to synchronize the processes operating on a
shared-memory block, unless `no_synchronize=true` is passed as an argument.
"""
function begin_s_r_z_anyv_region(; no_synchronize::Bool=false)
    if loop_ranges[].is_anyv
        return
    end
    if !no_synchronize
        _block_synchronize()
    end
    loop_ranges[] = loop_ranges_store[(:anyv,)]
end
export begin_s_r_z_anyv_region

# Create begin_anyv_*_region() functions to use within a begin_s_r_z_anyv_region() region.
for dims ∈ anyv_dimension_combinations
    # Create an expression-function/macro combination for each level of the
    # loop
    dims_symb = Symbol(dims_string(dims))

    # Create a function for beginning regions where 'dims' are parallelized
    sync_name = Symbol(:begin_, dims_symb, :_region)
    eval(quote
             """
             Begin 'anyv' sub-region in which $($dims[2:end]) velocity space dimensions
             are parallelized by being split between processes.

             Returns immediately if loop_ranges[] is already set to the parallel
             dimensions being requested. This allows the begin_anyv_*_region() calls to be
             placed where they make logical sense, with no cost if a call happens to be
             repeated (e.g. in different functions).

             Calls `_anyv_subblock_synchronize()` to synchronize the processes operating on
             an 'anyv' shared-memory sub-block, unless `no_synchronize=true` is passed as
             an argument.
             """
             function $sync_name(; no_synchronize::Bool=false)
                 if loop_ranges[].parallel_dims == $dims
                     return
                 end
                 if !loop_ranges[].is_anyv
                     error("Trying to change the 'anyv' sub-region when not an an 'anyv' "
                           * "region")
                 end
                 if !no_synchronize
                     _anyv_subblock_synchronize()
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

Returns immediately if loop_ranges[] is already set for a serial region. This allows the
begin_*_region() calls to be placed where they make logical sense, with no cost if a
call happens to be repeated (e.g. in different functions).

Calls `_block_synchronize()` to synchronize the processes operating on a shared-memory
block, unless `no_synchronize=true` is passed as an argument.
"""
function begin_serial_region(; no_synchronize::Bool=false)
    if loop_ranges[].parallel_dims == ()
        return
    end
    if !no_synchronize
        _block_synchronize()
    end
    loop_ranges[] = loop_ranges_store[()]
end
export begin_serial_region

"""
Run a block of code on only anyv-subblock-rank-0 of each group of processes operating on
an 'anyv' shared-memory subblock
"""
macro anyv_serial_region(blk)
    return quote
        if loop_ranges[].anyv_rank0
            $(esc(blk))
        end
    end
end
export @anyv_serial_region

end # looping
