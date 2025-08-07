"""
Provides convenience macros for shared-memory-parallel loops
"""
module looping

using ..debugging
using ..communication: _block_synchronize, comm_block,
                       _anysv_subblock_synchronize, comm_anysv_subblock,
                       anysv_subblock_rank, anysv_subblock_size, anysv_isubblock_index,
                       anysv_nsubblocks_per_block,
                       _anyzv_subblock_synchronize, comm_anyzv_subblock,
                       anyzv_subblock_rank, anyzv_subblock_size, anyzv_isubblock_index,
                       anyzv_nsubblocks_per_block
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
const anysv_dimension_combinations = tuple((:anysv,),
                                          ((:anysv, d...) for d ∈
                                           unique(combinations((:s, :vperp, :vpa))))...)
const anyzv_dimension_combinations = tuple((:anyzv,),
                                          ((:anyzv, d...) for d ∈
                                           unique(combinations((:z, :vperp, :vpa))))...)

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
    is_anysv::Bool
    anysv_rank0::Bool
    is_anyzv::Bool
    anyzv_rank0::Bool
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
function get_subblock_splits(sub_block_size, n)
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

        # Call get_subblock_splits() recursively to split `remaining` over the rest of
        # the dimensions, combining the results into a Vector of possible
        # factorizations.
        for other_factors in get_subblock_splits(remaining, n-1)
            # ignore any duplicates
            if !([this_factor; other_factors] ∈ result)
                push!(result, [this_factor; other_factors])
            end
        end
    end
    return result
end

"""
Find possible divisions of each number less than or equal to `block_size` into `n`
factors.
"""
function get_splits(block_size, n)
    result = Vector{Vector{mk_int}}(undef, 0)
    for nproc ∈ block_size:-1:1
        result = vcat(result, get_subblock_splits(nproc, n))
    end
    return result
end

"""
Calculate the maximum number of grid points on any process

This is a measure of the maximum amount of work to do on a single process. Minimising this
will make the parallelisation as efficient as possible.

Arguments
---------
nprocs_list : Vector{mk_int}
    Number of processes for each dimension
sizes : Vector{mk_int}
    Size of each dimension
"""
function get_max_work(nprocs_list, sizes)
    return prod(ceil(mk_int, s/n) for (n,s) ∈ zip(nprocs_list, sizes))
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

    if dim_size == 0 || sub_block_rank ≥ sub_block_size
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
    effective_block_size = prod(best_split) # May be less than block_size
    ranges = get_ranges_from_split(block_rank, effective_block_size, best_split,
                                   dim_sizes_list)
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
function get_splits_and_max_work_from_sizes(block_size, dim_sizes_list)
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

    # Sort splits so that those with most splitting on the slowest-varying dimensions come
    # first. This ensures that in the case of a tie, we split the slower-varying
    # dimension.
    sort!(splits; rev=true)

    max_work = [get_max_work(s, dim_sizes_list) for s ∈ splits]

    return splits, max_work
end

function get_best_split_from_sizes(block_size, dim_sizes_list)
    splits, max_work =
        get_splits_and_max_work_from_sizes(block_size, dim_sizes_list)

    best_index = argmin(max_work)
    return splits[best_index]
end

"""
"""
function get_ranges_from_split(block_rank, effective_block_size, split, dim_sizes_list)
    if block_rank ≥ effective_block_size
        # This process is not needed by this split
        return [1:0 for _ ∈ dim_sizes_list]
    end
    if effective_block_size < prod(split)
        error("effective_block_size=$effective_block_size is smaller than "
              * "prod(split)=", prod(split))
    end

    # Get rank of this process in each sub-block (with sub-block sizes given by split).
    sb_ranks = zeros(mk_int, length(dim_sizes_list))
    sub_rank = block_rank
    remaining_block_size = effective_block_size
    for (i, sb_size) in enumerate(split)
        remaining_block_size = remaining_block_size ÷ sb_size
        sb_ranks[i] = sub_rank ÷ remaining_block_size
        sub_rank = sub_rank % remaining_block_size
    end

    return [get_local_range(sb_rank, sb_size, dim_size)
            for (sb_rank, sb_size, dim_size) in zip(sb_ranks, split, dim_sizes_list)]
end

"""
Find the numbers of processes for each dimension that optimize load balance for 'anysv'
type loops for a certain block_size.

The 'anysv' parallelisation patterns are designed for use in the collision operator. They
all share the same parallelisation in spatial dimensions so that the region type can be
switched between 'anysv' types within a loop over spatial dimensions (@loop_r_z). It is
only defined for ions, not for neutrals.

Parts of the collision operator cannot conveniently be parallelised over velocity
dimensions, so this function aims to assign as much parallelism as possible to the spatial
dimensions.
"""
function get_best_anysv_split(block_size, dim_sizes)

    spatial_vperp_dim_sizes_list = Tuple(dim_sizes[d] for d ∈ (:r, :z, :vperp))
    vperp_splits, vperp_max_work =
        get_splits_and_max_work_from_sizes(block_size, spatial_vperp_dim_sizes_list)

    spatial_vpa_dim_sizes_list = Tuple(dim_sizes[d] for d ∈ (:r, :z, :vpa))
    vpa_splits, vpa_max_work =
        get_splits_and_max_work_from_sizes(block_size, spatial_vpa_dim_sizes_list)

    if vperp_splits != vpa_splits
        error("vperp_splits=$vperp_splits and vpa_splits=$vpa_splits should be identical "
              * "as the splits should only depend on the length of "
              * "spatial_*_dim_sizes_list")
    end

    # Base the choice on the worst load balance from vperp_load_balances and
    # vpa_load_balances, as each dimension will be parallelised over separately in part of
    # the collision operator, and the load balance when parallelising over both vperp and
    # vpa must be at least as good as the better of the two.
    max_work = max.(vperp_max_work, vpa_max_work)

    # Number of processes assigned to the velocity dimension(s) by each split.
    # The velocity dimension is the 'fastest varying', so is the left-most entry in each
    # split.
    v_dim_nprocs = [s[end] for s ∈ vpa_splits]

    # Penalise the max_work values so that we favour low numbers of processes for the
    # velocity dimension(s). It is an arbitrary choice to use `1.0 + 0.5 * v_dim_nprocs /
    # block_size[]` for this - the motivation is that the spread of `max_work` values
    # tends to be less than ~1.5 times the best value so even a not very strong
    # penalisation just leads to never splitting the velocity space unless there are more
    # processes than spatial grid points. In principle we could use any function that gets
    # bigger with the number of processes...
    max_work = @. max_work * (1.0 + v_dim_nprocs^2 / 20)

    best_index = argmin(max_work)

    best_split = vpa_splits[best_index]

    # Because we only optimize for 1 velocity space dimension, it can happen that there
    # are 'extra' factors that the algorithm does not think can be use for velocity space,
    # and end up in the process number for one of the other dimensions. However because at
    # least for part of the time the velocity space can use a 2d parallelisation within an
    # anysv region, it can make use of those processors. So check for 'extra' factors, and
    # move them to the velocity dimension.
    for i ∈ 1:length(best_split) - 1
        dim_size = spatial_vpa_dim_sizes_list[i]
        if best_split[i] > dim_size
            # May have an extra factor

            all_factors = factor(Vector, best_split[i])

            # Check biggest factors first
            sort!(all_factors; rev=true)

            for f ∈ all_factors
                if best_split[i] / f ≥ dim_size
                    best_split[i] /= f
                    best_split[end] *= f
                end
            end
        end
    end

    return best_split
end

"""
"""
function get_anysv_ranges(block_rank, split, anysv_dims, dim_sizes)
    effective_block_size = prod(split) # May be less than block_size

    if :s ∈ anysv_dims && :vperp ∈ anysv_dims && :vpa ∈ anysv_dims
        s_vperp_vpa_split = get_best_split_from_sizes(split[end],
                                                      (dim_sizes[:s], dim_sizes[:vperp], dim_sizes[:vpa]))
        split = [split[1:end-1]..., s_vperp_vpa_split...]
    elseif :s ∈ anysv_dims && :vperp ∈ anysv_dims
        s_vperp_split = get_best_split_from_sizes(split[end],
                                                  (dim_sizes[:s], dim_sizes[:vperp]))
        split = [split[1:end-1]..., s_vperp_split...]
    elseif :s ∈ anysv_dims && :vpa ∈ anysv_dims
        s_vpa_split = get_best_split_from_sizes(split[end],
                                                (dim_sizes[:s], dim_sizes[:vpa]))
        split = [split[1:end-1]..., s_vpa_split...]
    elseif :vperp ∈ anysv_dims && :vpa ∈ anysv_dims
        vperp_vpa_split = get_best_split_from_sizes(split[end],
                                                    (dim_sizes[:vperp], dim_sizes[:vpa]))
        split = [split[1:end-1]..., vperp_vpa_split...]
    end

    dim_sizes_list = (dim_sizes[d] for d ∈ tuple(:r, :z, anysv_dims[2:end]...))

    if !(:s ∈ anysv_dims || :vpa ∈ anysv_dims || :vperp ∈ anysv_dims)
        # A 'serial' (in species/velocity space) region
        ranges = get_ranges_from_split(block_rank, effective_block_size, split[1:end-1], dim_sizes_list)
    else
        ranges = get_ranges_from_split(block_rank, effective_block_size, split, dim_sizes_list)
    end

    dims = tuple(:r, :z, anysv_dims[2:end]...,)
    result = Dict(d=>r for (d,r) ∈ zip(dims, ranges))

    # Iterate over all points in ranges not being parallelized
    for d in all_dimensions
        if !(d in dims)
            result[d] = 1:dim_sizes[d]
        end
    end

    if !(:s ∈ anysv_dims || :vpa ∈ anysv_dims || :vperp ∈ anysv_dims)
        # For a 'serial' 'anysv' region, following @begin_anysv_region(), only loop over
        # species/velocity space on the rank-0 process of the anysv subblock
        #
        # Calculate the rank in the subblock from block_rank rather than using
        # `anysv_subblock_rank[]` so that we can test this function without having to set
        # up communications.
        if block_rank % split[end] != 0
            result[:s] = 1:0
            result[:vpa] = 1:0
            result[:vperp] = 1:0
        end
    end

    return result
end

"""
Find the numbers of processes for each dimension that optimize load balance for 'anyzv'
type loops for a certain block_size.

The 'anyzv' parallelisation patterns are designed for use in the kinetic electron implicit
solve. They all share the same parallelisation in the r-dimension so that the region type
can be switched between 'anyzv' types within a loop over r (@loop_r).

The matrix-solve needed for the kinetic electron preconditioner is a bottleneck and
requires a relatively large amount of communication so probably does not scale perfectly.
To maximise parallelism, parallelise the r-dimension as much as possible before
parallelising z/vperp/vpa.

This region type excludes the final point in the r-dimension, assuming that it will be
filled in by a boundary condition or call to
[`moment_kinetics.communication.halo_swap!`](@ref).
"""
function get_best_anyzv_split(block_size, dim_sizes)

    # For this version, skip one point (the last point) in r
    r_size = dim_sizes[:r] - 1

    # Choose the number of processes to distribute the r grid across as:
    # 1. some factor of block_size
    factors = factor(Vector, block_size)
    r_splits = [[1]; collect(combinations(factors))]
    r_splits = unique(prod(r) for r ∈ r_splits)
    sort!(r_splits)

    # 2. minimises the amount of work per subblock
    r_work = [get_max_work(s, [r_size]) for s ∈ r_splits]
    min_work = minimum(r_work)

    # 3. for the given minimum amount of work, maximise the number of processes per
    #    subblock.
    # r_splits was sorted in ascending order, so want the entry corresponding to the first
    # instance of min_work.
    min_ind = findfirst(x -> (x == min_work), r_work)
    r_nproc = r_splits[min_ind]

    zv_nproc, rem = divrem(block_size, r_nproc)
    if rem != 0
        # Sanity check - this should never happen.
        error("Selected r_nproc=$r_nproc does not divide block_size=$block_size.")
    end

    return [r_nproc, zv_nproc]
end

"""
"""
function get_anyzv_ranges(block_rank, split, anyzv_dims, dim_sizes)
    effective_block_size = prod(split) # Allow for this being less than block_size, but should not be for current anyzv implementation.

    anyzv_dim_sizes = Dict(k => (k == :r ? max(v - 1, 1) : v) for (k,v) ∈ dim_sizes)

    if :z ∈ anyzv_dims && :vperp ∈ anyzv_dims && :vpa ∈ anyzv_dims
        z_vperp_vpa_split = get_best_split_from_sizes(split[end],
                                                      (anyzv_dim_sizes[:z], anyzv_dim_sizes[:vperp], anyzv_dim_sizes[:vpa]))
        split = [split[1:end-1]..., z_vperp_vpa_split...]
    elseif :z ∈ anyzv_dims && :vperp ∈ anyzv_dims
        z_vperp_split = get_best_split_from_sizes(split[end],
                                                  (anyzv_dim_sizes[:z], anyzv_dim_sizes[:vperp]))
        split = [split[1:end-1]..., z_vperp_split...]
    elseif :z ∈ anyzv_dims && :vpa ∈ anyzv_dims
        z_vpa_split = get_best_split_from_sizes(split[end],
                                                (anyzv_dim_sizes[:z], anyzv_dim_sizes[:vpa]))
        split = [split[1:end-1]..., z_vpa_split...]
    elseif :vperp ∈ anyzv_dims && :vpa ∈ anyzv_dims
        vperp_vpa_split = get_best_split_from_sizes(split[end],
                                                    (anyzv_dim_sizes[:vperp], anyzv_dim_sizes[:vpa]))
        split = [split[1:end-1]..., vperp_vpa_split...]
    end

    # Excluding final grid point in r, see get_best_anyzv_split().
    dim_sizes_list = (anyzv_dim_sizes[d] for d ∈ tuple(:r, anyzv_dims[2:end]...))

    serial_region = !(:z ∈ anyzv_dims || :vpa ∈ anyzv_dims || :vperp ∈ anyzv_dims)

    if serial_region
        # A 'serial' (in z/velocity space) region
        ranges = get_ranges_from_split(block_rank, effective_block_size, split[1:end-1], dim_sizes_list)
    else
        ranges = get_ranges_from_split(block_rank, effective_block_size, split, dim_sizes_list)
    end

    dims = tuple(:r, anyzv_dims[2:end]...,)
    result = Dict(d=>r for (d,r) ∈ zip(dims, ranges))

    # Iterate over all points in ranges not being parallelized
    for d in all_dimensions
        if !(d in dims)
            result[d] = 1:anyzv_dim_sizes[d]
        end
    end

    if serial_region
        # For a 'serial' 'anyzv' region, following @begin_anyzv_region(), only loop over
        # species/velocity space on the rank-0 process of the anyzv subblock
        #
        # Calculate the rank in the subblock from block_rank rather than using
        # `anyzv_subblock_rank[]` so that we can test this function without having to set
        # up communications.
        if block_rank % split[end] != 0
            result[:z] = 1:0
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
variable in @begin_*_region() functions
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
                     parallel_dims=(), rank0=rank0, is_anysv=false, anysv_rank0=rank0,
                     is_anyzv=false, anyzv_rank0=rank0,
                     Dict(d=>1:n for (d,n) in dim_sizes)...)
             else
                 loop_ranges_store[()] = LoopRanges(;
                     parallel_dims=(), rank0=rank0, is_anysv=false, anysv_rank0=rank0,
                     is_anyzv=false, anyzv_rank0=rank0,
                     Dict(d=>1:0 for (d,_) in dim_sizes)...)
             end

             for dims ∈ dimension_combinations
                 loop_ranges_store[dims] = LoopRanges(;
                     parallel_dims=dims, rank0=rank0, is_anysv=false, anysv_rank0=rank0,
                     is_anyzv=false, anyzv_rank0=rank0,
                     get_best_ranges(block_rank, block_size, dims, dim_sizes)...)
             end


             # Set up looping for 'anysv' regions - used for the collision operator
             #####################################################################

             anysv_split = get_best_anysv_split(block_size, dim_sizes)

             anysv_subblock_size[] = anysv_split[end]
             number_of_anysv_blocks = prod(anysv_split[1:end-1])
             anysv_subblock_index = block_rank[] ÷ anysv_subblock_size[]
             if anysv_subblock_index ≥ number_of_anysv_blocks
                 anysv_subblock_index = nothing
                 anysv_rank_within_subblock = -1
             else
                 anysv_rank_within_subblock = block_rank[] % anysv_subblock_size[]
             end

             # Create communicator for the anysv subblock. OK to do this here as
             # communication.setup_distributed_memory_MPI() must have already been called
             # to set block_size[] and block_rank[]
             comm_anysv_subblock[] = MPI.Comm_split(comm_block[], anysv_subblock_index,
                                                   anysv_rank_within_subblock)
             anysv_subblock_rank[] = anysv_rank_within_subblock
             anysv_isubblock_index[] = anysv_subblock_index
             anysv_nsubblocks_per_block[] = number_of_anysv_blocks
             anysv_rank0 = (anysv_subblock_rank[] == 0)

             for dims ∈ anysv_dimension_combinations
                 loop_ranges_store[dims] = LoopRanges(;
                     parallel_dims=dims, rank0=rank0, is_anysv=true,
                     anysv_rank0=anysv_rank0, is_anyzv=true, anyzv_rank0=rank0,
                     get_anysv_ranges(block_rank, anysv_split, dims, dim_sizes)...)
             end

             #####################################################################


             # Set up looping for 'anyzv' regions - used for kinetic electron implicit
             # solve
             #########################################################################

             anyzv_split = get_best_anyzv_split(block_size, dim_sizes)

             anyzv_subblock_size[] = anyzv_split[end]
             number_of_anyzv_blocks = prod(anyzv_split[1:end-1])
             anyzv_subblock_index = block_rank[] ÷ anyzv_subblock_size[]
             if anyzv_subblock_index ≥ number_of_anyzv_blocks
                 anyzv_subblock_index = nothing
                 anyzv_rank_within_subblock = -1
             else
                 anyzv_rank_within_subblock = block_rank[] % anyzv_subblock_size[]
             end

             # Create communicator for the anyzv subblock. OK to do this here as
             # communication.setup_distributed_memory_MPI() must have already been called
             # to set block_size[] and block_rank[]
             comm_anyzv_subblock[] = MPI.Comm_split(comm_block[], anyzv_subblock_index,
                                                    anyzv_rank_within_subblock)
             anyzv_subblock_rank[] = anyzv_rank_within_subblock
             anyzv_isubblock_index[] = anyzv_subblock_index
             anyzv_nsubblocks_per_block[] = number_of_anyzv_blocks
             anyzv_rank0 = (anyzv_subblock_rank[] == 0)

             for dims ∈ anyzv_dimension_combinations
                 loop_ranges_store[dims] = LoopRanges(;
                     parallel_dims=dims, rank0=rank0, is_anysv=false, anysv_rank0=rank0,
                     is_anyzv=true, anyzv_rank0=anyzv_rank0,
                     get_anyzv_ranges(block_rank, anyzv_split, dims, dim_sizes)...)
             end

             #########################################################################

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
             parallel_dims=(), rank0=rank0, is_anysv=false, anysv_rank0=rank0,
             is_anyzv=false, anyzv_rank0=rank0, serial_ranges...)
     else
         serial_ranges = Dict(d=>1:0 for (d,_) in dim_sizes)
         loop_ranges_store[()] = LoopRanges(;
             parallel_dims=(), rank0=rank0, is_anysv=false, anysv_rank0=rank0,
             is_anyzv=false, anyzv_rank0=rank0, serial_ranges...)
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
                 parallel_dims=dims, rank0=rank0, is_anysv=false, anysv_rank0=rank0,
                 is_anyzv=false, anyzv_rank0=rank0, ranges...)
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
                 parallel_dims=dims, rank0=rank0, is_anysv=false, anysv_rank0=rank0,
                 is_anyzv=false, anyzv_rank0=rank0, this_ranges...)
         end
     end

     # Set up looping for 'anysv' regions - used for the collision operator
     ######################################################################

     anysv_split = [1, 1, block_size]

     anysv_subblock_size[] = anysv_split[end]
     number_of_anysv_blocks = prod(anysv_split[1:end-1])
     anysv_subblock_index = block_rank[] ÷ anysv_subblock_size[]
     anysv_rank_within_subblock = block_rank[] % anysv_subblock_size[]

     # Create communicator for the anysv subblock. OK to do this here as
     # communication.setup_distributed_memory_MPI() must have already been called
     # to set block_size[] and block_rank[]
     comm_anysv_subblock[] = MPI.Comm_split(comm_block[], anysv_subblock_index,
                                            anysv_rank_within_subblock)
     anysv_subblock_rank[] = MPI.Comm_rank(comm_anysv_subblock[])
     anysv_isubblock_index[] = anysv_subblock_index
     anysv_nsubblocks_per_block[] = number_of_anysv_blocks
     anysv_rank0 = (anysv_subblock_rank[] == 0)

     for dims ∈ anysv_dimension_combinations
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
                 parallel_dims=dims, rank0=rank0, is_anysv=true, anysv_rank0=anysv_rank0,
                 is_anyzv=false, anyzv_rank0=rank0, ranges...)
         else
             this_ranges = Dict(d=>1:n for (d,n) in dim_sizes)
             if !rank0
                 for d ∈ (:s, :vperp, :vpa)
                     this_ranges[d] = 1:0
                 end
             end
             loop_ranges_store[dims] = LoopRanges(;
                 parallel_dims=dims, rank0=rank0, is_anysv=true, anysv_rank0=anysv_rank0,
                 is_anyzv=false, anyzv_rank0=rank0, this_ranges...)
         end
     end

     ######################################################################

     # Set up looping for 'anyzv' regions - used for the kinetic electron implicit solve
     ###################################################################################

     anyzv_split = [1, 1, block_size]

     anyzv_subblock_size[] = anyzv_split[end]
     number_of_anyzv_blocks = prod(anyzv_split[1:end-1])
     anyzv_subblock_index = block_rank[] ÷ anyzv_subblock_size[]
     anyzv_rank_within_subblock = block_rank[] % anyzv_subblock_size[]

     # Create communicator for the anyzv subblock. OK to do this here as
     # communication.setup_distributed_memory_MPI() must have already been called
     # to set block_size[] and block_rank[]
     comm_anyzv_subblock[] = MPI.Comm_split(comm_block[], anyzv_subblock_index,
                                            anyzv_rank_within_subblock)
     anyzv_subblock_rank[] = MPI.Comm_rank(comm_anyzv_subblock[])
     anyzv_isubblock_index[] = anyzv_subblock_index
     anyzv_nsubblocks_per_block[] = number_of_anyzv_blocks
     anyzv_rank0 = (anyzv_subblock_rank[] == 0)

     for dims ∈ anyzv_dimension_combinations
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
                 parallel_dims=dims, rank0=rank0, is_anysv=false, anysv_rank0=rank0,
                 is_anyzv=true, anyzv_rank0=anyzv_rank0, ranges...)
         else
             this_ranges = Dict(d=>1:n for (d,n) in dim_sizes)
             if !rank0
                 for d ∈ (:z, :vperp, :vpa)
                     this_ranges[d] = 1:0
                 end
             end
             loop_ranges_store[dims] = LoopRanges(;
                 parallel_dims=dims, rank0=rank0, is_anysv=false, anysv_rank0=rank0,
                 is_anyzv=true, anyzv_rank0=anyzv_rank0, this_ranges...)
         end
     end

     ###################################################################################

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
            ex = quote
                @inbounds begin
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
    sync_macro_name = Symbol("@", sync_name)
    sync_name_internal = Symbol(sync_name, :_internal)
    eval(quote
             """
             Begin region in which $($dims) dimensions are parallelized by being split
             between processes.

             Returns immediately if loop_ranges[] is already set to the parallel
             dimensions being requested. This allows the @begin_*_region() calls to be
             placed where they make logical sense, with no cost if a call happens to be
             repeated (e.g. in different functions).

             Calls `_block_synchronize()` to synchronize the processes operating on a
             shared-memory block, unless `true` is passed as the `no_synchronize`
             argument.
             """
             macro $sync_name(no_synchronize::Bool=false)
                 id_hash = @debug_block_synchronize_quick_ifelse(
                                hash(string(@__FILE__, @__LINE__)),
                                nothing
                               )
                 return quote
                     $($sync_name_internal)($id_hash, $(esc(no_synchronize)))
                 end
             end

             function $sync_name_internal(call_site::Union{Nothing,UInt64},
                                          no_synchronize::Bool)
                 if !loop_ranges[].is_anysv && !loop_ranges[].is_anyzv && loop_ranges[].parallel_dims == $dims
                     return
                 end
                 if !no_synchronize
                     _block_synchronize(call_site)
                 end
                 loop_ranges[] = loop_ranges_store[$dims]
             end
             export $sync_macro_name
         end)
end

"""
Begin region in which (:r,:z) dimensions and species/velocity dimensions are parallelized
by being split between processes, and which species/velocity dimensions are parallelized
can be switched within the outer loop over (:r,:z). This parallelization scheme is
intended for use in the collision operator.

Returns immediately if loop_ranges[] is already set to the parallel dimensions being
requested. This allows the @begin_*_region() calls to be placed where they make logical
sense, with no cost if a call happens to be repeated (e.g. in different functions).

Calls `_block_synchronize()` to synchronize the processes operating on a
shared-memory block, unless `no_synchronize=true` is passed as an argument.
"""
macro begin_r_z_anysv_region(no_synchronize::Bool=false)
    id_hash = @debug_block_synchronize_quick_ifelse(
                   hash(string(@__FILE__, @__LINE__)),
                   nothing
                  )
    return quote
        begin_r_z_anysv_region_internal($id_hash, $(esc(no_synchronize)))
    end
end
function begin_r_z_anysv_region_internal(call_site, no_synchronize::Bool)
    if loop_ranges[].is_anysv
        return
    end
    if !no_synchronize
        _block_synchronize(call_site)
    end
    loop_ranges[] = loop_ranges_store[(:anysv,)]
end
export @begin_r_z_anysv_region

# Create @begin_anysv_*_region() functions to use within a @begin_r_z_anysv_region() region.
for dims ∈ anysv_dimension_combinations
    # Create an expression-function/macro combination for each level of the
    # loop
    dims_symb = Symbol(dims_string(dims))

    # Create a function for beginning regions where 'dims' are parallelized
    sync_name = Symbol(:begin_, dims_symb, :_region)
    sync_macro_name = Symbol("@", sync_name)
    sync_name_internal = Symbol(sync_name, :_internal)
    eval(quote
             """
             Begin 'anysv' sub-region in which $($dims[2:end]) species/velocity space
             dimensions are parallelized by being split between processes.

             Returns immediately if loop_ranges[] is already set to the parallel
             dimensions being requested. This allows the @begin_anysv_*_region() calls to be
             placed where they make logical sense, with no cost if a call happens to be
             repeated (e.g. in different functions).

             Calls `_anysv_subblock_synchronize()` to synchronize the processes operating on
             an 'anysv' shared-memory sub-block, unless `true` is passed as the
             `no_synchronize` argument.
             """
             macro $sync_name(no_synchronize::Bool=false)
                 id_hash = @debug_block_synchronize_quick_ifelse(
                                hash(string(@__FILE__, @__LINE__)),
                                nothing
                               )
                 return quote
                     $($sync_name_internal)($id_hash, $(esc(no_synchronize)))
                 end
             end

             function $sync_name_internal(call_site, no_synchronize::Bool=false)
                 if loop_ranges[].parallel_dims == $dims
                     return
                 end
                 if !loop_ranges[].is_anysv
                     error("Trying to change the 'anysv' sub-region when not an an 'anysv' "
                           * "region")
                 end
                 if !no_synchronize
                     _anysv_subblock_synchronize(call_site)
                 end
                 loop_ranges[] = loop_ranges_store[$dims]
             end
             export $sync_macro_name
         end)
end

"""
Begin region in which r-dimension and z/velocity dimensions are parallelized by being
split between processes, and which z/velocity dimensions are parallelized can be switched
within the outer loop over r. This parallelization scheme is intended for use in the
kinetic electron implicit solve.

Returns immediately if loop_ranges[] is already set to the parallel dimensions being
requested. This allows the @begin_*_region() calls to be placed where they make logical
sense, with no cost if a call happens to be repeated (e.g. in different functions).

Calls `_block_synchronize()` to synchronize the processes operating on a
shared-memory block, unless `no_synchronize=true` is passed as an argument.
"""
macro begin_r_anyzv_region(no_synchronize::Bool=false)
    id_hash = @debug_block_synchronize_quick_ifelse(
                   hash(string(@__FILE__, @__LINE__)),
                   nothing
                  )
    return quote
        begin_r_anyzv_region_internal($id_hash, $(esc(no_synchronize)))
    end
end
function begin_r_anyzv_region_internal(call_site, no_synchronize::Bool)
    if loop_ranges[].is_anyzv
        return
    end
    if !no_synchronize
        _block_synchronize(call_site)
    end
    loop_ranges[] = loop_ranges_store[(:anyzv,)]
end
export @begin_r_anyzv_region

# Create @begin_anyzv_*_region() functions to use within a @begin_r_anyzv_region() region.
for dims ∈ anyzv_dimension_combinations
    # Create an expression-function/macro combination for each level of the
    # loop
    dims_symb = Symbol(dims_string(dims))

    # Create a function for beginning regions where 'dims' are parallelized
    sync_name = Symbol(:begin_, dims_symb, :_region)
    sync_macro_name = Symbol("@", sync_name)
    sync_name_internal = Symbol(sync_name, :_internal)
    eval(quote
             """
             Begin 'anyzv' sub-region in which $($dims[2:end]) z/velocity space
             dimensions are parallelized by being split between processes.

             Returns immediately if loop_ranges[] is already set to the parallel
             dimensions being requested. This allows the @begin_anyzv_*_region() calls to be
             placed where they make logical sense, with no cost if a call happens to be
             repeated (e.g. in different functions).

             Calls `_anyzv_subblock_synchronize()` to synchronize the processes operating on
             an 'anyzv' shared-memory sub-block, unless `true` is passed as the
             `no_synchronize` argument.
             """
             macro $sync_name(no_synchronize::Bool=false)
                 id_hash = @debug_block_synchronize_quick_ifelse(
                                hash(string(@__FILE__, @__LINE__)),
                                nothing
                               )
                 return quote
                     $($sync_name_internal)($id_hash, $(esc(no_synchronize)))
                 end
             end

             function $sync_name_internal(call_site, no_synchronize::Bool=false)
                 if loop_ranges[].parallel_dims == $dims
                     return
                 end
                 if !loop_ranges[].is_anyzv
                     error("Trying to change the 'anyzv' sub-region when not an an 'anyzv' "
                           * "region")
                 end
                 if !no_synchronize
                     _anyzv_subblock_synchronize(call_site)
                 end
                 loop_ranges[] = loop_ranges_store[$dims]
             end
             export $sync_macro_name
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
@begin_*_region() calls to be placed where they make logical sense, with no cost if a
call happens to be repeated (e.g. in different functions).

Calls `_block_synchronize()` to synchronize the processes operating on a shared-memory
block, unless `true` is passed as the `no_synchronize` argument.
"""
macro begin_serial_region(no_synchronize::Bool=false)
    id_hash = @debug_block_synchronize_quick_ifelse(
                   hash(string(@__FILE__, @__LINE__)),
                   nothing
                  )
    return quote
        begin_serial_region_internal($id_hash, $(esc(no_synchronize)))
    end
end
function begin_serial_region_internal(call_site, no_synchronize::Bool)
    if loop_ranges[].parallel_dims == ()
        return
    end
    if !no_synchronize
        _block_synchronize(call_site)
    end
    loop_ranges[] = loop_ranges_store[()]
end
export @begin_serial_region

"""
Run a block of code on only anysv-subblock-rank-0 of each group of processes operating on
an 'anysv' shared-memory subblock
"""
macro anysv_serial_region(blk)
    return quote
        if loop_ranges[].anysv_rank0
            $(esc(blk))
        end
    end
end
export @anysv_serial_region

"""
Run a block of code on only anyzv-subblock-rank-0 of each group of processes operating on
an 'anyzv' shared-memory subblock
"""
macro anyzv_serial_region(blk)
    return quote
        if loop_ranges[].anyzv_rank0
            $(esc(blk))
        end
    end
end
export @anyzv_serial_region

end # looping
