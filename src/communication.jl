"""
Communication functions and setup

Split the grid into 'blocks'. Each block can use shared memory (MPI shared memory
arrays). At the moment only works with a single 'block' containing the whole grid -
eventually add more MPI communication functions to communicate between blocks. A block
should probably be a 'NUMA region' for maximum efficiency.

Note: charge-exchange collisions loop over neutral species for each ion species. At the
moment this loop is not parallelised (although it could be, by introducing some more
loop ranges), as at the moment we only run with 1 ion species and 1 neutral species.
"""
module communication

export allocate_shared_float, block_synchronize, block_rank, block_size, comm_block,
       comm_world, finalize_comms!, initialize_comms!, get_coordinate_local_range,
       get_species_local_range, global_rank, MPISharedArray

using MPI
using SHA

using ..debugging
using ..type_definitions: mk_float, mk_int

const comm_world = MPI.Comm()
const comm_block = MPI.Comm()

# Use Ref for these variables so that they can be made `const` (so have a definite
# type), but contain a value assigned at run-time.
const global_rank = Ref{mk_int}()
const global_size = Ref{mk_int}()
const block_rank = Ref{mk_int}()
const block_size = Ref{mk_int}()

const global_Win_store = Vector{MPI.Win}(undef, 0)

function __init__()
    MPI.Init()

    comm_world.val = MPI.COMM_WORLD.val

    # For now, just use comm_world. In future should probably somehow figure out which
    # processors are in a 'NUMA region'... OpenMPI has a special communicator for this,
    # but MPICH doesn't seem to. Might be simplest to rely on user input??
    comm_block.val = comm_world.val

    global_rank[] = MPI.Comm_rank(comm_world)
    global_size[] = MPI.Comm_size(comm_world)
    block_rank[] = MPI.Comm_rank(comm_block)
    block_size[] = MPI.Comm_size(comm_block)
end

@debug_shared_array begin
    """
    Special type for debugging race conditions in accesses to shared-memory arrays.
    Only used if debugging._debug_level is high enough.
    """

    export DebugMPISharedArray

    struct DebugMPISharedArray{T, N} <: AbstractArray{T, N}
        data::Array{T,N}
        is_read::Array{Bool,N}
        is_written::Array{Bool, N}
        creation_stack_trace::String
    end

    # Constructors
    function DebugMPISharedArray(array::Array)
        dims = size(array)
        is_read = Array{Bool}(undef, dims)
        is_read .= false
        is_written = Array{Bool}(undef, dims)
        is_written .= false
        creation_stack_trace = string([string(s, "\n") for s in stacktrace()]...)
        return DebugMPISharedArray(array, is_read, is_written, creation_stack_trace)
    end

    # Define functions needed for AbstractArray interface
    # (https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array)
    Base.size(A::DebugMPISharedArray{T, N}) where {T, N} = size(A.data)
    function Base.getindex(A::DebugMPISharedArray{T, N}, I::Vararg{mk_int,N}) where {T, N}
        A.is_read[I...] = true
        return getindex(A.data, I...)
    end
    function Base.setindex!(A::DebugMPISharedArray{T, N}, v::T, I::Vararg{mk_int,N}) where {T, N}
        A.is_written[I...] = true
        return setindex!(A.data, v, I...)
    end
    # Overload Base.convert() so that it is forbidden to convert a DebugMPISharedArray
    # to Array. If this happens, it can cause surprising results (i.e. debug checks do
    # not run), and it should never be useful.
    # Define 3 versions here because Julia dispatches to the most specific applicable
    # method, so need to make sure these are the most specific versions, regardless of
    # how the type passed to the first argument was defined.
    function Base.convert(::Type{Array{T,N}} where {T,N}, a::DebugMPISharedArray)
        error("Forbidden to convert DebugMPISharedArray to Array - this would "
              * "silently disable the debug checks")
    end
    function Base.convert(::Type{Array{T}} where {T}, a::DebugMPISharedArray)
        error("Forbidden to convert DebugMPISharedArray to Array - this would "
              * "silently disable the debug checks")
    end
    function Base.convert(::Type{Array}, a::DebugMPISharedArray)
        error("Forbidden to convert DebugMPISharedArray to Array - this would "
              * "silently disable the debug checks")
    end

    # Keep a global Vector of references to all created DebugMPISharedArray
    # instances, so their is_read and is_written members can be checked and
    # reset by block_synchronize()
    const global_debugmpisharedarray_store = Vector{DebugMPISharedArray}(undef, 0)
end

const MPISharedArray = @debug_shared_array_ifelse(DebugMPISharedArray, Array)

"""
Get a shared-memory array of `mk_float` (shared by all processes in a 'block')

Create a shared-memory array using `MPI.Win_allocate_shared()`. Pointer to the memory
allocated is wrapped in a Julia array. Memory is not managed by the Julia array though.
A reference to the `MPI.Win` needs to be freed - this is done by saving the `MPI.Win`
into a `Vector` in the `Communication` module, which has all its entries freed by the
`finalize_comms!()` function, which should be called when `moment_kinetics` is done
running a simulation/test.

Arguments
---------
dims - mk_int or Tuple{mk_int}
    Dimensions of the array to be created. Dimensions passed define the size of the
    array which is being handled by the 'block' (rather than the global array, or a
    subset for a single process).

Returns
-------
Array{mk_float}
"""
function allocate_shared(T, dims)
    br = block_rank[]
    bs = block_size[]
    n = prod(dims)

    if br == 0
        # Allocate points on rank-0 for simplicity
        n_local = n
    else
        n_local = 0
    end

    @debug_shared_array_allocate begin
        # Check that allocate_shared was called from the same place on all ranks
        st = stacktrace()
        stackstring = string([string(s, "\n") for s ∈ st]...)

        # Only include file and line number in the string that we hash so that
        # function calls with different signatures are not seen as different
        # (e.g. time_advance!() with I/O arguments on rank-0 but not on other
        # ranks).
        signaturestring = string([string(s.file, s.line) for s ∈ st]...)

        hash = sha256(signaturestring)
        all_hashes = MPI.Allgather(hash, comm_block)
        l = length(hash)
        for i ∈ 1:length(all_hashes)÷l
            if all_hashes[(i - 1) * l + 1: i * l] != hash
                error("allocate_shared() called inconsistently\n",
                      "rank $(block_rank[]) called from:\n",
                      stackstring)
            end
        end
    end

    win, ptr = MPI.Win_allocate_shared(T, n_local, comm_block)

    # Array is allocated contiguously, but `ptr` points to the 'locally owned' part.
    # We want to use as a shared array, so want to wrap the entire shared array.
    # Get start pointer of array from rank-0 process. Cannot use ptr, as this
    # is null when n_local=0.
    _, _, base_ptr = MPI.Win_shared_query(win, 0)
    base_ptr = Ptr{T}(base_ptr)

    if base_ptr == Ptr{Nothing}(0)
        error("Got null pointer when trying to allocate shared array")
    end

    # Don't think `win::MPI.Win` knows about the type of the pointer (its concrete type
    # is something like `MPI.Win(Ptr{Nothing} @0x00000000033affd0)`), so it's fine to
    # put them all in the same global_Win_store - this won't introduce type instability
    push!(global_Win_store, win)

    array = unsafe_wrap(Array, base_ptr, dims)

    @debug_shared_array begin
        # If @debug_shared_array is active, create DebugMPISharedArray instead of Array
        debug_array = DebugMPISharedArray(array)
        push!(global_debugmpisharedarray_store, debug_array)
        return debug_array
    end

    return array
end

"""
Get local range of species indices when splitting a loop over processes in a block

Arguments
---------
n_species : mk_int
    Number of species.
offset : mk_int, default 0
    Start the species index range at offset+1, instead of 1. Used to create ranges over
    neutral species.

Returns
-------
UnitRange{mk_int}
    Range of species indices to iterate over on this process
Bool
    Is this process the first in the group iterating over a species?
"""
function get_species_local_range(n_species, offset=0)
    bs = block_size[]
    br = block_rank[]

    if n_species >= bs
        # More species than processes (or same number) so split up species amoung
        # processes
        if n_species % bs != 0
            error("Number of species ($n_species) bigger than block size ($bs) "
                  * "but block size does not divide n_species.")
        end

        n_local = n_species ÷ bs
        return mk_int(br * n_local + 1 + offset):mk_int((br + 1) * n_local + offset), true
    else
        # More processes than species, so assign group of processes to each species
        if bs % n_species != 0
            error("Block size ($bs) is bigger than number of species "
                  * "($n_species) but n_species does not divide block size")
        end
        group_size = bs ÷ n_species
        group_rank = br % group_size
        species_ind = br ÷ group_size + 1
        return mk_int(species_ind + offset):mk_int(species_ind + offset), group_rank == 0
    end
end

"""
Get local range of (outer-loop) coordinate indices when splitting a loop over processes
in a block
"""
function get_coordinate_local_range(n, n_species)
    n_procs = block_size[]
    if n_species >= n_procs
        # No need to split loop over coordinate - will only run on a single processor
        # anyway
        return 1:n
    elseif n_procs % n_species != 0
        error("Number of species must divide n_procs when n_procs>n_species")
    end

    # Split processors into sub-blocks, where each sub-block gets one species.
    n_sub_block_procs = n_procs ÷ n_species
    sub_block_rank = block_rank[] % n_sub_block_procs

    # Assign either (n÷n_sub_block_procs) or (n÷n_sub_block_procs+1) points to each
    # processor, with the lower number (n÷n_sub_block_procs) on lower-number processors,
    # because the root process might have slightly more work to do in general.
    # This calculation is not at all optimized, but is not going to take long, and is
    # only done in initialization, so it is more important to be simple and robust.
    remaining = n
    done = false
    n_points_for_proc = zeros(mk_int, n_sub_block_procs)
    while !done
        for i ∈ n_sub_block_procs:-1:1
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
    #points_per_proc = div(n, n_sub_block_procs, RoundUp)
    #remaining = n
    #n_points_for_proc = zeros(mk_int, n_sub_block_procs)
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
Call an MPI Barrier for all processors in a block.

Used to synchronise processors that are working on the same shared-memory array(s)
between operations, to avoid race conditions. Should be (much) cheaper than a global MPI
Barrier because it only requires communication within a single node.
"""
function block_synchronize()
    MPI.Barrier(comm_block)

    @debug_block_synchronize begin
        st = stacktrace()
        stackstring = string([string(s, "\n") for s ∈ st]...)

        # Only include file and line number in the string that we hash so that
        # function calls with different signatures are not seen as different
        # (e.g. time_advance!() with I/O arguments on rank-0 but not on other
        # ranks).
        signaturestring = string([string(s.file, s.line) for s ∈ st]...)

        hash = sha256(signaturestring)
        all_hashes = reshape(MPI.Allgather(hash, comm_block), length(hash),
                             block_size[])
        for i ∈ 1:block_size[]
            h = all_hashes[:, i]
            if h != hash
                error("block_synchronize() called inconsistently\n",
                      "rank $(block_rank[]) called from:\n",
                      stackstring)
            end
        end
    end

    @debug_shared_array begin
        # Check for potential race conditions:
        # * Between block_synchronize() any element of an array should be written to by
        #   at most one rank.
        # * If an element is not written to, any rank can read it.
        # * If an element is written to, only the rank that writes to it should read it.
        for (arraynum, array) ∈ enumerate(global_debugmpisharedarray_store)
            dims = size(array)
            global_dims = (dims..., block_size[])
            global_is_read = reshape(MPI.Allgather(array.is_read, comm_block),
                                     global_dims...)
            global_is_written = reshape(MPI.Allgather(array.is_written, comm_block),
                                        global_dims...)
            for i ∈ CartesianIndices(array)
                n_reads = sum(global_is_read[i, :])
                n_writes = sum(global_is_written[i, :])
                if n_writes > 1
                    error("Shared memory array written at $i from multiple ranks "
                          * "between calls to block_synchronize(). Array was created "
                          * "at:\n"
                          * array.creation_stack_trace)
                elseif n_writes == 1 && n_reads > 0
                    if global_is_written[i, block_rank[] + 1]
                        read_procs = Vector{Int}(undef, 0)
                        for (r, is_read) ∈ enumerate(global_is_read[i, :])
                            if r == block_rank[] + 1
                                continue
                            elseif is_read
                                push!(read_procs, r)
                            end
                        end
                        if length(read_procs) > 0
                            # 'rank' is 0-based, but read_procs was 1-based, so correct
                            read_procs .-= 1
                            error("Shared memory array was written at $i on rank "
                                  * "$(block_rank[]) but read from ranks $read_procs "
                                  * "Array was created at:\n"
                                  * array.creation_stack_trace)
                        end
                    end
                end
            end

            array.is_read .= false
            array.is_written .= false
        end

        MPI.Barrier(comm_block)
    end
end

"""
Set up communications

Check that global variables are in the correct state (i.e. caches were emptied
correctly if they were used before).

Also does some set up for debugging routines, if they are active.
"""
function initialize_comms!()
    if length(global_Win_store) > 0
        free_shared_arrays()
    end

    return nothing
end

"""
Clean up from communications

Do any needed clean-up for MPI, etc. Does not call `MPI.Finalize()` - this is called
anyway when Julia exits, and we do not want to call it explicitly so that multiple runs
can be done in a single Julia session.

Frees any shared-memory arrays.
"""
function finalize_comms!()
    free_shared_arrays()

    return nothing
end

function free_shared_arrays()
    @debug_shared_array resize!(global_debugmpisharedarray_store, 0)

    for w ∈ global_Win_store
        MPI.free(w)
    end
    resize!(global_Win_store, 0)

    return nothing
end

end # communication
