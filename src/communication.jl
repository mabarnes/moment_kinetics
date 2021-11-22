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

export allocate_shared, allocate_shared_keep_order, block_synchronize,
       block_rank, block_size, comm_block, comm_world, finalize_comms!,
       initialize_comms!, get_coordinate_local_range, get_species_local_range,
       global_rank, MPISharedArray

using MPI
using NamedDims
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

    struct DebugMPISharedArray{L, T, N} <: AbstractArray{T, N}
        data::NamedDimsArray{L,T,N}
        is_read::Array{Bool,N}
        is_written::Array{Bool, N}
        creation_stack_trace::String
        @debug_detect_redundant_block_synchronize begin
            previous_is_read::Array{Bool,N}
            previous_is_written::Array{Bool, N}
        end
    end

    # Constructors
    function DebugMPISharedArray(array::Array)
        dims = size(array)
        is_read = Array{Bool}(undef, dims)
        is_read .= false
        is_written = Array{Bool}(undef, dims)
        is_written .= false
        creation_stack_trace = string([string(s, "\n") for s in stacktrace()]...)
        @debug_detect_redundant_block_synchronize begin
            # Initialize as `true` so that the first call to block_synchronize() with
            # @debug_detect_redundant_block_synchronize activated does not register the
            # previous call as unnecessary
            previous_is_read = Array{Bool}(undef, dims)
            previous_is_read .= true
            previous_is_written = Array{Bool}(undef, dims)
            previous_is_written .= true
            return DebugMPISharedArray(array, is_read, is_written, creation_stack_trace,
                                       previous_is_read, previous_is_written)
        end
        return DebugMPISharedArray(array, is_read, is_written, creation_stack_trace)
    end

    # Define functions needed for AbstractArray interface
    # (https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array)
    Base.size(A::DebugMPISharedArray{L, T, N}) where {L, T, N} = size(A.data)
    function Base.getindex(A::DebugMPISharedArray{L, T, N}, I::Vararg{mk_int,N}) where {L, T, N}
        A.is_read[I...] = true
        return getindex(A.data, I...)
    end
    function Base.setindex!(A::DebugMPISharedArray{L, T, N}, v::T, I::Vararg{mk_int,N}) where {L, T, N}
        A.is_written[I...] = true
        return setindex!(A.data, v, I...)
    end
    # Overload Base.convert() so that it is forbidden to convert a DebugMPISharedArray
    # to Array. If this happens, it can cause surprising results (i.e. debug checks do
    # not run), and it should never be useful.
    # Define 3 versions here because Julia dispatches to the most specific applicable
    # method, so need to make sure these are the most specific versions, regardless of
    # how the type passed to the first argument was defined.
    function Base.convert(::Type{NamedDimsArray{L,T,N}} where {L,T,N}, a::DebugMPISharedArray)
        error("Forbidden to convert DebugMPISharedArray to Array - this would "
              * "silently disable the debug checks")
    end
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

const MPISharedArray = @debug_shared_array_ifelse(DebugMPISharedArray, NamedDimsArray)

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
function allocate_shared(T; dims...)
    return allocate_shared_keep_order(T; dims...)
end
# variant where dimensions of returned array are not sorted
function allocate_shared_keep_order(T; dims...)
    br = block_rank[]
    bs = block_size[]
    n = prod(values(dims))

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

    # values(dims) returns a NamedTuple, so need to call values() on the result
    # to get a Tuple of the actual values - see
    # https://discourse.julialang.org/t/unexpected-behaviour-of-values-for-generic-kwargs/71938
    bare_array = unsafe_wrap(Array, base_ptr, values(values(dims)))
    array = NamedDimsArray{keys(dims)}(bare_array)

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

# Need to put this before block_synchronize() so that original_error() is defined.
@debug_error_stop_all begin
    # Redefine Base.error(::String) so that it communicates the error to other
    # processes, which can pick it up in block_synchronize() so that all processes are
    # stopped. This should make interactive debugging easier, since all processes will
    # stop if there is an error, instead of hanging.  Using ^C to stop hanging processes
    # seems to mess up the MPI state so it's not possible to call, e.g.,
    # `run_moment_kinetics()` again without restarting Julia.
    #
    # Also calls finalize_comms!() on all processes when erroring, so that a later run
    # starts from a clean state. Note: if you catch the `ErrorException` raised by
    # `error()` and try to continue, most likely you will get a segfault (when
    # @debug_error_stop_all is active) because all the shared-memory arrays are deleted.
    #
    # Only want this active for debug runs, because for production runs we don't want
    # extra MPI.Allgather() calls.
    #
    # The following implementation feels like a horrible hack, so I (JTO) am worried it
    # might be fragile in the long run, but it only affects the code when debugging is
    # enabled, and only really exists for convenience - if it breaks we can just delete
    # it.
    #
    # Using 'world age' allows us to call the original Base.error(::String) from inside
    # this redefined Base.error(::String). Implementation copied from here:
    # https://discourse.julialang.org/t/how-to-call-the-original-function-when-overriding-it/56865/6
    const world_age_before_definition = Base.get_world_counter()
    original_error(message) = Base.invoke_in_world(world_age_before_definition,
                                                   Base.error, message)
    function Base.error(message::String)
        # Communicate to all processes (which pick up the messages in
        # block_synchronize()) that there was an error
        _ = MPI.Allgather(true, comm_world)

        # Clean up MPI-allocated memory before raising error
        finalize_comms!()

        original_error(message)
    end

    # This needs to be called before each blocking collective operation (e.g.
    # MPI.Barrier()), to make sure any errors that have occured on other processes are
    # communicated.
    function _gather_errors()
        # Gather a flag from all processes saying if there has been an error. If there
        # was, error here too so that all processes stop.
        all_errors = MPI.Allgather(false, comm_world)
        if any(all_errors)
            error_procs = Vector{mk_int}(undef, 0)
            for (i, flag) ∈ enumerate(all_errors)
                if flag
                    # (i-1) because MPI ranks are 0-based indices
                    push!(error_procs, i-1)
                end
            end

            # Clean up MPI-allocated memory before raising error
            finalize_comms!()

            original_error("Stop due to errors on other processes. Processes with "
                           * "errors were: $error_procs.")
        end
    end
end

@debug_detect_redundant_block_synchronize begin
    const previous_block_synchronize_stackstring = Ref{String}("")
    const debug_detect_redundant_is_active = Ref{Bool}(false)
end
"""
Call an MPI Barrier for all processors in a block.

Used to synchronise processors that are working on the same shared-memory array(s)
between operations, to avoid race conditions. Should be (much) cheaper than a global MPI
Barrier because it only requires communication within a single node.

Note: some debugging code currently assumes that if block_synchronize() is called on one
block, it is called simultaneously on all blocks. It seems likely that this will always
be true, but if it ever changes (i.e. different blocks doing totally different work),
the debugging routines need to be updated.
"""
function block_synchronize()
    @debug_error_stop_all _gather_errors()
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
        #
        # This function is used in two ways:
        # 1. To throw an error when @debug_shared_array is activated.
        # 2. To show when an error would be raised at one block_synchronize() call if
        #    the previous block_synchronize() call were removed.
        function check_array(array, array_is_read, array_is_written; check_redundant=false)
            dims = size(array)
            global_dims = (dims..., block_size[])
            @debug_error_stop_all _gather_errors()
            global_is_read = reshape(MPI.Allgather(array_is_read, comm_block),
                                     global_dims...)
            global_is_written = reshape(MPI.Allgather(array_is_written, comm_block),
                                        global_dims...)
            for i ∈ CartesianIndices(array)
                n_reads = sum(global_is_read[i, :])
                n_writes = sum(global_is_written[i, :])
                if n_writes > 1
                    if check_redundant
                        # In the @debug_detect_redundant_block_synchronize case, cannot
                        # use Base.error() (as redefined by @debug_error_stop_all),
                        # because the redefined function cleans up (deletes) the
                        # shared-memory arrays, so would cause segfaults.
                        return false
                    else
                        error("Shared memory array written at $i from multiple ranks "
                              * "between calls to block_synchronize(). Array was "
                              * "created at:\n"
                              * array.creation_stack_trace)
                    end
                elseif n_writes == 1 && n_reads > 0
                    if global_is_written[i, block_rank[] + 1]
                        read_procs = Vector{mk_int}(undef, 0)
                        for (r, is_read) ∈ enumerate(global_is_read[i, :])
                            if r == block_rank[] + 1
                                continue
                            elseif is_read
                                push!(read_procs, r)
                            end
                        end
                        if length(read_procs) > 0
                            if check_redundant
                                # In the @debug_detect_redundant_block_synchronize case,
                                # cannot use Base.error() (as redefined by
                                # @debug_error_stop_all), because the redefined function
                                # cleans up (deletes) the shared-memory arrays, so would
                                # cause segfaults.
                                return false
                            else
                                # 'rank' is 0-based, but read_procs was 1-based, so
                                # correct
                                read_procs .-= 1
                                error("Shared memory array was written at $i on rank "
                                      * "$(block_rank[]) but read from ranks "
                                      * "$read_procs Array was created at:\n"
                                      * array.creation_stack_trace)
                            end
                        end
                    end
                end
            end
            return true
        end

        @debug_detect_redundant_block_synchronize previous_was_unnecessary = true
        for (arraynum, array) ∈ enumerate(global_debugmpisharedarray_store)

            check_array(array, array.is_read, array.is_written)

            @debug_detect_redundant_block_synchronize begin
                # debug_detect_redundant_is_active[] is set to true at the beginning of
                # time_advance!() so that we do not do these checks during
                # initialization: they cause problems with @debug_initialize_NaN during
                # array allocation; generally it does not matter if there are a few
                # extra block_synchronize() calls during initialization, so it is not
                # worth the effort to trim them down to the absolute minimum.
                if debug_detect_redundant_is_active[]
                    # Note `||` cannot broadcast because it 'short-circuits', so only
                    # accepts scalar Bool arguments. Therefore we need to use `.|`
                    # (broadcasted, bit-wise or) to combine arrays of Bool.  Need to
                    # explicitly convert the result to a Vector{Bool}, because by
                    # default the result is a BitVector, and passing BitVector to
                    # MPI.Allgather() causes an error
                    combined_is_read = Array{Bool}(array.previous_is_read .|
                                                   array.is_read)
                    combined_is_written = Array{Bool}(array.previous_is_written .|
                                                      array.is_written)

                    if !check_array(array, combined_is_read, combined_is_written,
                                    check_redundant=true)
                        # If there was a failure for at least one array, the previous
                        # block_synchronize was necessary - if the previous call was not
                        # there, for this array array.is_read and array.is_written would
                        # have the values of combined_is_read and combined_is_written,
                        # and would fail the check_array() above this
                        # @debug_detect_redundant_block_synchronize block.
                        previous_was_unnecessary = false
                    end

                    array.previous_is_read .= array.is_read
                    array.previous_is_written .= array.is_written
                end
            end

            array.is_read .= false
            array.is_written .= false
        end
        @debug_detect_redundant_block_synchronize begin
            if debug_detect_redundant_is_active[]
                # Check the previous call was unnecessary on all processes, not just
                # this one
                previous_was_unnecessary = MPI.Allreduce(previous_was_unnecessary,
                                                         MPI.Op(&, Bool), comm_world)

                if (previous_was_unnecessary && global_size[] > 1)
                    # The intention of this debug block is to detect when calls to
                    # block_synchronize() are not necessary and can be removed. It's not
                    # obvious that this will always work - it might be that a call to
                    # block_synchronize() is necessary with some options, but not
                    # necessary with others. Hopefully it will be possible to handle
                    # this by moving the block_synchronize() call inside appropriate
                    # if-clauses. If not, it might be necessary to define something like
                    # block_synchronize_ignore_redundant() to skip this check because
                    # the check is ambiguous.
                    #
                    # If we are running in serial (global_size[] == 1), then none of the
                    # block_synchronize() calls are 'necessary', so this check is not
                    # useful.
                    error("Previous call to block_synchronize() was not necessary. "
                          * "Call was from:\n"
                          * "$(previous_block_synchronize_stackstring[])")
                end

                st = stacktrace()
                stackstring = string([string(s, "\n") for s ∈ st]...)
                previous_block_synchronize_stackstring[] = stackstring
            end
        end

        @debug_error_stop_all _gather_errors()
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

    @debug_detect_redundant_block_synchronize begin
        debug_detect_redundant_is_active[] = false
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
