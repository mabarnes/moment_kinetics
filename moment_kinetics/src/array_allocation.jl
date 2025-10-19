"""
"""
module array_allocation

export allocate_float, allocate_int, allocate_complex, allocate_bool, allocate_shared

using ..type_definitions: mk_float, mk_int
using ..communication
using ..debugging
using ..moment_kinetics_structs: coordinate
@debug_initialize_NaN using ..communication: block_rank

using MPI

# For debugging of shared-memory arrays, we need to specify the names of the dimensions
# of the array being allocated. For consistency, we also provide a similar interface for
# non-shared-memory arrays, even though for the moment the names do nothing.

# Allocate a non-shared memory array from dimension sizes
function _allocate_array(type::Type, dims::mk_int...)
    return Array{type}(undef, dims...)
end

# Overload to handle coordinate, NamedTuple, or Pair arguments
function _allocate_array(type::Type, dims...)
    function get_int(a)
        if isa(a, coordinate)
            return a.n
        elseif isa(a, NamedTuple)
            return mk_int(a.n)
        elseif isa(a, Integer)
            return mk_int(a)
        elseif isa(a, Pair)
            return get_int(a[2])
        else
            error("Unrecognised type of argument $a")
            error("Incorrect argument $a to `_allocate_array`. Arguments should be "
                  * "coordinate, coordinate-like NamedTuple, `:name=>n` "
                  * "(`Pair{Symbol,mk_int}`), or mk_int.")
        end
    end
    return _allocate_array(type, (get_int(d) for d ∈ dims)...)
end

# Overload to handle keyword arguments
function _allocate_array(type::Type; kwargs...)
    if length(kwargs) == 0
        # Special handling if there are no kwargs to avoid stack overflow.
        return Array{type}(undef)
    end
    return _allocate_array(type, (v for v ∈ values(kwargs))...)
end

"""
allocate array with dimensions given by dims and entries of type Bool
"""
function allocate_bool(dims...; kwargs...)
    return _allocate_array(Bool, dims...; kwargs...)
end
 
"""
variant where array is in shared memory for all processors in the 'block'
"""
function allocate_shared_bool(dims...; comm=nothing, maybe_debug=true)
    return array = allocate_shared(Bool, comm, maybe_debug, dims...)
end

"""
allocate 1d array with dimensions given by dims and entries of type mk_int
"""
function allocate_int(dims...; kwargs...)
    return _allocate_array(mk_int, dims...; kwargs...)
end

"""
variant where array is in shared memory for all processors in the 'block'
"""
function allocate_shared_int(dims...; comm=nothing, maybe_debug=true)
    return array = allocate_shared(mk_int, comm, maybe_debug, dims...)
end

"""
allocate array with dimensions given by dims and entries of type mk_float
"""
function allocate_float(dims...; kwargs...)
    array = _allocate_array(mk_float, dims...; kwargs...)
    @debug_initialize_NaN begin
        array .= NaN
    end
    return array
end

"""
variant where array is in shared memory for all processors in the 'block'
"""
function allocate_shared_float(dims...; comm=nothing, maybe_debug=true)
    array = allocate_shared(mk_float, comm, maybe_debug, dims...)
    @debug_initialize_NaN begin
        # Initialize as NaN to try and catch use of uninitialized values
        if comm === nothing
            comm_rank = block_rank[]
            this_comm = comm_block[]
        elseif comm == MPI.COMM_NULL
            comm_rank = -1
            this_comm = nothing
        else
            # Get MPI.Comm_rank when comm is not nothing
            comm_rank = MPI.Comm_rank(comm)
            this_comm = comm
        end
        if comm_rank == 0
            array .= NaN
            @debug_track_initialized begin
                # Track initialization as if the array was not initialized to NaN
                array.is_initialized .= false
                # Track usage as if this array has not been written
                array.is_written .= false
            end
        end
        if this_comm !== nothing
            MPI.Barrier(this_comm)
        end
    end
    return array
end

"""
allocate 1d array with dimensions given by dims and entries of type Complex{mk_float}
"""
function allocate_complex(dims...; kwargs...)
    array = _allocate_array(Complex{mk_float}, dims...; kwargs...)
    @debug_initialize_NaN begin
        array .= NaN
    end
    return array
end

"""
variant where array is in shared memory for all processors in the 'block'
"""
function allocate_shared_complex(dims...; comm=nothing, maybe_debug=true)
    array = allocate_shared(Complex{mk_float}, comm, maybe_debug, dims...)
    @debug_initialize_NaN begin
        # Initialize as NaN to try and catch use of uninitialized values
        if comm === nothing
            comm_rank = block_rank[]
            this_comm = comm_block[]
        elseif comm == MPI.COMM_NULL
            comm_rank = -1
            this_comm = nothing
        else
            # Get MPI.Comm_rank when comm is not nothing
            comm_rank = MPI.Comm_rank(comm)
            this_comm = comm
        end
        if comm_rank == 0
            array .= NaN
            @debug_track_initialized begin
                # Track initialization as if the array was not initialized to NaN
                array.is_initialized .= false
            end
        end
        if this_comm !== nothing
            MPI.Barrier(this_comm)
        end
    end
    return array
end

end
