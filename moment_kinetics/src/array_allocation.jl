"""
"""
module array_allocation

export allocate_float, allocate_int, allocate_complex, allocate_bool, allocate_shared

using ..type_definitions: mk_float, mk_int
using ..communication
using ..debugging
@debug_initialize_NaN using ..communication: block_rank, _block_synchronize

"""
allocate array with dimensions given by dims and entries of type Bool
"""
function allocate_bool(dims...)
    return array = Array{Bool}(undef, dims...)
end
 
"""
variant where array is in shared memory for all processors in the 'block'
"""
function allocate_shared_bool(dims...; comm=nothing)
    return array = allocate_shared(Bool, dims; comm=comm)
end

"""
allocate 1d array with dimensions given by dims and entries of type mk_int
"""
function allocate_int(dims...)
    return array = Array{mk_int}(undef, dims...)
end

"""
variant where array is in shared memory for all processors in the 'block'
"""
function allocate_shared_int(dims...; comm=nothing)
    return array = allocate_shared(mk_int, dims; comm=comm)
end

"""
allocate array with dimensions given by dims and entries of type mk_float
"""
function allocate_float(dims...)
    array = Array{mk_float}(undef, dims...)
    @debug_initialize_NaN begin
        array .= NaN
    end
    return array
end

"""
variant where array is in shared memory for all processors in the 'block'
"""
function allocate_shared_float(dims...; comm=nothing)
    array = allocate_shared(mk_float, dims; comm=comm)
    @debug_initialize_NaN begin
        # Initialize as NaN to try and catch use of uninitialized values
        if block_rank[] == 0
            array .= NaN
            @debug_track_initialized begin
                # Track initialization as if the array was not initialized to NaN
                array.is_initialized .= false
            end
        end
        _block_synchronize()
    end
    return array
end

"""
allocate 1d array with dimensions given by dims and entries of type Complex{mk_float}
"""
function allocate_complex(dims...)
    array = Array{Complex{mk_float}}(undef, dims...)
    @debug_initialize_NaN begin
        array .= NaN
    end
    return array
end

"""
variant where array is in shared memory for all processors in the 'block'
"""
function allocate_shared_complex(dims...; comm=nothing)
    array = allocate_shared(Complex{mk_float}, dims; comm=comm)
    @debug_initialize_NaN begin
        # Initialize as NaN to try and catch use of uninitialized values
        if block_rank[] == 0
            array .= NaN
            @debug_track_initialized begin
                # Track initialization as if the array was not initialized to NaN
                array.is_initialized .= false
            end
        end
        _block_synchronize()
    end
    return array
end

end
