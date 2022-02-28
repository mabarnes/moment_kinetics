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
function allocate_shared_bool(dims...)
    return array = allocate_shared(Bool, dims)
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
function allocate_shared_int(dims...)
    return array = allocate_shared(mk_int, dims)
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
function allocate_shared_float(dims...)
    array = allocate_shared(mk_float, dims)
    @debug_initialize_NaN begin
        # Initialize as NaN to try and catch use of uninitialized values
        if block_rank[] == 0
            array .= NaN
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
function allocate_shared_complex(dims...)
    array = allocate_shared(Complex{mk_float}, dims)
    @debug_initialize_NaN begin
        # Initialize as NaN to try and catch use of uninitialized values
        if block_rank[] == 0
            array .= NaN
        end
        _block_synchronize()
    end
    return array
end

end
