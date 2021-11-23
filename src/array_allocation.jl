module array_allocation

export allocate_float, allocate_int, allocate_complex, allocate_bool, allocate_shared

using NamedDims

using ..type_definitions: mk_float, mk_int
using ..communication: allocate_shared, allocate_shared_keep_order, block_rank
using ..debugging
@debug_initialize_NaN using ..communication: block_synchronize

# Check dimension order so that all arrays have dimensions in the same order,
# unless they are explicitly allocated with a different order.
function check_dim_order(dims::Tuple{Symbol})
    # Dummy implementation as a placeholder - implement this later
    return nothing
end

# allocate array with dimensions given by dims and entries of type Bool
function allocate_bool(; dims...)
    # values(dims) returns a NamedTuple, so need to call values() on the result
    # to get a Tuple of the actual values - see
    # https://discourse.julialang.org/t/unexpected-behaviour-of-values-for-generic-kwargs/71938
    return array = NamedDimsArray{keys(dims)}(
        Array{Bool}(undef, values(values(dims))...))
end
# variant where dimensions of returned array are not sorted
function allocate_bool_keep_order(; dims...)
    # values(dims) returns a NamedTuple, so need to call values() on the result
    # to get a Tuple of the actual values - see
    # https://discourse.julialang.org/t/unexpected-behaviour-of-values-for-generic-kwargs/71938
    return array = NamedDimsArray{keys(dims)}(
        Array{Bool}(undef, values(values(dims))...))
end
# variant where array is in shared memory for all processors in the 'block'
function allocate_shared_bool(; dims...)
    return array = allocate_shared(Bool; dims...)
end
# variant where array is in shared memory for all processors in the 'block' and
# dimensions of returned array are not sorted
function allocate_shared_bool_keep_order(; dims...)
    return array = allocate_shared_keep_order(Bool; dims...)
end

# allocate 1d array with dimensions given by dims and entries of type mk_int
function allocate_int(; dims...)
    # values(dims) returns a NamedTuple, so need to call values() on the result
    # to get a Tuple of the actual values - see
    # https://discourse.julialang.org/t/unexpected-behaviour-of-values-for-generic-kwargs/71938
    return array = NamedDimsArray{keys(dims)}(
        Array{mk_int}(undef, values(values(dims))...))
end
# variant where dimensions of returned array are not sorted
function allocate_int_keep_order(; dims...)
    # values(dims) returns a NamedTuple, so need to call values() on the result
    # to get a Tuple of the actual values - see
    # https://discourse.julialang.org/t/unexpected-behaviour-of-values-for-generic-kwargs/71938
    return array = NamedDimsArray{keys(dims)}(
        Array{mk_int}(undef, values(values(dims))...))
end
# variant where array is in shared memory for all processors in the 'block'
function allocate_shared_int(; dims...)
    return array = allocate_shared(mk_int; dims...)
end
# variant where array is in shared memory for all processors in the 'block' and
# dimensions of returned array are not sorted
function allocate_shared_int_keep_order(; dims...)
    return array = allocate_shared_keep_order(mk_int; dims...)
end

# allocate array with dimensions given by dims and entries of type mk_float
function allocate_float(; dims...)
    # values(dims) returns a NamedTuple, so need to call values() on the result
    # to get a Tuple of the actual values - see
    # https://discourse.julialang.org/t/unexpected-behaviour-of-values-for-generic-kwargs/71938
    array = NamedDimsArray{keys(dims)}(
        Array{mk_float}(undef, values(values(dims))...))
    @debug_initialize_NaN begin
        array .= NaN
    end
    return array
end
# variant where dimensions of returned array are not sorted
function allocate_float_keep_order(; dims...)
    # values(dims) returns a NamedTuple, so need to call values() on the result
    # to get a Tuple of the actual values - see
    # https://discourse.julialang.org/t/unexpected-behaviour-of-values-for-generic-kwargs/71938
    array = NamedDimsArray{keys(dims)}(
        Array{mk_float}(undef, values(values(dims))...))
    @debug_initialize_NaN begin
        array .= NaN
    end
    return array
end
# variant where array is in shared memory for all processors in the 'block'
function allocate_shared_float(; dims...)
    array = allocate_shared(mk_float; dims...)
    @debug_initialize_NaN begin
        # Initialize as NaN to try and catch use of uninitialized values
        if block_rank[] == 0
            array .= NaN
        end
        block_synchronize()
    end
    return array
end
# variant where array is in shared memory for all processors in the 'block' and
# dimensions of returned array are not sorted
function allocate_shared_float_keep_order(; dims...)
    array = allocate_shared(mk_float; dims...)
    @debug_initialize_NaN begin
        # Initialize as NaN to try and catch use of uninitialized values
        if block_rank[] == 0
            array .= NaN
        end
        block_synchronize()
    end
    return array
end

# allocate 1d array with dimensions given by dims and entries of type Complex{mk_float}
function allocate_complex(; dims...)
    # values(dims) returns a NamedTuple, so need to call values() on the result
    # to get a Tuple of the actual values - see
    # https://discourse.julialang.org/t/unexpected-behaviour-of-values-for-generic-kwargs/71938
    array = NamedDimsArray{keys(dims)}(
        Array{Complex{mk_float}}(undef, values(values(dims))...))
    @debug_initialize_NaN begin
        array .= NaN
    end
    return array
end
# variant where dimensions of returned array are not sorted
function allocate_complex_keep_order(; dims...)
    # values(dims) returns a NamedTuple, so need to call values() on the result
    # to get a Tuple of the actual values - see
    # https://discourse.julialang.org/t/unexpected-behaviour-of-values-for-generic-kwargs/71938
    array = NamedDimsArray{keys(dims)}(
        Array{Complex{mk_float}}(undef, values(values(dims))...))
    @debug_initialize_NaN begin
        array .= NaN
    end
    return array
end
# variant where array is in shared memory for all processors in the 'block'
function allocate_shared_complex(; dims...)
    array = allocate_shared(Complex{mk_float}; dims...)
    @debug_initialize_NaN begin
        # Initialize as NaN to try and catch use of uninitialized values
        if block_rank[] == 0
            array .= NaN
        end
        block_synchronize()
    end
    return array
end
# variant where array is in shared memory for all processors in the 'block' and
# dimensions of returned array are not sorted
function allocate_shared_complex_keep_order(; dims...)
    array = allocate_shared(Complex{mk_float}; dims...)
    @debug_initialize_NaN begin
        # Initialize as NaN to try and catch use of uninitialized values
        if block_rank[] == 0
            array .= NaN
        end
        block_synchronize()
    end
    return array
end

end
