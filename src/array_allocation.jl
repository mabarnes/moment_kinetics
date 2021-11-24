module array_allocation

export allocate_float, allocate_int, allocate_complex, allocate_bool,
       allocate_shared, drop_dim

using NamedDims

using ..type_definitions: mk_float, mk_int
using ..communication: allocate_shared, block_rank
using ..debugging
@debug_initialize_NaN using ..communication: block_synchronize

# Utility function to create (at compile-time) a list of dimensions with one
# entry dropped. Val{} arguments used so that the whole thing should be
# evaluated at compile time.
function drop_dim(::Val{to_drop}, ::Val{dims}) where {to_drop,dims}
    return filter(d->d!=to_drop, dims)
end

# allocate array with dimensions given by dims and entries of type Bool
function allocate_bool(::Val{dims}; dim_sizes...) where dims
    return array = NamedDimsArray{dims}(
        Array{Bool}(undef, (dim_sizes[d] for d in dims)...))
end
# variant where array is in shared memory for all processors in the 'block'
function allocate_shared_bool(::Val{dims}; dim_sizes...) where dims
    return array = allocate_shared(Bool, dims; dim_sizes...)
end

# allocate 1d array with dimensions given by dims and entries of type mk_int
function allocate_int(::Val{dims}; dim_sizes...) where dims
    return array = NamedDimsArray{dims}(
        Array{mk_int}(undef, (dim_sizes[d] for d in dims)...))
end
# variant where array is in shared memory for all processors in the 'block'
function allocate_shared_int(dims; dim_sizes...)
    return array = allocate_shared(mk_int, dims; dim_sizes...)
end

# allocate array with dimensions given by dims and entries of type mk_float
function allocate_float(::Val{dims}; dim_sizes...) where dims
    array = NamedDimsArray{dims}(
        Array{mk_float}(undef, (dim_sizes[d] for d in dims)...))
    @debug_initialize_NaN begin
        array .= NaN
    end
    return array
end
# variant where array is in shared memory for all processors in the 'block'
function allocate_shared_float(dims; dim_sizes...)
    array = allocate_shared(mk_float, dims; dim_sizes...)
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
function allocate_complex(::Val{dims}; dim_sizes...) where dims
    # values(dims) returns a NamedTuple, so need to call values() on the result
    # to get a Tuple of the actual values - see
    # https://discourse.julialang.org/t/unexpected-behaviour-of-values-for-generic-kwargs/71938
    array = NamedDimsArray{dims}(
        Array{Complex{mk_float}}(undef, (dim_sizes[d] for d in dims)...))
    @debug_initialize_NaN begin
        array .= NaN
    end
    return array
end
# variant where array is in shared memory for all processors in the 'block'
function allocate_shared_complex(dims; dim_sizes...)
    array = allocate_shared(Complex{mk_float}, dims; dim_sizes...)
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
