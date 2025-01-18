"""
    select_slice(variable::AbstractArray, dims::Symbol...; input=nothing, it=nothing,
                 is=1, ir=nothing, iz=nothing, ivperp=nothing, ivpa=nothing,
                 ivzeta=nothing, ivr=nothing, ivz=nothing)

Returns a slice of `variable` that includes only the dimensions given in `dims...`, e.g.
```
select_slice(variable, :t, :r)
```
to get a two dimensional slice with t- and r-dimensions.

Any other dimensions present in `variable` have a single point selected. By default this
point is set by the options in `input` (which must be a NamedTuple) (or the final point
for time or the size of the dimension divided by 3 if `input` is not given). These
defaults can be overridden using the keyword arguments `it`, `is`, `ir`, `iz`, `ivperp`,
`ivpa`, `ivzeta`, `ivr`, `ivz`. Ranges can also be passed to these keyword arguments for
the 'kept dimensions' in `dims` to select a subset of those dimensions.

This function only recognises what the dimensions of `variable` are by the number of
dimensions in the array. It assumes that either the variable has already been sliced to
the correct dimensions (if `ndims(variable) == length(dims)` it just returns `variable`)
or that `variable` has the full number of dimensions it could have (i.e. 'field' variables
have 3 dimensions, 'moment' variables 4, 'ion distribution function' variables 6 and
'neutral distribution function' variables 7).
"""
function select_slice end

function select_slice(variable::AbstractMKArray{T,1}, dims::Symbol...; input=nothing,
                      is=nothing, kwargs...) where T
    if length(dims) > 1
        error("Tried to get a slice of 1d variable with dimensions $dims")
    elseif length(dims) < 1
        error("1d variable must have already been sliced, so don't know what the dimensions are")
    else
        # Array is not a standard shape, so assume it is already sliced to the right 2
        # dimensions
        return variable
    end
end

function select_slice(variable::AbstractMKArray{T,2}, dims::Symbol...; input=nothing,
                      is=nothing, kwargs...) where T
    if length(dims) > 2
        error("Tried to get a slice of 2d variable with dimensions $dims")
    elseif length(dims) < 2
        error("2d variable must have already been sliced, so don't know what the dimensions are")
    else
        # Array is not a standard shape, so assume it is already sliced to the right 2
        # dimensions
        return variable
    end
end

function select_slice(variable::AbstractMKArray{T,3}, dims::Symbol...; input=nothing,
                      it=nothing, is=nothing, ir=nothing, iz=nothing, kwargs...) where T
    # Array is (z,r,t)

    if length(dims) > 3
        error("Tried to get a slice of 3d variable with dimensions $dims")
    end

    if it !== nothing
        it0 = it
    elseif input === nothing || :it0 ∉ input
        it0 = size(variable, 3)
    else
        it0 = input.it0
    end
    if ir !== nothing
        ir0 = ir
    elseif input === nothing || :ir0 ∉ input
        ir0 = max(size(variable, 2) ÷ 3, 1)
    else
        ir0 = input.ir0
    end
    if iz !== nothing
        iz0 = iz
    elseif input === nothing || :iz0 ∉ input
        iz0 = max(size(variable, 1) ÷ 3, 1)
    else
        iz0 = input.iz0
    end

    slice = variable
    if :t ∉ dims || it !== nothing
        slice = selectdim(slice, 3, it0)
    end
    if :r ∉ dims || ir !== nothing
        slice = selectdim(slice, 2, ir0)
    end
    if :z ∉ dims || iz !== nothing
        slice = selectdim(slice, 1, iz0)
    end

    return slice
end

function select_slice(variable::AbstractMKArray{T,4}, dims::Symbol...; input=nothing,
                      it=nothing, is=1, ir=nothing, iz=nothing, kwargs...) where T
    # Array is (z,r,species,t)

    if it !== nothing
        it0 = it
    elseif input === nothing || :it0 ∉ input
        it0 = size(variable, 4)
    else
        it0 = input.it0
    end
    if ir !== nothing
        ir0 = ir
    elseif input === nothing || :ir0 ∉ input
        ir0 = max(size(variable, 2) ÷ 3, 1)
    else
        ir0 = input.ir0
    end
    if iz !== nothing
        iz0 = iz
    elseif input === nothing || :iz0 ∉ input
        iz0 = max(size(variable, 1) ÷ 3, 1)
    else
        iz0 = input.iz0
    end

    slice = variable
    if :t ∉ dims || it !== nothing
        slice = selectdim(slice, 4, it0)
    end
    slice = selectdim(slice, 3, is)
    if :r ∉ dims || ir !== nothing
        slice = selectdim(slice, 2, ir0)
    end
    if :z ∉ dims || iz !== nothing
        slice = selectdim(slice, 1, iz0)
    end

    return slice
end

function select_slice(variable::AbstractMKArray{T,5}, dims::Symbol...; input=nothing,
                      it=nothing, is=1, ir=nothing, iz=nothing, ivperp=nothing,
                      ivpa=nothing, kwargs...) where T
    # Array is (vpa,vperp,z,r,t)

    if it !== nothing
        it0 = it
    elseif input === nothing || :it0 ∉ input
        it0 = size(variable, 5)
    else
        it0 = input.it0
    end
    if ir !== nothing
        ir0 = ir
    elseif input === nothing || :ir0 ∉ input
        ir0 = max(size(variable, 4) ÷ 3, 1)
    else
        ir0 = input.ir0
    end
    if iz !== nothing
        iz0 = iz
    elseif input === nothing || :iz0 ∉ input
        iz0 = max(size(variable, 3) ÷ 3, 1)
    else
        iz0 = input.iz0
    end
    if ivperp !== nothing
        ivperp0 = ivperp
    elseif input === nothing || :ivperp0 ∉ input
        ivperp0 = max(size(variable, 2) ÷ 3, 1)
    else
        ivperp0 = input.ivperp0
    end
    if ivpa !== nothing
        ivpa0 = ivpa
    elseif input === nothing || :ivpa0 ∉ input
        ivpa0 = max(size(variable, 1) ÷ 3, 1)
    else
        ivpa0 = input.ivpa0
    end

    slice = variable
    if :t ∉ dims || it !== nothing
        slice = selectdim(slice, 5, it0)
    end
    if :r ∉ dims || ir !== nothing
        slice = selectdim(slice, 4, ir0)
    end
    if :z ∉ dims || iz !== nothing
        slice = selectdim(slice, 3, iz0)
    end
    if :vperp ∉ dims || ivperp !== nothing
        slice = selectdim(slice, 2, ivperp0)
    end
    if :vpa ∉ dims || ivpa !== nothing
        slice = selectdim(slice, 1, ivpa0)
    end

    return slice
end

function select_slice(variable::AbstractMKArray{T,6}, dims::Symbol...; input=nothing,
                      it=nothing, is=1, ir=nothing, iz=nothing, ivperp=nothing,
                      ivpa=nothing, kwargs...) where T
    # Array is (vpa,vperp,z,r,species,t)

    if it !== nothing
        it0 = it
    elseif input === nothing || :it0 ∉ input
        it0 = size(variable, 6)
    else
        it0 = input.it0
    end
    if ir !== nothing
        ir0 = ir
    elseif input === nothing || :ir0 ∉ input
        ir0 = max(size(variable, 4) ÷ 3, 1)
    else
        ir0 = input.ir0
    end
    if iz !== nothing
        iz0 = iz
    elseif input === nothing || :iz0 ∉ input
        iz0 = max(size(variable, 3) ÷ 3, 1)
    else
        iz0 = input.iz0
    end
    if ivperp !== nothing
        ivperp0 = ivperp
    elseif input === nothing || :ivperp0 ∉ input
        ivperp0 = max(size(variable, 2) ÷ 3, 1)
    else
        ivperp0 = input.ivperp0
    end
    if ivpa !== nothing
        ivpa0 = ivpa
    elseif input === nothing || :ivpa0 ∉ input
        ivpa0 = max(size(variable, 1) ÷ 3, 1)
    else
        ivpa0 = input.ivpa0
    end

    slice = variable
    if :t ∉ dims || it !== nothing
        slice = selectdim(slice, 6, it0)
    end
    slice = selectdim(slice, 5, is)
    if :r ∉ dims || ir !== nothing
        slice = selectdim(slice, 4, ir0)
    end
    if :z ∉ dims || iz !== nothing
        slice = selectdim(slice, 3, iz0)
    end
    if :vperp ∉ dims || ivperp !== nothing
        slice = selectdim(slice, 2, ivperp0)
    end
    if :vpa ∉ dims || ivpa !== nothing
        slice = selectdim(slice, 1, ivpa0)
    end

    return slice
end

function select_slice(variable::AbstractMKArray{T,7}, dims::Symbol...; input=nothing,
                      it=nothing, is=1, ir=nothing, iz=nothing, ivzeta=nothing,
                      ivr=nothing, ivz=nothing, kwargs...) where T
    # Array is (vz,vr,vzeta,z,r,species,t)

    if it !== nothing
        it0 = it
    elseif input === nothing || :it0 ∉ input
        it0 = size(variable, 7)
    else
        it0 = input.it0
    end
    if ir !== nothing
        ir0 = ir
    elseif input === nothing || :ir0 ∉ input
        ir0 = max(size(variable, 5) ÷ 3, 1)
    else
        ir0 = input.ir0
    end
    if iz !== nothing
        iz0 = iz
    elseif input === nothing || :iz0 ∉ input
        iz0 = max(size(variable, 4) ÷ 3, 1)
    else
        iz0 = input.iz0
    end
    if ivzeta !== nothing
        ivzeta0 = ivzeta
    elseif input === nothing || :ivzeta0 ∉ input
        ivzeta0 = max(size(variable, 3) ÷ 3, 1)
    else
        ivzeta0 = input.ivzeta0
    end
    if ivr !== nothing
        ivr0 = ivr
    elseif input === nothing || :ivr0 ∉ input
        ivr0 = max(size(variable, 2) ÷ 3, 1)
    else
        ivr0 = input.ivr0
    end
    if ivz !== nothing
        ivz0 = ivz
    elseif input === nothing || :ivz0 ∉ input
        ivz0 = max(size(variable, 1) ÷ 3, 1)
    else
        ivz0 = input.ivz0
    end

    slice = variable
    if :t ∉ dims || it !== nothing
        slice = selectdim(slice, 7, it0)
    end
    slice = selectdim(slice, 6, is)
    if :r ∉ dims || ir !== nothing
        slice = selectdim(slice, 5, ir0)
    end
    if :z ∉ dims || iz !== nothing
        slice = selectdim(slice, 4, iz0)
    end
    if :vzeta ∉ dims || ivzeta !== nothing
        slice = selectdim(slice, 3, ivzeta0)
    end
    if :vr ∉ dims || ivr !== nothing
        slice = selectdim(slice, 2, ivr0)
    end
    if :vz ∉ dims || ivz !== nothing
        slice = selectdim(slice, 1, ivz0)
    end

    return slice
end

"""
    select_time_slice(time::AbstractVector, range)

Variant of `select_slice()` to be used on 'time' arrays, which are always 1D.
"""
function select_time_slice(time::AbstractMKVector, range)
    if range === nothing
        return time
    else
        return @view time[range]
    end
end


"""
get_dimension_slice_indices(keep_dims...; input, it=nothing, is=nothing,
                            ir=nothing, iz=nothing, ivperp=nothing, ivpa=nothing,
                            ivzeta=nothing, ivr=nothing, ivz=nothing)

Get indices for dimensions to slice

The indices are taken from `input`, unless they are passed as keyword arguments

The dimensions in `keep_dims` are not given a slice (those are the dimensions we want in
the variable after slicing).
"""
function get_dimension_slice_indices(keep_dims...; run_info, input, it=nothing,
                                     is=nothing, ir=nothing, iz=nothing, ivperp=nothing,
                                     ivpa=nothing, ivzeta=nothing, ivr=nothing,
                                     ivz=nothing)
    if isa(input, AbstractDict)
        input = Dict_to_NamedTuple(input)
    end
    return (:it=>(it === nothing ? (:t ∈ keep_dims ? nothing : input.it0) : it),
            :is=>(is === nothing ? (:s ∈ keep_dims ? nothing : input.is0) : is),
            :ir=>(ir === nothing ? (:r ∈ keep_dims ? nothing : input.ir0) : ir),
            :iz=>(iz === nothing ? (:z ∈ keep_dims ? nothing : input.iz0) : iz),
            :ivperp=>(ivperp === nothing ? (:vperp ∈ keep_dims ? nothing : input.ivperp0) : ivperp),
            :ivpa=>(ivpa === nothing ? (:vpa ∈ keep_dims ? nothing : input.ivpa0) : ivpa),
            :ivzeta=>(ivzeta === nothing ? (:vzeta ∈ keep_dims ? nothing : input.ivzeta0) : ivzeta),
            :ivr=>(ivr === nothing ? (:vr ∈ keep_dims ? nothing : input.ivr0) : ivr),
            :ivz=>(ivz === nothing ? (:vz ∈ keep_dims ? nothing : input.ivz0) : ivz))
end
