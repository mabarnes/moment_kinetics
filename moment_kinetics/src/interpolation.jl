"""
Interpolation routines intended for post-processing.

Note these are not guaranteed to be highly optimized!
"""
module interpolation

export interpolate_to_grid_z

using ..array_allocation: allocate_float
using ..moment_kinetics_structs: null_spatial_dimension_info, null_velocity_dimension_info
using ..type_definitions: mk_float

"""
Interpolation from a regular grid to a 1d grid with arbitrary spacing

Arguments
---------
result : Array{mk_float, 1}
    Array to be overwritten with the result of the interpolation
new_grid : Array{mk_float, 1}
    Grid of points to interpolate `coord` to
f : Array{mk_float}
    Field to be interpolated
coord : coordinate
    `coordinate` struct giving the coordinate along which f varies
spectral : discretization_info
    struct containing information for discretization, whose type determines which method
    is used.
"""
function interpolate_to_grid_1d! end

function interpolate_to_grid_1d!(result, new_grid, f, coord,
                                 spectral::null_spatial_dimension_info)
    # There is only one point in the 'old grid' represented by coord (as indicated by the
    # type of the `spectral` argument), and we are interpolating in a spatial dimension.
    # Assume that the variable should be taken to be constant in this dimension to
    # 'interpolate'.
    result .= f[1]

    return nothing
end

function interpolate_to_grid_1d!(result, new_grid, f, coord,
                                 spectral::null_velocity_dimension_info)
    # There is only one point in the 'old grid' represented by coord (as indicated by the
    # type of the `spectral` argument), and we are interpolating in a velocity space
    # dimension. Assume that the profile 'should be' a Maxwellian over the new grid, with
    # a width of 1 in units of the reference speed.
    @. result = f[1] * exp(-new_grid^2)

    return nothing
end

"""
Interpolation from a regular grid to a 1d grid with arbitrary spacing

This version allocates a new array for the result, which is returned.

Arguments
---------
new_grid : Array{mk_float, 1}
    Grid of points to interpolate `coord` to
f : Array{mk_float}
    Field to be interpolated
coord : coordinate
    `coordinate` struct giving the coordinate along which f varies
spectral : Bool or chebyshev_info
    struct containing information for discretization, whose type determines which method
    is used.

Returns
-------
result : Array
    Array with the values of `f` interpolated to the points in `new_grid`.
"""
function interpolate_to_grid_1d(newgrid, args...)
    # Array for output
    result = similar(newgrid)

    interpolate_to_grid_1d!(result, newgrid, args...)

    return result
end

"""
"""
function interpolate_to_grid_z!(result::Array{mk_float, 3}, newgrid, f::Array{mk_float, 3}, z, spectral)
    size_f = size(f)
    for is ∈ 1:size_f[3]
        for ivpa ∈ 1:size_f[1]
            @views interpolate_to_grid_1d!(result[ivpa, :, is], newgrid, f[ivpa, :, is], z, spectral)
        end
    end

    return nothing
end

"""
"""
function interpolate_to_grid_z(newgrid, f::Array{mk_float, 3}, z, spectral)
    size_f = size(f)
    result = allocate_float(size_f[1], size(newgrid)[1], size_f[3])

    interpolate_to_grid_z!(result, newgrid, f, z, spectral)

    return result
end

"""
"""
function interpolate_to_grid_z!(result::Array{mk_float, 2}, newgrid, f::Array{mk_float, 2}, z, spectral)
    size_f = size(f)
    for is ∈ 1:size_f[2]
        @views interpolate_to_grid_1d!(result[:, is], newgrid, f[:, is], z, spectral)
    end

    return nothing
end

"""
"""
function interpolate_to_grid_z(newgrid, f::Array{mk_float, 2}, z, spectral)
    size_f = size(f)
    result = allocate_float(size(newgrid)[1], size_f[2])

    interpolate_to_grid_z!(result, newgrid, f, z, spectral)

    return result
end

"""
"""
function interpolate_to_grid_z!(result::Array{mk_float, 1}, newgrid, f::Array{mk_float, 1}, z, spectral)
    interpolate_to_grid_1d!(result, newgrid, f, z, spectral)

    return nothing
end

"""
"""
function interpolate_to_grid_z(newgrid, f::Array{mk_float, 1}, z, spectral)
    return interpolate_to_grid_1d(newgrid, f, z, spectral)
end

"""
"""
function interpolate_to_grid_vpa!(result::Array{mk_float, 3}, newgrid, f::Array{mk_float, 3}, vpa, spectral)
    size_f = size(f)
    for is ∈ 1:size_f[3]
        for iz ∈ 1:size_f[2]
            @views interpolate_to_grid_1d!(result[:, iz, is], newgrid, f[:, iz, is], vpa, spectral)
        end
    end

    return nothing
end

"""
"""
function interpolate_to_grid_vpa(newgrid, f::Array{mk_float, 3}, vpa, spectral)
    size_f = size(f)
    result = allocate_float(size(newgrid)[1], size_f[2:3]...)

    interpolate_to_grid_vpa!(result, newgrid, f, vpa, spectral)

    return result
end

"""
"""
function interpolate_to_grid_vpa!(result::AbstractVector{mk_float}, newgrid,
                                  f::AbstractVector{mk_float}, vpa, spectral)

    interpolate_to_grid_1d!(result, newgrid, f, vpa, spectral)

    return nothing
end

"""
"""
function interpolate_to_grid_vpa(newgrid, f::AbstractVector{mk_float}, vpa, spectral)

    return interpolate_to_grid_1d(newgrid, f, vpa, spectral)
end

end
