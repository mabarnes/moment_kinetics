"""
Interpolation routines intended for post-processing.

Note these are not guaranteed to be highly optimized!
"""
module interpolation

export interpolate_to_grid_z

using ..array_allocation: allocate_float
using ..moment_kinetics_structs: null_spatial_dimension_info, null_velocity_dimension_info
using ..type_definitions: mk_float, mk_int

"""
    single_element_interpolate!(result, newgrid, f, imin, imax, coord, spectral)

Interpolation within a single element.

`f` is an array with the values of the input variable in the element to be interpolated.
`imin` and `imax` give the start and end points of the element in the grid (used to
calculate shift and scale factors to a normalised grid).

`newgrid` gives the points within the element where output is required. `result` is filled
with the interpolated values at those points.

`coord` is the `coordinate` struct for the dimension along which interpolation is being
done. `spectral` is the corresponding `discretization_info`.
"""
function single_element_interpolate! end

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

function interpolate_to_grid_1d!(result, newgrid, f, coord, spectral)
    # define local variable nelement for convenience
    nelement = coord.nelement_local

    n_new = size(newgrid)[1]
    # Find which points belong to which element.
    # kstart[j] contains the index of the first point in newgrid that is within element
    # j, and kstart[nelement+1] is n_new if the last point is within coord.grid, or the
    # index of the first element outside coord.grid otherwise.
    # Assumes points in newgrid are sorted.
    # May not be the most efficient algorithm.
    # Find start/end points for each element, storing their indices in kstart
    kstart = Vector{mk_int}(undef, nelement+1)

    # First element includes both boundary points, while all others have only one (to
    # avoid duplication), so calculate the first element outside the loop.
    if coord.radau_first_element && coord.irank == 0
        first_element_spectral = spectral.radau
        # For a grid with a Radau first element, the lower boundary of the first element
        # is at coord=0, and the coordinate range is 0<coord<∞ so we will never need to to
        # extrapolate to negative values, and points between coord=0 and coord.grid[1] are
        # really within the first element, so the index for the first point greater than 0
        # (the left boundary of the coordinate grid) in `newgrid` is always 1.
        kstart[1] = 1

        # In a coordinate with a Radau first element, no point should be less than zero.
        # If bounds checking is enabled, check that first `newgrid` point is ≥0.
        @boundscheck newgrid[1] ≥ 0.0
    else
        first_element_spectral = spectral.lobatto
        # set the starting index by finding the start of coord.grid
        kstart[1] = searchsortedfirst(newgrid, coord.grid[1])
    end

    # check to see if any of the newgrid points are to the left of the first grid point
    for j ∈ 1:kstart[1]-1
        # if the new grid location is outside the bounds of the original grid,
        # extrapolate f with Gaussian-like decay beyond the domain
        result[j] = f[1] * exp(-(coord.grid[1] - newgrid[j])^2)
    end
    @inbounds for j ∈ 1:nelement
        # Search from kstart[j] to try to speed up the sort, but means result of
        # searchsortedfirst() is offset by kstart[j]-1 from the beginning of newgrid.
        kstart[j+1] = kstart[j] - 1 + @views searchsortedfirst(newgrid[kstart[j]:end], coord.grid[coord.imax[j]])
    end

    if kstart[1] < kstart[2]
        imin = coord.imin[1]
        imax = coord.imax[1]
        kmin = kstart[1]
        kmax = kstart[2] - 1
        @views single_element_interpolate!(result[kmin:kmax], newgrid[kmin:kmax],
                                           f[imin:imax], imin, imax, 1, coord,
                                           first_element_spectral)
    end
    @inbounds for j ∈ 2:nelement
        kmin = kstart[j]
        kmax = kstart[j+1] - 1
        if kmin <= kmax
            imin = coord.imin[j] - 1
            imax = coord.imax[j]
            @views single_element_interpolate!(result[kmin:kmax], newgrid[kmin:kmax],
                                               f[imin:imax], imin, imax, j, coord,
                                               spectral.lobatto)
        end
    end

    for k ∈ kstart[nelement+1]:n_new
        result[k] = f[end] * exp(-(newgrid[k] - coord.grid[end])^2)
    end

    return nothing
end

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
