module hermite_spline_interpolation

using ..calculus: derivative!
using ..finite_differences: finite_difference_info
import ..interpolation: interpolate_to_grid_1d!

"""
Interpolation from a regular grid to a 1d grid with arbitrary spacing

This version of interpolate_to_grid_1d!() is used when moment_kinetics is in finite
difference mode.

Arguments
---------
new_grid : Array{mk_float, 1}
    Grid of points to interpolate `coord` to
f : Array{mk_float}
    Field to be interpolated
coord : coordinate
    `coordinate` struct giving the coordinate along which f varies
not_spectral : finite_difference_info
    A finite_difference_info argument here indicates that the coordinate is not
    spectral-element discretized, i.e. it is on a uniform ('finite difference') grid.
derivative : Val(n)
    The value of `n` the integer in the `Val{n}` indicates the order of the derivative to
    be calculated of the interpolating function (only a few values of `n` are supported).
    Defaults to Val(0), which means just calculating the interpolating function itself.
"""
function interpolate_to_grid_1d! end

function interpolate_to_grid_1d!(result, new_grid, f, coord,
                                 not_spectral::finite_difference_info,
                                 derivative::Val{0}=Val(0), args...)
    x = coord.grid
    n_new = length(new_grid)

    # Store dfdx in scratch array
    dfdx = coord.scratch3
    derivative!(dfdx, f, coord, not_spectral)

    new_grid_ind = 1

    # First handle any points that are below the range of x
    while true
        if new_grid[new_grid_ind] > x[1]
            break
        end
        result[new_grid_ind] = f[1] * exp(-(x[1] - new_grid[new_grid_ind])^2)
        new_grid_ind += 1
    end

    # Hermite spline cubic interpolation within the range of x
    # From https://en.wikipedia.org/wiki/Cubic_Hermite_spline
    # p(x) = h00(t)*pk + h10(t)*(xk1 - xk)*mk + h01(t)*pk1 + h11(t)*(xk1 - xk)*mk1
    # where:
    # * xk, xk1 are the values of x at the beginning and end of the interval (at indices
    #   k and k+1)
    # * t = (x - xk)/(xk1 - xk)
    # * pk = p(xk), pk1 = p(xk1)
    # * mk = dp/dx(xk), mk1 = dp/dx(xk1)
    # * h00(t) = (1 + 2*t)(1-t)^2
    # * h10(t) = t(1-t)^2
    # * h01(t) = t^2(3-2t)
    # * h11(t) = t^2(t-1)
    # To simplify things slightly, define
    # nk = (xk1 - xk)*mk
    # nk1 = (xk1 - xk)*mk1
    x_ind = 1
    xk = 0.0
    xk1 = 0.0
    Dx = 1.0
    pk = 0.0
    pk1 = 0.0
    nk = 0.0
    nk1 = 0.0
    while new_grid_ind <= n_new && new_grid[new_grid_ind] <= x[end]
        if new_grid[new_grid_ind] > x[x_ind+1]
            # Current point is beyond the current interval of x, so increment x_ind
            x_ind += 1
            while new_grid[new_grid_ind] > x[x_ind+1]
                x_ind += 1
            end
            xk = x[x_ind]
            xk1 = x[x_ind+1]
            Dx = xk1 - xk
            pk = f[x_ind]
            pk1 = f[x_ind+1]
            nk = dfdx[x_ind] * Dx
            nk1 = dfdx[x_ind+1] * Dx
        end
        t = (new_grid[new_grid_ind] - xk) / Dx
        oneminust = 1.0 - t
        oneminustsquared = oneminust * oneminust
        tsquared = t*t
        h00 = (1.0 + 2.0 * t) * oneminustsquared
        h10 = t * oneminustsquared
        h01 = tsquared * (3.0 - 2.0 * t)
        h11 = - tsquared * oneminust
        result[new_grid_ind] = h00*pk + h10*nk + h01*pk1 + h11*nk1
        new_grid_ind += 1
    end

    # Finally handle any points that are above the range of x
    for k ∈ new_grid_ind:length(new_grid)
        result[k] = f[end] * exp(-(new_grid[k] - x[end])^2)
    end

    return nothing
end

function interpolate_to_grid_1d!(result, new_grid, f, coord,
                                 not_spectral::finite_difference_info,
                                 derivative::Val{1}, args...)
    error("First derivative interpolation not implemented for finite-difference yet.")
end

end # hermite_spline_interpolation
