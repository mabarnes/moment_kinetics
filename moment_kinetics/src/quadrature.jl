"""
"""
module quadrature

export composite_simpson_weights

using ..array_allocation: allocate_float

"""
trapezium_weights creates, computes, and returns an array for the 1D integration weights
associated with each grid point using the trapezium rule.
"""
function trapezium_weights(grid)
    n = length(grid)
    wgts = allocate_float(n)
    # assume equal grid spacing
    h = grid[2]-grid[1]

    wgts[1] = h / 2.0
    wgts[2:end-1] .= h
    wgts[end] = h / 2.0

    return wgts
end

"""
composite_simpson_weights creates, computes, and returns an array for the
1D integration weights associated with each grid point using composite Simpson's rule
"""
function composite_simpson_weights(grid)
    n = length(grid)
    wgts = allocate_float(n)
    # assume equal grid spacing
    h = grid[2]-grid[1]
    # constant coefficients needed for composite Simpson's rule
    c1 = h/3.0
    c2 = 4.0*h/3.0
    c3 = 2.0*h/3.0
    # composite Simpson's rule requires n odd;
    # if n even, use Simpson's 3/8 rule for first segment.
    # this uses the first 4 points, with the 4th point being
    # re-used for the composite Simpson's rule for the remaining points.
    if mod(n,2) == 0
        wgts[1] = 3.0*h/8.0
        wgts[2] = 9.0*h/8.0
        wgts[3] = wgts[2]
        wgts[4] = wgts[1]
        if n > 4
            wgts[n] = c1
        end
    else
        wgts[1] = c1
        wgts[2] = c2
        # if n = 3, use Simpson's rule
        if n == 3
            wgts[3] = c1
        # otherwise, will be using composite Simpson's rule
        else
            wgts[3] = c3
            wgts[4] = c2
            wgts[n] = c1
        end
    end
    if n > 5
        if mod(n,2) == 0
            wgts[4] += c1
            wgts[5:2:n-1] .= c2
            wgts[6:2:n-1] .= c3
        else
            wgts[5:2:n-1] .= c3
            wgts[6:2:n-1] .= c2
        end
    end
    return wgts
end

end
