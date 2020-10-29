module coordinates

import array_allocation: allocate_float, allocate_int
import file_io: open_output_file
import chebyshev: scaled_chebyshev_grid

export define_coordinate, write_coordinate

# structure containing basic information related to coordinates
struct coordinate
    # name is the name of the variable associated with this coordiante
    name::String
    # n is the total number of grid points associated with this coordinate
    n::Int64
    # ngrid is the number of grid points per element in this coordinate
    ngrid::Int64
    # nelement is the number of elements associated with this coordinate
    nelement::Int64
    # L is the box length in this coordinate
    L::Float64
    # grid is the location of the grid points
    grid::Array{Float64,1}
    # cell_width is the width associated with the cells between grid points
    cell_width::Array{Float64,1}
    # igrid contains the grid point index within the element
    igrid::Array{Int64,1}
    # ielement contains the element index
    ielement::Array{Int64,1}
    # imin[j] contains the minimum index on the full grid for element j
    imin::Array{Int64,1}
    # imax[j] contains the maximum index on the full grid for element j
    imax::Array{Int64,1}
    # discretization option for the grid
    discretization::String
    # bc is the boundary condition option for this coordinate
    bc::String
    # wgts contains the integration weights associated with each grid point
    wgts::Array{Float64,1}
end
# create arrays associated with a given coordinate,
# setup the coordinate grid, and populate the coordinate structure
# containing all of this information
#function define_coordinate(ngrid, nelement, L, discretization, bc)
function define_coordinate(input)
    # total number of grid points is ngrid for the first element
    # plus ngrid-1 unique points for each additional element due
    # to the repetition of a point at the element boundary
    n = (input.ngrid-1)*input.nelement + 1
    # obtain index mapping from full grid to the
    # grid within each element (igrid, ielement)
    igrid, ielement = full_to_elemental_grid_map(input.ngrid, input.nelement, n)
    # obtain index mapping from the grid within each element
    # to the full grid
    imin, imax = elemental_to_full_grid_map(input.ngrid, input.nelement)
    # initialize the grid and the integration weights associated with the grid
    grid, wgts = init_grid(input.ngrid, input.nelement, n, input.L, imin, imax,
        igrid, input.discretization)
    # calculate the widths of the cells between neighboring grid points
    cell_width = grid_spacing(grid, n)

    return coordinate(input.name, n, input.ngrid, input.nelement, input.L, grid,
        cell_width, igrid, ielement, imin, imax, input.discretization, input.bc, wgts)
end
# setup a grid with n grid points on the interval [-L/2,L/2]
function init_grid(ngrid, nelement, n, L, imin, imax, igrid, discretization)
    if discretization == "chebyshev_pseudospectral"
        # initialize chebyshev grid defined on [-L/2,L/2]
        # with n grid points chosen to facilitate
        # the fast Chebyshev transform (aka the discrete cosine transform)
        # needed to obtain Chebyshev spectral coefficients
        # 'wgts' are the integration weights attached to each grid points
        # that are those associated with Clenshaw-Curtis quadrature
        grid, wgts = scaled_chebyshev_grid(ngrid, nelement, n, L, imin, imax)
#    println("sum(wgts): ", sum(wgts))
    elseif discretization == "finite_difference"
        # initialize equally spaced grid defined on [-L/2,L/2]
        grid = equally_spaced_grid(n, L)
        # allocate arrays for the integration weights
        # NB: should be able to save memory as weights are repeated for each element
        wgts = allocate_float(n)
        # the integration weights are the (equal) grid spacings
        tmp = 1/(n-1)
        wgts .= tmp
    end
    # return the locations of the grid points
    return grid, wgts
end
# setup an equally spaced grid with n grid points
# between [-L/2,L/2]
function equally_spaced_grid(n, L)
    # create array for the equally spaced grid with n grid points
    grid = allocate_float(n)
    @inbounds for i ∈ 1:n
        grid[i] = -0.5*L + (i-1)*L/(n-1)
    end
    return grid
end
# given a set of grid point locations
# calculate and return the length
# associated with the cell between adjacent grid points
function grid_spacing(grid, n)
    # array to contain the cell widths
    d = allocate_float(n)
    @inbounds begin
        for i ∈ 2:n
            d[i-1] =  grid[i]-grid[i-1]
        end
        # final (nth) entry corresponds to cell beyond the grid boundary
        # only time this may be needed is if periodic BCs are used
        d[n] = d[1]
    end
    return d
end
# setup arrays containing a map from the unpacked grid point indices
# to the element index and the grid point index within each element
function full_to_elemental_grid_map(ngrid, nelement, n)
    igrid = allocate_int(n)
    ielement = allocate_int(n)
    k = 1
    for i ∈ 1:ngrid
        ielement[k] = 1
        igrid[k] = i
        k += 1
    end
    if nelement > 1
        for j ∈ 2:nelement
            # avoid double-counting overlapping point
            # at boundary between elements
            for i ∈ 2:ngrid
                ielement[k] = j
                igrid[k] = i
                k += 1
            end
        end
    end
    return igrid, ielement
end
# returns imin and imax, which contain the minimum and maximum
# indices on the full grid for each element
function elemental_to_full_grid_map(ngrid, nelement)
    imin = allocate_int(nelement)
    imax = allocate_int(nelement)
    @inbounds begin
        # the first element contains ngrid entries
        imin[1] = 1
        imax[1] = ngrid
        # each additional element contributes ngrid-1 unique entries
        # due to repetition of one grid point at the boundary
        if nelement > 1
            for i ∈ 2:nelement
                imin[i] = imax[i-1] + 1
                imax[i] = imin[i] + ngrid - 2
            end
        end
    end
    return imin, imax
end
# integration_weights creates, computes, and returns an array for the
# 1D integration weights associated with each grid point
function integration_weights(grid, discretization)

end
# write the grid point locations for this coordinate to file
function write_coordinate(coord, run_name, extension)
    # open a file with the desired extension
    io = open_output_file(run_name, extension)
    # write grid point locations to file
    @inbounds for i ∈ 1:coord.n
        println(io, "index: ", i, "  grid: ", coord.grid[i], "  weight: ", coord.wgts[i])
    end
    # close the file
    close(io)
end

end
