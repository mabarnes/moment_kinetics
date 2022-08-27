"""
"""
module coordinates

export define_coordinate, write_coordinate
export equally_spaced_grid

using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float, allocate_int
using ..calculus: derivative!
using ..file_io: open_output_file
using ..chebyshev: scaled_chebyshev_grid, setup_chebyshev_pseudospectral
using ..quadrature: composite_simpson_weights
using ..input_structs: advection_input

"""
structure containing basic information related to coordinates
"""
struct coordinate
    # name is the name of the variable associated with this coordiante
    name::String
    # n is the total number of grid points associated with this coordinate
    n::mk_int
    # ngrid is the number of grid points per element in this coordinate
    ngrid::mk_int
    # nelement is the number of elements associated with this coordinate
    nelement::mk_int
    # L is the box length in this coordinate
    L::mk_float
    # grid is the location of the grid points
    grid::Array{mk_float,1}
    # cell_width is the width associated with the cells between grid points
    cell_width::Array{mk_float,1}
    # igrid contains the grid point index within the element
    igrid::Array{mk_int,1}
    # ielement contains the element index
    ielement::Array{mk_int,1}
    # imin[j] contains the minimum index on the full grid for element j
    imin::Array{mk_int,1}
    # imax[j] contains the maximum index on the full grid for element j
    imax::Array{mk_int,1}
    # discretization option for the grid
    discretization::String
    # if the discretization is finite differences, fd_option provides the precise scheme
    fd_option::String
    # bc is the boundary condition option for this coordinate
    bc::String
    # wgts contains the integration weights associated with each grid point
    wgts::Array{mk_float,1}
    # uniform_grid contains location of grid points mapped to a uniform grid
    # if finite differences used for discretization, no mapping required, and uniform_grid = grid
    uniform_grid::Array{mk_float,1}
    # duniform_dgrid is the local derivative of the uniform grid with respect to
    # the coordinate grid
    duniform_dgrid::Array{mk_float,2}
    # scratch is an array used for intermediate calculations requiring n entries
    scratch::Array{mk_float,1}
    # scratch2 is an array used for intermediate calculations requiring n entries
    scratch2::Array{mk_float,1}
    # scratch3 is an array used for intermediate calculations requiring n entries
    scratch3::Array{mk_float,1}
    # scratch_2d and scratch2_2d are arrays used for intermediate calculations requiring
    # ngrid x nelement entries
    scratch_2d::Array{mk_float,2}
    scratch2_2d::Array{mk_float,2}
    # struct containing advection speed options/inputs
    advection::advection_input
end

"""
create arrays associated with a given coordinate,
setup the coordinate grid, and populate the coordinate structure
containing all of this information
"""
function define_coordinate(input, composition=nothing)
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
    # also obtain the Chebyshev theta grid and spacing if chosen as discretization option
    grid, wgts, uniform_grid = init_grid(input.ngrid, input.nelement, n, input.L,
        imin, imax, igrid, input.discretization)
    # calculate the widths of the cells between neighboring grid points
    cell_width = grid_spacing(grid, n)
    # duniform_dgrid is the local derivative of the uniform grid with respect to
    # the coordinate grid
    duniform_dgrid = allocate_float(input.ngrid, input.nelement)
    # scratch is an array used for intermediate calculations requiring n entries
    scratch = allocate_float(n)
    # scratch_2d is an array used for intermediate calculations requiring ngrid x nelement entries
    scratch_2d = allocate_float(input.ngrid, input.nelement)
    # struct containing the advection speed options/inputs for this coordinate
    advection = input.advection

    coord = coordinate(input.name, n, input.ngrid, input.nelement, input.L, grid,
        cell_width, igrid, ielement, imin, imax, input.discretization, input.fd_option,
        input.bc, wgts, uniform_grid, duniform_dgrid, scratch, copy(scratch),
        copy(scratch), scratch_2d, copy(scratch_2d), advection)

    if input.discretization == "chebyshev_pseudospectral"
        # create arrays needed for explicit Chebyshev pseudospectral treatment in this
        # coordinate and create the plans for the forward and backward fast Chebyshev
        # transforms
        spectral = setup_chebyshev_pseudospectral(coord)
        # obtain the local derivatives of the uniform grid with respect to the used grid
        derivative!(coord.duniform_dgrid, coord.uniform_grid, coord, spectral)
    else
        # create dummy Bool variable to return in place of the above struct
        spectral = false
        coord.duniform_dgrid .= 1.0
    end

    return coord, spectral
end

"""
setup a grid with n grid points on the interval [-L/2,L/2]
"""
function init_grid(ngrid, nelement, n, L, imin, imax, igrid, discretization)
    uniform_grid = equally_spaced_grid(n,L)
    if n == 1
        grid = allocate_float(n)
        grid[1] = 0
        wgts = allocate_float(n)
        wgts[1] = 1.0
    elseif discretization == "chebyshev_pseudospectral"
        # initialize chebyshev grid defined on [-L/2,L/2]
        # with n grid points chosen to facilitate
        # the fast Chebyshev transform (aka the discrete cosine transform)
        # needed to obtain Chebyshev spectral coefficients
        # 'wgts' are the integration weights attached to each grid points
        # that are those associated with Clenshaw-Curtis quadrature
        grid, wgts = scaled_chebyshev_grid(ngrid, nelement, n, L, imin, imax)
    elseif discretization == "finite_difference"
        # initialize equally spaced grid defined on [-L/2,L/2]
        grid = uniform_grid
        # use composite Simpson's rule to obtain integration weights associated with this coordinate
        wgts = composite_simpson_weights(grid)
    else
        error("discretization option '$discretization' unrecognized")
    end
    # return the locations of the grid points
    return grid, wgts, uniform_grid
end

"""
setup an equally spaced grid with n grid points
between [-L/2,L/2]
"""
function equally_spaced_grid(n, L)
    # create array for the equally spaced grid with n grid points
    grid = allocate_float(n)
    @inbounds for i ∈ 1:n
        grid[i] = -0.5*L + (i-1)*L/(n-1)
    end
    return grid
end

"""
given a set of grid point locations
calculate and return the length
associated with the cell between adjacent grid points
"""
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

"""
setup arrays containing a map from the unpacked grid point indices
to the element index and the grid point index within each element
"""
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

"""
returns imin and imax, which contain the minimum and maximum
indices on the full grid for each element
"""
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

end
