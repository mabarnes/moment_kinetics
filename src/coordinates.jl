"""
"""
module coordinates

export define_coordinate, write_coordinate
export equally_spaced_grid
export set_element_boundaries

using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float, allocate_int
using ..calculus: derivative!
using ..chebyshev: scaled_chebyshev_grid, setup_chebyshev_pseudospectral
using ..quadrature: composite_simpson_weights
using ..input_structs: advection_input

using MPI

"""
structure containing basic information related to coordinates
"""
struct coordinate
    # name is the name of the variable associated with this coordiante
    name::String
    # n_global is the total number of grid points associated with this coordinate
    n_global::mk_int
    # n is the total number of local grid points associated with this coordinate
    n::mk_int
    # ngrid is the number of grid points per element in this coordinate
    ngrid::mk_int
    # nelement is the number of elements associated with this coordinate globally
    nelement_global::mk_int
    # nelement_local is the number of elements associated with this coordinate on this rank
    nelement_local::mk_int
    # nrank is total number of ranks in the calculation of this coord
    nrank::mk_int
    # irank is the rank of this process
    irank::mk_int
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
    # buffer of size 1 for communicating information about cell boundaries
    send_buffer::Array{mk_float,1}
    # buffer of size 1 for communicating information about cell boundaries
    receive_buffer::Array{mk_float,1}
    # the MPI communicator appropriate for this calculation
    comm::MPI.Comm
    # local range to slice from variables to write to output file
    local_io_range::UnitRange{Int64}
    # global range to write into in output file
    global_io_range::UnitRange{Int64}
    # scale for each element
    element_scale::Array{mk_float,1}
    # shift for each element
    element_shift::Array{mk_float,1}
    # option used to set up element spacing
    element_spacing_option::String
end

"""
create arrays associated with a given coordinate,
setup the coordinate grid, and populate the coordinate structure
containing all of this information
"""
function define_coordinate(input, parallel_io::Bool=false)
    # total number of grid points is ngrid for the first element
    # plus ngrid-1 unique points for each additional element due
    # to the repetition of a point at the element boundary
    n_global = (input.ngrid-1)*input.nelement_global + 1
    # local number of points on this process
    n_local = (input.ngrid-1)*input.nelement_local + 1
    # obtain index mapping from full (local) grid to the
    # grid within each element (igrid, ielement)
    igrid, ielement = full_to_elemental_grid_map(input.ngrid,
        input.nelement_local, n_local)
    # obtain (local) index mapping from the grid within each element
    # to the full grid
    imin, imax = elemental_to_full_grid_map(input.ngrid, input.nelement_local)
    # initialise the data used to construct the grid
    # boundaries for each element
    element_boundaries = set_element_boundaries(input.nelement_global, input.L, input.element_spacing_option)
    # shift and scale factors for each local element
    element_scale, element_shift = set_element_scale_and_shift(input.nelement_global, input.nelement_local, input.irank, element_boundaries)
    # initialize the grid and the integration weights associated with the grid
    # also obtain the Chebyshev theta grid and spacing if chosen as discretization option
    grid, wgts, uniform_grid = init_grid(input.ngrid, input.nelement_local, n_global, n_local, input.irank, input.L, element_scale, element_shift,
        imin, imax, igrid, input.discretization, input.name)
    # calculate the widths of the cells between neighboring grid points
    cell_width = grid_spacing(grid, n_local)
    # duniform_dgrid is the local derivative of the uniform grid with respect to
    # the coordinate grid
    duniform_dgrid = allocate_float(input.ngrid, input.nelement_local)
    # scratch is an array used for intermediate calculations requiring n entries
    scratch = allocate_float(n_local)
    # scratch_2d is an array used for intermediate calculations requiring ngrid x nelement entries
    scratch_2d = allocate_float(input.ngrid, input.nelement_local)
    # struct containing the advection speed options/inputs for this coordinate
    advection = input.advection
    # buffers for cyclic communication of boundary points
    # each chain of elements has only two external (off-rank)
    # endpoints, so only two pieces of information must be shared
    send_buffer = allocate_float(1)
    receive_buffer = allocate_float(1)
    # Add some ranges to support parallel file io
    if !parallel_io
        # No parallel io, just write everything
        local_io_range = 1:n_local
        global_io_range = 1:n_local
    elseif input.irank == input.nrank-1
        # Include endpoint on final block
        local_io_range = 1:n_local
        global_io_range = input.irank*(n_local-1)+1:n_global
    else
        # Skip final point, because it is shared with the next block
        # Choose to skip final point in each block so all blocks (except the final one)
        # write a 'chunk' of the same size to the output file. This makes it simple to
        # align HDF5 'chunks' with the data being written
        local_io_range = 1 : n_local-1
        global_io_range = input.irank*(n_local-1)+1 : (input.irank+1)*(n_local-1)
    end
    coord = coordinate(input.name, n_global, n_local, input.ngrid,
        input.nelement_global, input.nelement_local, input.nrank, input.irank, input.L, grid,
        cell_width, igrid, ielement, imin, imax, input.discretization, input.fd_option,
        input.bc, wgts, uniform_grid, duniform_dgrid, scratch, copy(scratch), copy(scratch),
        scratch_2d, copy(scratch_2d), advection, send_buffer, receive_buffer, input.comm,
        local_io_range, global_io_range, element_scale, element_shift, input.element_spacing_option)

    if input.discretization == "chebyshev_pseudospectral" && coord.n > 1
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

function set_element_boundaries(nelement_global, L, element_spacing_option)
    # set global element boundaries
    element_boundaries = allocate_float(nelement_global+1)
    if element_spacing_option == "sqrt" && nelement_global > 3
        # number of boundaries of sqrt grid
        nsqrt = floor(mk_int,(nelement_global)/2) + 1
        if nelement_global%2 > 0 # odd
            if nsqrt < 3
                fac = 2.0/3.0
            else
                fac = 1.0/( 3.0/2.0 - 0.5*((nsqrt-2)/(nsqrt-1))^2)
            end
        else
            fac = 1.0
        end
        
        for j in 1:nsqrt
            element_boundaries[j] = -(L/2.0) + fac*(L/2.0)*((j-1)/(nsqrt-1))^2
        end
        for j in 1:nsqrt
            element_boundaries[(nelement_global+1)+ 1 - j] = (L/2.0) - fac*(L/2.0)*((j-1)/(nsqrt-1))^2
        end
        
    elseif element_spacing_option == "uniform" || (element_spacing_option == "sqrt" && nelement_global < 4) # uniform spacing 
        for j in 1:nelement_global+1
            element_boundaries[j] = L*((j-1)/(nelement_global) - 0.5)
        end
    else 
        println("ERROR: element_spacing_option: ",element_spacing_option, " not supported")
    end
    return element_boundaries
end

function set_element_scale_and_shift(nelement_global, nelement_local, irank, element_boundaries)
    element_scale = allocate_float(nelement_local)
    element_shift = allocate_float(nelement_local)
    
    for j in 1:nelement_local
        iel_global = j + irank*nelement_local
        upper_boundary = element_boundaries[iel_global+1]
        lower_boundary = element_boundaries[iel_global]
        element_scale[j] = 0.5*(upper_boundary-lower_boundary)
        element_shift[j] = 0.5*(upper_boundary+lower_boundary)
    end
    return element_scale, element_shift
end
"""
setup a grid with n_global grid points on the interval [-L/2,L/2]
"""
function init_grid(ngrid, nelement_local, n_global, n_local, irank, L, element_scale, element_shift,
                   imin, imax, igrid, discretization, name)
    uniform_grid = equally_spaced_grid(n_global, n_local, irank, L)
    uniform_grid_shifted = equally_spaced_grid_shifted(n_global, n_local, irank, L)
    if n_global == 1
        grid = allocate_float(n_local)
        grid[1] = 0.0
        wgts = allocate_float(n_local)
        if name == "vr" || name == "vzeta"
            wgts[1] = sqrt(pi) # to cancel factor of 1/sqrt{pi} in integrate_over_neutral_vspace, velocity_moments.jl
                               # in the case that the code runs in 1V mode
        else
            wgts[1] = 1.0
        end
    elseif discretization == "chebyshev_pseudospectral"
        if name == "vperp"
            # initialize chebyshev grid defined on [-L/2,L/2]
            grid, wgts = scaled_chebyshev_grid(ngrid, nelement_local, n_local, element_scale, element_shift, imin, imax)
            grid .= grid .+ L/2.0 # shift to [0,L] appropriate to vperp variable
            wgts = 2.0 .* wgts .* grid # to include 2 vperp in jacobian of integral
                                        # see note above on normalisation
        else
            # initialize chebyshev grid defined on [-L/2,L/2]
            # with n grid points chosen to facilitate
            # the fast Chebyshev transform (aka the discrete cosine transform)
            # needed to obtain Chebyshev spectral coefficients
            # 'wgts' are the integration weights attached to each grid points
            # that are those associated with Clenshaw-Curtis quadrature
            grid, wgts = scaled_chebyshev_grid(ngrid, nelement_local, n_local, element_scale, element_shift, imin, imax)
        end
    elseif discretization == "finite_difference"
        if name == "vperp"
            # initialize equally spaced grid defined on [0,L]
            grid = uniform_grid_shifted
            # use composite Simpson's rule to obtain integration weights associated with this coordinate
            wgts = composite_simpson_weights(grid)
            wgts = 2.0 .* wgts .* grid # to include 2 vperp in jacobian of integral
                                     # assumes pdf normalised like 
                                     # f^N = Pi^{3/2} c_s^3 f / n_ref         
        else #default case 
            # initialize equally spaced grid defined on [-L/2,L/2]
            grid = uniform_grid
            # use composite Simpson's rule to obtain integration weights associated with this coordinate
            wgts = composite_simpson_weights(grid)
        end    
    else
        error("discretization option '$discretization' unrecognized")
    end
    # return the locations of the grid points
    return grid, wgts, uniform_grid
end

"""
setup an equally spaced grid with n_global grid points
between [-L/2,L/2]
"""
function equally_spaced_grid(n_global, n_local, irank, L)
    # create array for the equally spaced grid with n_local grid points
    grid = allocate_float(n_local)
    istart = (n_local - 1)*irank + 1
    grid_spacing = L / (n_global - 1)
    coord_start = -0.5*L + (istart-1)*grid_spacing
    @inbounds for i ∈ 1:n_local
        grid[i] =  coord_start + (i-1)*grid_spacing
    end
    return grid
end

"""
setup an equally spaced grid with n_global grid points
between [0,L]
"""
function equally_spaced_grid_shifted(n_global, n_local, irank, L)
    # create array for the equally spaced grid with n_local grid points
    grid = allocate_float(n_local)
    istart = (n_local - 1)*irank + 1
    grid_spacing = L / (n_global - 1)
    coord_start = (istart-1)*grid_spacing
    @inbounds for i ∈ 1:n_local
        grid[i] =  coord_start + (i-1)*grid_spacing
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
