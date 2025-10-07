"""
"""
module coordinates

export coordinate
export define_coordinate, write_coordinate
export equally_spaced_grid
export set_element_boundaries

using LinearAlgebra
using ..type_definitions: mk_float, mk_int, OptionsDict
using ..array_allocation: allocate_float, allocate_shared_float, allocate_int, allocate_shared_int
using ..calculus: derivative!
using ..chebyshev: scaled_chebyshev_grid, scaled_chebyshev_radau_grid, setup_chebyshev_pseudospectral
using ..communication
using ..finite_differences: finite_difference_info
using ..fourier: setup_fourier_pseudospectral
using ..gauss_legendre: scaled_gauss_legendre_lobatto_grid, scaled_gauss_legendre_radau_grid, setup_gausslegendre_pseudospectral
using ..input_structs
using ..quadrature: trapezium_weights, composite_simpson_weights
using ..moment_kinetics_structs: coordinate, null_spatial_dimension_info,
                                 null_velocity_dimension_info, null_vperp_dimension_info

using MPI
using OrderedCollections: OrderedDict

"""
    get_coordinate_input(input_dict, name; ignore_MPI=false,
                         warn_unexpected::Bool=false)

Read the input for coordinate `name` from `input_dict`, setting defaults, etc.
"""
function get_coordinate_input(input_dict, name; ignore_MPI=false,
                              warn_unexpected::Bool=false)
    if name == "z"
        default_bc = "wall"
    elseif name == "r"
        default_bc = "default"
    elseif name == "vperp"
        default_bc = "default"
    else
        default_bc = "zero"
    end
    coord_input_dict = set_defaults_and_check_section!(
        input_dict, name, warn_unexpected;
        # ngrid is number of grid points per element
        ngrid=1,
        # nelement is the number of elements in total
        nelement=1,
        # nelement_local is the number of elements on each process
        nelement_local=-1,
        # L is the box length in this coordinate
        L=1.0,
        # discretization option for the coordinate grid supported options are
        # "chebyshev_pseudospectral", "gausslegendre_pseudospectral" and
        # "finite_difference"
        discretization="chebyshev_pseudospectral",
        # option for implementation of chebyshev discretization: "FFT" or "matrix"
        cheb_option="FFT",
        # finite_difference_option determines the finite difference scheme to be used
        # supported options are "third_order_upwind", "second_order_upwind" and
        # "first_order_upwind"
        finite_difference_option="third_order_upwind",
        element_spacing_option="uniform",
        # which boundary condition to use
        bc=default_bc,
       )
    if coord_input_dict["nelement_local"] == -1 || ignore_MPI
        coord_input_dict["nelement_local"] = coord_input_dict["nelement"]
    end
    if !warn_unexpected && name == "r" && coord_input_dict["bc"] ∉ ("default", "periodic")
        error("Radial boundary conditions other than \"periodic\" should be set in "
              * "[inner_r_bc_*] and [outer_r_bc_*] sections, not in [r], but got "
              * "bc=$(coord_input_dict["bc"]) in [r] section.")
    end
    if name == "vperp" && coord_input_dict["bc"] == "default"
        if coord_input_dict["ngrid"] == 1 && coord_input_dict["nelement"] == 1
            # 1V simulation, so boundary condition should be "none"
            coord_input_dict["bc"] = "none"
        else
            # 2V simulation, so boundary condition should be "zero"
            coord_input_dict["bc"] = "zero"
        end
    end
    # Make a copy so we do not add "name" to the global input_dict
    coord_input_dict = copy(coord_input_dict)
    coord_input_dict["name"] = name

    # Get some parameters that may be used for the boundary condition
    if input_dict === nothing
        boundary_parameters = nothing
    else
        boundary_parameters_defaults = OrderedDict{Symbol,Any}()
        if name == "z"
            # parameter controlling the cutoff of the ion distribution function in the vpa
            # domain at the wall in z
            boundary_parameters_defaults[:epsz] = 0.0
        end
        boundary_parameters_input = set_defaults_and_check_section!(
            input_dict, "$(name)_boundary_condition_parameters", warn_unexpected;
            boundary_parameters_defaults...
           )
        boundary_parameters = Dict_to_NamedTuple(boundary_parameters_input)
    end

    coord_input_dict = deepcopy(coord_input_dict)
    coord_input_dict["boundary_parameters"] = boundary_parameters

    coord_input = Dict_to_NamedTuple(coord_input_dict)

    return coord_input
end

"""
    define_coordinate(input_dict, name; parallel_io::Bool=false,
                      run_directory=nothing, ignore_MPI=false,
                      collision_operator_dim::Bool=true)
    define_coordinate(coord_input::NamedTuple; parallel_io::Bool=false,
                      run_directory=nothing, ignore_MPI=false,
                      collision_operator_dim::Bool=true, irank=0, nrank=1,
                      comm=MPI.COMM_NULL)

Create arrays associated with a given coordinate, setup the coordinate grid, and populate
the coordinate structure containing all of this information.

When `input_dict` is passed, any missing settings will be set with default values.

When `coord_input` is passed, it should be a `NamedTuple` as generated by
[`get_coordinate_input`](@ref), which contains a field for every coordinate input option.
"""
function define_coordinate end

function define_coordinate(input_dict, name, warn_unexpected::Bool=false; kwargs...)

    coord_input = get_coordinate_input(input_dict, name, warn_unexpected=warn_unexpected)

    return define_coordinate(coord_input; kwargs...)
end

function define_coordinate(coord_input::NamedTuple; parallel_io::Bool=false,
                           run_directory=nothing, ignore_MPI=false,
                           collision_operator_dim::Bool=true, irank=0, nrank=1,
                           comm=MPI.COMM_NULL)

    if coord_input.name ∉ ("r", "z")
        if irank != 0 || nrank != 1 || comm != MPI.COMM_NULL
            if comm == MPI.COMM_NULL
                comm_message = "comm is MPI.COMM_NULL"
            else
                comm_message = "comm is not MPI.COMM_NULL"
            end
            error("Distributed-memory MPI is not supported for coordinate "
                  * "$(coord_input.name), but got irank=$irank, nrank=$nrank and "
                  * "$comm_message")
        end
    end

    # total number of grid points is ngrid for the first element
    # plus ngrid-1 unique points for each additional element due
    # to the repetition of a point at the element boundary
    n_global = (coord_input.ngrid-1)*coord_input.nelement + 1
    # local number of points on this process
    n_local = (coord_input.ngrid-1)*coord_input.nelement_local + 1
    # obtain index mapping from full (local) grid to the
    # grid within each element (igrid, ielement)
    igrid, ielement = full_to_elemental_grid_map(coord_input.ngrid,
        coord_input.nelement_local, n_local)
    # obtain (local) index mapping from the grid within each element
    # to the full grid
    imin, imax, igrid_full = elemental_to_full_grid_map(coord_input.ngrid,
                                                        coord_input.nelement_local)
    # initialise the data used to construct the grid
    # boundaries for each element
    element_boundaries = set_element_boundaries(coord_input.nelement,
                                                coord_input.L,
                                                coord_input.element_spacing_option,
                                                coord_input.name)
    # shift and scale factors for each local element
    element_scale, element_shift =
        set_element_scale_and_shift(coord_input.nelement, coord_input.nelement_local,
                                    irank, element_boundaries)

    # initialize the grid and the integration weights associated with the grid
    # also obtain the Chebyshev theta grid and spacing if chosen as discretization option
    grid, wgts, uniform_grid, radau_first_element =
        init_grid(coord_input.ngrid, coord_input.nelement_local, n_global, n_local, irank,
                  coord_input.L, element_scale, element_shift, imin, imax, igrid,
                  coord_input.discretization, coord_input.name)
    # calculate the widths of the cells between neighboring grid points
    cell_width = grid_spacing(grid, n_local)
    # duniform_dgrid is the local derivative of the uniform grid with respect to
    # the coordinate grid
    duniform_dgrid = allocate_float(coord_input.ngrid, coord_input.nelement_local)
    # scratch is an array used for intermediate calculations requiring n entries
    scratch = allocate_float(; Symbol(coord_input.name)=>n_local)
    # scratch_int_nelement_plus_1 is an array used for intermediate calculations requiring
    # nelement+1 entries
    scratch_int_nelement_plus_1 = allocate_int(coord_input.nelement_local + 1)
    if ignore_MPI
        scratch_shared = allocate_float(; Symbol(coord_input.name)=>n_local)
        scratch_shared2 = allocate_float(; Symbol(coord_input.name)=>n_local)
        scratch_shared3 = allocate_float(; Symbol(coord_input.name)=>n_local)
        scratch_shared4 = allocate_float(; Symbol(coord_input.name)=>n_local)
        scratch_shared_int = allocate_int(; Symbol(coord_input.name)=>n_local)
        scratch_shared_int2 = allocate_int(; Symbol(coord_input.name)=>n_local)
    else
        scratch_shared = allocate_shared_float(; Symbol(coord_input.name)=>n_local)
        scratch_shared2 = allocate_shared_float(; Symbol(coord_input.name)=>n_local)
        scratch_shared3 = allocate_shared_float(; Symbol(coord_input.name)=>n_local)
        scratch_shared4 = allocate_shared_float(; Symbol(coord_input.name)=>n_local)
        scratch_shared_int = allocate_shared_int(; Symbol(coord_input.name)=>n_local)
        scratch_shared_int2 = allocate_shared_int(; Symbol(coord_input.name)=>n_local)
    end
    # Initialise scratch_shared* so that the debug checks do not complain when they get
    # printed by `println(io, all_inputs)` in mk_input().
    if block_rank[] == 0
        scratch_shared .= NaN
        scratch_shared2 .= NaN
        scratch_shared3 .= NaN
        scratch_shared4 .= NaN
        scratch_shared_int .= typemin(mk_int)
        scratch_shared_int2 .= typemin(mk_int)
    end
    if !ignore_MPI
        @_block_synchronize()
    end
    # scratch_2d is an array used for intermediate calculations requiring ngrid x nelement entries
    scratch_2d = allocate_float(coord_input.ngrid, coord_input.nelement_local)
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
    elseif irank == nrank-1
        # Include endpoint on final block
        local_io_range = 1:n_local
        global_io_range = irank*(n_local-1)+1:n_global
    else
        # Skip final point, because it is shared with the next block
        # Choose to skip final point in each block so all blocks (except the final one)
        # write a 'chunk' of the same size to the output file. This makes it simple to
        # align HDF5 'chunks' with the data being written
        local_io_range = 1 : n_local-1
        global_io_range = irank*(n_local-1)+1 : (irank+1)*(n_local-1)
    end

    # Precompute some values for Lagrange polynomial evaluation
    other_nodes = allocate_float(coord_input.ngrid-1, coord_input.ngrid,
                                 coord_input.nelement_local)
    one_over_denominator = allocate_float(coord_input.ngrid, coord_input.nelement_local)
    for ielement ∈ 1:coord_input.nelement_local
        if ielement == 1
            this_imin = imin[ielement]
        else
            this_imin = imin[ielement] - 1
        end
        this_imax = imax[ielement]
        this_grid = grid[this_imin:this_imax]
        for j ∈ 1:coord_input.ngrid
            @views other_nodes[1:j-1,j,ielement] .= this_grid[1:j-1]
            @views other_nodes[j:end,j,ielement] .= this_grid[j+1:end]

            if coord_input.ngrid == 1
                one_over_denominator[j,ielement] = 1.0
            else
                one_over_denominator[j,ielement] = 1.0 / prod(this_grid[j] - n for n ∈ @view other_nodes[:,j,ielement])
            end
        end
    end

    periodic = (coord_input.bc == "periodic")
    if irank == nrank - 1
        if periodic
            nextrank = 0
        else
            nextrank = MPI.PROC_NULL
        end
    else
        nextrank = irank + 1
    end
    if irank == 0
        if periodic
            prevrank = nrank - 1
        else
            prevrank = MPI.PROC_NULL
        end
    else
        prevrank = irank - 1
    end

    mask_low = allocate_float(; Symbol(coord_input.name)=>n_local)
    mask_low .= 1.0
    mask_up = allocate_float(; Symbol(coord_input.name)=>n_local)
    mask_up .= 1.0
    zeroval = 1.0e-8
    for i in 1:n_local
        if grid[i] > zeroval
            mask_low[i] = 0.0
        end
        if grid[i] < -zeroval
            mask_up[i] = 0.0
        end
    end
    coord = coordinate(coord_input.name, n_global, n_local, coord_input.ngrid,
        coord_input.nelement, coord_input.nelement_local, nrank, irank, nextrank,
        prevrank, mk_float(coord_input.L), grid, cell_width, igrid, ielement, imin, imax,
        igrid_full, coord_input.discretization, coord_input.finite_difference_option,
        coord_input.cheb_option, coord_input.bc, periodic,
        coord_input.boundary_parameters, wgts, uniform_grid, duniform_dgrid, scratch,
        copy(scratch), copy(scratch), copy(scratch), copy(scratch), copy(scratch),
        copy(scratch), copy(scratch), copy(scratch), copy(scratch),
        scratch_int_nelement_plus_1, scratch_shared, scratch_shared2, scratch_shared3,
        scratch_shared4, scratch_shared_int, scratch_shared_int2, scratch_2d, 
        copy(scratch_2d), send_buffer, receive_buffer, comm, local_io_range, 
        global_io_range, element_scale, element_shift, coord_input.element_spacing_option, 
        element_boundaries, radau_first_element, other_nodes, one_over_denominator, 
        mask_up, mask_low)

    if coord.n == 1 && coord.name == "vperp"
        spectral = null_vperp_dimension_info()
        coord.duniform_dgrid .= 1.0
    elseif coord.n == 1 && occursin("v", coord.name)
        spectral = null_velocity_dimension_info()
        coord.duniform_dgrid .= 1.0
    elseif coord.n == 1
        spectral = null_spatial_dimension_info()
        coord.duniform_dgrid .= 1.0
    elseif coord_input.discretization == "chebyshev_pseudospectral"
        # create arrays needed for explicit Chebyshev pseudospectral treatment in this
        # coordinate and create the plans for the forward and backward fast Chebyshev
        # transforms
        spectral = setup_chebyshev_pseudospectral(coord, run_directory; ignore_MPI=ignore_MPI)
        # obtain the local derivatives of the uniform grid with respect to the used grid
        derivative!(coord.duniform_dgrid, coord.uniform_grid, coord, spectral)
    elseif coord_input.discretization == "gausslegendre_pseudospectral"
        # create arrays needed for explicit GaussLegendre pseudospectral treatment in this
        # coordinate and create the matrices for differentiation
        spectral = setup_gausslegendre_pseudospectral(coord, collision_operator_dim=collision_operator_dim)
        # obtain the local derivatives of the uniform grid with respect to the used grid
        derivative!(coord.duniform_dgrid, coord.uniform_grid, coord, spectral)
    elseif coord_input.discretization == "fourier_pseudospectral"
        if coord.bc ∉ ("periodic", "default")
            error("fourier_pseudospectral discretization can only be used for a periodic dimension")
        end
        spectral = setup_fourier_pseudospectral(coord, run_directory; ignore_MPI=ignore_MPI)
        derivative!(coord.duniform_dgrid, coord.uniform_grid, coord, spectral)
    else
        # finite_difference_info is just a type so that derivative methods, etc., dispatch
        # to the finite difference versions, it does not contain any information.
        spectral = finite_difference_info()
        coord.duniform_dgrid .= 1.0
    end

    return coord, spectral
end

"""
    define_test_coordinate(input_dict::AbstractDict; kwargs...)
    define_test_coordinate(name; collision_operator_dim=true, kwargs...)

Wrapper for `define_coordinate()` to make creating a coordinate for tests slightly less
verbose.

When passing `input_dict`, it must contain a "name" field, and can contain other settings
- "ngrid", "nelement", etc. Options other than "name" will be set using defaults if they
are not passed. `kwargs` are the keyword arguments for [`define_coordinate`](@ref).

The second form allows the coordinate input options to be passed as keyword arguments. For
this form, apart from `collision_operator_dim`, the keyword arguments of
[`define_coordinate`](@ref) cannot be passed, and `ignore_MPI=true` is always set, as this
is most often useful for tests.
"""
function define_test_coordinate end
function define_test_coordinate(input_dict::AbstractDict; kwargs...)
    input_dict = deepcopy(input_dict)
    name = pop!(input_dict, "name")
    return define_coordinate(OptionsDict(name => input_dict), name; kwargs...)
end
function define_test_coordinate(name; collision_operator_dim=true, kwargs...)
    coord_input_dict = OptionsDict(String(k) => v for (k,v) in kwargs)
    coord_input_dict["name"] = name
    return define_test_coordinate(coord_input_dict;
                                  collision_operator_dim=collision_operator_dim,
                                  ignore_MPI=true)
end

function set_element_boundaries(nelement_global, L, element_spacing_option, coord_name)
    # set global element boundaries between [-L/2,L/2]
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
    elseif startswith(element_spacing_option, "compressed")
        element_spacing_option_split = split(element_spacing_option, "_")
        if length(element_spacing_option_split) == 1
            compression_factor = 4.0
        else
            compression_factor = parse(mk_float, element_spacing_option_split[2])
        end

        #shifted_inds = collect(mk_float, 0:nelement_global) .- 0.5 .* nelement_global
        ## Choose element boundary positions to be given by
        ##   s = A*shifted_inds + B*shifted_inds^3
        ## Choose A and B so that, with simin=-nelement_global/2:
        ##   s(simin) = -L/2
        ##   s(simin+1) = -L/2 + L/nelement_global/compression_factor
        ## i.e. so that the grid spacing of the element nearest the wall is
        ## compression_factor smaller than the elements in a uniformly spaced grid.
        ##   simin*A + simin^3*B = -L/2
        ##   A = -(L/2 + simin^3*B)/simin
        ##
        ##   (simin+1)*A + (simin+1)^3*B = -L/2 + L/nelement_global/compression_factor
        ##   -(simin+1)*(L/2 + simin^3*B)/simin + (simin+1)^3*B = -L/2 + L/nelement_global/compression_factor
        ##   -(simin+1)*simin^3*B/simin + (simin+1)^3*B = -L/2 + L/nelement_global/compression_factor + (simin+1)*L/2/simin
        ##   (simin+1)*simin^2*B - (simin+1)^3*B = L/2 - L/nelement_global/compression_factor - (simin+1)*L/2/simin
        ##   B = (L/2 - L/nelement_global/compression_factor - (simin+1)*L/2/simin) / ((simin+1)*simin^2 - (simin+1)^3)

        #simin = -nelement_global / 2.0
        #B = (L/2.0 - L/nelement_global/compression_factor - (simin+1.0)*L/2.0/simin) / ((simin+1.0)*simin^2 - (simin+1.0)^3)
        #A = -(L/2.0 + simin^3*B)/simin

        #@. element_boundaries = A*shifted_inds + B*shifted_inds^3

        # To have the grid spacing change as little as possible from one element to the
        # next, the function that defines the element boundary positions should have
        # constant curvature. The curvature has to change sign at the mid-point of the
        # domain, so this means that the function must be defined piecewise - one piece
        # for the lower half and one for the upper half.
        # An apparently ideal way to do this would be to use a quadratic function, which
        # would mean that the ratio of the sizes of adjacent elements is the same
        # throughout the grid. However, a quadratic would mean a maximum compression
        # factor of 2 before the function becomes non-monotonic, see next:
        # We define the quadratic by making the gradient at the boundaries
        # `compression_factor` larger than the gradient L of the linear function that
        # would give a uniform grid.
        #   s(a) = A*a + B*a*|a|
        # where -0.5≤a≤0.5, and
        #   s(0.5) = L/2
        #   s'(0.5) = compression_factor*L
        # so
        #   A/2 + B/4 = L/2
        #   A + B = compression_factor*L
        # ⇒
        #   B = 2*(compression_factor - 1)*L
        #   A = L - B/2 = L - (compression_factor-1)*L = (2 - compression_factor)*L
        #
        # Therefore instead we choose a circular arc which can be monotonic while reaching
        # any gradient. To make a circle sensible, normalise s by L for this version.
        #   (s-s0)^2 + (a-a0)^2 = r^2
        # where -0.5≤a≤0.5, and
        #   s(0) = 0
        #   s(a) = 1/2
        #   s'(a) = 1/compression_factor
        # and for a>0, a0<0 and s0>0 while for a<0, a0>0 and s0<0. This gives
        #   s0^2 + a0^2 = r^2
        #   (1/2-s0)^2 + (1/2-a0)^2 = r^2 = s0^2 + a0^2
        #   2*(1/2-s0)/compression_factor + 2*(1/2-a0) = 0
        # solving these
        #   a0 = (1/2-s0)/compression_factor + 1/2
        #   1/4 - s0 + s0^2 + 1/4 - a0 + a0^2 = s0^2 + a0^2
        #   1/2 - s0 - a0 = 0
        #   s0 = 1/2 - a0 = 1/2 - (1/2-s0)/compression_factor - 1/2
        #   (1 - 1/compression_factor)*s0 = -1/compression_factor/2
        #   s0 = 1/compression_factor/2/(1/compression_factor-1)
        if abs(compression_factor - 1.0) < 1.0e-12
            # compression_factor is too close to 1, which would be a singular value where
            # s0=∞ and a0=-∞, so just use constant spacing.
            for j in 1:nelement_global+1
                element_boundaries[j] = L*((j-1)/(nelement_global) - 0.5)
            end
        else
            s0 = 1.0 / compression_factor / 2.0 / (1.0 / compression_factor - 1.0)
            a0 = (0.5 - s0)/compression_factor + 0.5
            a = collect(0:nelement_global) ./ nelement_global .- 0.5
            mid_ind_plus = (nelement_global + 1) ÷ 2 + 1
            mid_ind_minus = nelement_global ÷ 2 + 1
            @. element_boundaries[1:mid_ind_minus] =
                -L * (sqrt(s0^2 + a0^2 - (a[1:mid_ind_minus]+a0)^2) + s0)
            @. element_boundaries[mid_ind_plus:end] =
                L * (sqrt(s0^2 + a0^2 - (a[mid_ind_plus:end]-a0)^2) + s0)
        end
    elseif startswith(element_spacing_option, "coarse_tails")
        # Element boundaries at
        #
        # x = (1 + (BT)^2 / 3) T tan(BT a) / (1 + (BT a)^2 / 3)
        #
        # where a = (i - 1 - c) / c, c = (n-1)/2, i is the grid index, so that a=-1 at
        # i=1, a=1 at i=n and a=0 on the central grid point (if n is odd, so that there is
        # a central point). Also B=1/T*atan(L/2T).
        #
        # Choosing x∼tan(a) gives dx/da∼1+x^2 so that we get grid spacing roughly
        # proportional to x^2 for large |x|, which for w_∥ advection compensates the
        # w_∥^2 terms in moment-kinetics so that the CFL condition should be roughly
        # constant across the grid. The constant B.T multiplying a inside the tan() is
        # chosen so that the transition between roughly constant spacing and roughly x^2
        # spacing happens at x=T. The (1 + (BT a)^2 / 3) denominator removes the quadratic
        # part of the Taylor expansion of dx/da around a=0 so that we get a flatter region
        # of grid spacing for |x|<T. The rest of the factors ensure that x(±1)=±L/2.
        #
        # We choose T=5 so that the electron sheath cutoff, which is around
        # v_∥/vth≈3≈w_∥ is captured in the finer grid spacing in the 'constant' region.
        params = split(element_spacing_option, "coarse_tails")[2]
        if params == ""
            T = 5.0
        else
            T = parse(mk_float, params)
        end
        BT = atan(L / 2.0 / T)
        a = (collect(1:nelement_global+1) .- 1 .- nelement_global ./ 2.0) ./ (nelement_global ./ 2.0)
        @. element_boundaries = tan(BT * a) / (1.0 + (BT * a)^2 / 3.0)

        # Rather than writing out all the necessary factors explicitly, just normalise the
        # element_boundaries array so that its first/last values are ±L/2.
        @. element_boundaries *= L / 2.0 / element_boundaries[end]
    elseif element_spacing_option == "uniform" || (element_spacing_option == "sqrt" && nelement_global < 4) # uniform spacing 
        for j in 1:nelement_global+1
            element_boundaries[j] = L*((j-1)/(nelement_global) - 0.5)
        end
    else 
        println("ERROR: element_spacing_option: ",element_spacing_option, " not supported")
    end
    if coord_name == "vperp"
        #shift so that the range of element boundaries is [0,L]
        for j in 1:nelement_global+1
            element_boundaries[j] += L/2.0
        end
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
function init_grid(ngrid, nelement_local, n_global, n_local, irank, L, element_scale,
                   element_shift, imin, imax, igrid, discretization, name)
    uniform_grid = equally_spaced_grid(n_global, n_local, irank, L)
    uniform_grid_shifted = equally_spaced_grid_shifted(n_global, n_local, irank, L)
    radau_first_element = false
    if n_global == 1
        grid = allocate_float(; Symbol(name)=>n_local)
        grid[1] = 0.0
        wgts = allocate_float(; Symbol(name)=>n_local)
        wgts[1] = 1.0
    elseif discretization == "chebyshev_pseudospectral"
        if name == "vperp"
            # initialize chebyshev grid defined on [-L/2,L/2]
            grid, wgts = scaled_chebyshev_radau_grid(name, ngrid, nelement_local, n_local,
                                                     element_scale, element_shift, imin,
                                                     imax, irank)
            # Integrals over vperp are actually 2d integrals
            #   ∫d^2(v_⟂)=∫dv_⟂ v_⟂∫dϕ=2π∫dv_⟂ v_⟂
            # so need to multiply the weight by 2*π*v_⟂
            @. wgts = 2.0 * π * grid * wgts
            radau_first_element = true
        else
            # initialize chebyshev grid defined on [-L/2,L/2]
            # with n grid points chosen to facilitate
            # the fast Chebyshev transform (aka the discrete cosine transform)
            # needed to obtain Chebyshev spectral coefficients
            # 'wgts' are the integration weights attached to each grid points
            # that are those associated with Clenshaw-Curtis quadrature
            grid, wgts = scaled_chebyshev_grid(name, ngrid, nelement_local, n_local,
                                               element_scale, element_shift, imin, imax)
        end
    elseif discretization == "gausslegendre_pseudospectral"
        if name == "vperp"
            # use a radau grid for the 1st element near the origin
            grid, wgts = scaled_gauss_legendre_radau_grid(ngrid, nelement_local, n_local, element_scale, element_shift, imin, imax, irank)
            # Integrals over vperp are actually 2d integrals
            #   ∫d^2(v_⟂)=∫dv_⟂ v_⟂∫dϕ=2π∫dv_⟂ v_⟂
            # so need to multiply the weight by 2*π*v_⟂
            @. wgts = 2.0 * π * grid * wgts
            radau_first_element = true
        else
            grid, wgts = scaled_gauss_legendre_lobatto_grid(ngrid, nelement_local, n_local, element_scale, element_shift, imin, imax)
        end
    elseif discretization == "fourier_pseudospectral"
        if name == "vperp"
            error("vperp is not periodic, so cannot use fourier_pseudospectral discretization.")
        end
        if nelement_local > 1
            error("fourier_pseudospectral requires a single element")
        end
        grid = uniform_grid
        wgts = trapezium_weights(grid)
    elseif discretization == "finite_difference"
        if name == "vperp"
            # initialize equally spaced grid defined on [0,L]
            grid = uniform_grid_shifted
            # use composite Simpson's rule to obtain integration weights associated with this coordinate
            wgts = composite_simpson_weights(grid)
            # Integrals over vperp are actually 2d integrals
            #   ∫d^2(v_⟂)=∫dv_⟂ v_⟂∫dϕ=2π∫dv_⟂ v_⟂
            # so need to multiply the weight by 2*π*v_⟂
            @. wgts = 2.0 * π * grid * wgts
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
    return grid, wgts, uniform_grid, radau_first_element
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
    if n == 1
        d[1] = 1.0
    else
        d[1] = grid[2] - grid[1]
        for i ∈ 2:n-1
            d[i] =  0.5*(grid[i+1]-grid[i-1])
        end
        d[n] = grid[n] - grid[n-1]
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
    igrid_full = allocate_int(ngrid, nelement)
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
        
        for j in 1:nelement
            for i in 1:ngrid
                igrid_full[i,j] = i + (j - 1)*(ngrid - 1)
            end
        end
    end
    return imin, imax, igrid_full
end

end
