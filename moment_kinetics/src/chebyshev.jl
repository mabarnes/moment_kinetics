"""
"""
module chebyshev

export update_fcheby!
export update_df_chebyshev!
export setup_chebyshev_pseudospectral
export scaled_chebyshev_grid
export scaled_chebyshev_radau_grid
export chebyshev_spectral_derivative!
export chebyshev_info

using LinearAlgebra: mul!
using FFTW
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float, allocate_complex
using ..clenshaw_curtis: clenshawcurtisweights
import ..calculus: elementwise_derivative!
import ..interpolation: interpolate_to_grid_1d!
using ..moment_kinetics_structs: discretization_info

"""
Chebyshev pseudospectral discretization
"""
struct chebyshev_base_info{TForward <: FFTW.cFFTWPlan, TBackward <: AbstractFFTs.ScaledPlan}
    # fext is an array for storing f(z) on the extended domain needed
    # to perform complex-to-complex FFT using the fact that f(theta) is even in theta
    fext::Array{Complex{mk_float},1}
    # Chebyshev spectral coefficients of distribution function f
    # first dimension contains location within element
    # second dimension indicates the element
    f::Array{mk_float,2}
    # Chebyshev spectral coefficients of derivative of f
    df::Array{mk_float,1}
    # plan for the complex-to-complex, in-place, forward Fourier transform on Chebyshev-Gauss-Lobatto/Radau grid
    forward::TForward
    # plan for the complex-to-complex, in-place, backward Fourier transform on Chebyshev-Gauss-Lobatto/Radau grid
    # backward_transform::FFTW.cFFTWPlan
    backward::TBackward
    # elementwise differentiation matrix (ngrid*ngrid)
    Dmat::Array{mk_float,2}
    # elementwise differentiation vector (ngrid) for the point x = -1
    D0::Array{mk_float,1}
end

struct chebyshev_info{TForward <: FFTW.cFFTWPlan, TBackward <: AbstractFFTs.ScaledPlan} <: discretization_info
    lobatto::chebyshev_base_info{TForward, TBackward}
    radau::chebyshev_base_info{TForward, TBackward}
end

"""
create arrays needed for explicit Chebyshev pseudospectral treatment
and create the plans for the forward and backward fast Fourier transforms
"""
function setup_chebyshev_pseudospectral(coord)
    lobatto = setup_chebyshev_pseudospectral_lobatto(coord)
    radau = setup_chebyshev_pseudospectral_radau(coord)
    return chebyshev_info(lobatto,radau)
end

function setup_chebyshev_pseudospectral_lobatto(coord)
    # ngrid_fft is the number of grid points in the extended domain
    # in z = cos(theta).  this is necessary to turn a cosine transform on [0,π]
    # into a complex transform on [0,2π], which is more efficient in FFTW
    ngrid_fft = 2*(coord.ngrid-1)
    # create array for f on extended [0,2π] domain in theta = ArcCos[z]
    fext = allocate_complex(ngrid_fft)
    # create arrays for storing Chebyshev spectral coefficients of f and f'
    fcheby = allocate_float(coord.ngrid, coord.nelement_local)
    dcheby = allocate_float(coord.ngrid)
    # setup the plans for the forward and backward Fourier transforms
    forward_transform = plan_fft!(fext, flags=FFTW.MEASURE)
    backward_transform = plan_ifft!(fext, flags=FFTW.MEASURE)
    # create array for differentiation matrix 
    Dmat = allocate_float(coord.ngrid, coord.ngrid)
    cheb_derivative_matrix_elementwise!(Dmat,coord.ngrid)
    D0 = allocate_float(coord.ngrid)
    D0 .= Dmat[1,:]
    # return a structure containing the information needed to carry out
    # a 1D Chebyshev transform
    return chebyshev_base_info(fext, fcheby, dcheby, forward_transform, backward_transform, Dmat, D0)
end

function setup_chebyshev_pseudospectral_radau(coord)
        # ngrid_fft is the number of grid points in the extended domain
        # in z = cos(theta).  this is necessary to turn a cosine transform on [0,π]
        # into a complex transform on [0,2π], which is more efficient in FFTW
        ngrid_fft = 2*coord.ngrid - 1
        # create array for f on extended [0,2π] domain in theta = ArcCos[z]
        fext = allocate_complex(ngrid_fft)
        # create arrays for storing Chebyshev spectral coefficients of f and f'
        fcheby = allocate_float(coord.ngrid, coord.nelement_local)
        dcheby = allocate_float(coord.ngrid)
        # setup the plans for the forward and backward Fourier transforms
        forward_transform = plan_fft!(fext, flags=FFTW.MEASURE)
        backward_transform = plan_ifft!(fext, flags=FFTW.MEASURE)
        # create array for differentiation matrix 
        Dmat = allocate_float(coord.ngrid, coord.ngrid)
        cheb_derivative_matrix_elementwise_radau_by_FFT!(Dmat, coord, fcheby, dcheby, fext, forward_transform)
        D0 = allocate_float(coord.ngrid)
        cheb_lower_endpoint_derivative_vector_elementwise_radau_by_FFT!(D0, coord, fcheby, dcheby, fext, forward_transform)
        # return a structure containing the information needed to carry out
        # a 1D Chebyshev transform
        return chebyshev_base_info(fext, fcheby, dcheby, forward_transform, backward_transform, Dmat, D0)
end

"""
initialize chebyshev grid scaled to interval [-box_length/2, box_length/2]
we no longer pass the box_length to this function, but instead pass precomputed
arrays element_scale and element_shift that are needed to compute the grid.

ngrid -- number of points per element (including boundary points)
nelement_local -- number of elements in the local (distributed memory MPI) grid
n -- total number of points in the local grid (excluding duplicate points)
element_scale -- the scale factor in the transform from the coordinates 
                 where the element limits are -1, 1 to the coordinate where
                 the limits are Aj = coord.grid[imin[j]-1] and Bj = coord.grid[imax[j]]
                 element_scale = 0.5*(Bj - Aj)
element_shift -- the centre of the element in the extended grid coordinate
                 element_shift = 0.5*(Aj + Bj)
imin -- the array of minimum indices of each element on the extended grid.
        By convention, the duplicated points are not included, so for element index j > 1
        the lower boundary point is actually imin[j] - 1
imax -- the array of maximum indices of each element on the extended grid.
"""
function scaled_chebyshev_grid(ngrid, nelement_local, n,
			element_scale, element_shift, imin, imax)
    # initialize chebyshev grid defined on [1,-1]
    # with n grid points chosen to facilitate
    # the fast Chebyshev transform (aka the discrete cosine transform)
    # needed to obtain Chebyshev spectral coefficients
    # this grid goes from +1 to -1
    chebyshev_grid = chebyshevpoints(ngrid)
    # create array for the full grid
    grid = allocate_float(n)
    
    # account for the fact that the minimum index needed for the chebyshev_grid
    # within each element changes from 1 to 2 in going from the first element
    # to the remaining elements
    k = 1
    @inbounds for j ∈ 1:nelement_local
        scale_factor = element_scale[j]
        shift = element_shift[j]
        # reverse the order of the original chebyshev_grid (ran from [1,-1])
        # and apply the scale factor and shift
        grid[imin[j]:imax[j]] .= (reverse(chebyshev_grid)[k:ngrid] * scale_factor) .+ shift
        # after first element, increase minimum index for chebyshev_grid to 2
        # to avoid double-counting boundary element
        k = 2
    end
    wgts = clenshaw_curtis_weights(ngrid, nelement_local, n, imin, imax, element_scale)
    return grid, wgts
end

function scaled_chebyshev_radau_grid(ngrid, nelement_local, n,
			element_scale, element_shift, imin, imax, irank)
    # initialize chebyshev grid defined on [1,-1]
    # with n grid points chosen to facilitate
    # the fast Chebyshev transform (aka the discrete cosine transform)
    # needed to obtain Chebyshev spectral coefficients
    # this grid goes from +1 to -1
    chebyshev_grid = chebyshevpoints(ngrid)
    chebyshev_radau_grid = chebyshev_radau_points(ngrid)
    # create array for the full grid
    grid = allocate_float(n)
    # setup the scale factor by which the Chebyshev grid on [-1,1]
    # is to be multiplied to account for the full domain [-L/2,L/2]
    # and the splitting into nelement elements with ngrid grid points
    if irank == 0 # use a Chebyshev-Gauss-Radau element for the lowest element on rank 0
        scale_factor = element_scale[1]
        shift = element_shift[1]
        grid[imin[1]:imax[1]] .= (chebyshev_radau_grid[1:ngrid] * scale_factor) .+ shift
        # account for the fact that the minimum index needed for the chebyshev_grid
        # within each element changes from 1 to 2 in going from the first element
        # to the remaining elements
        k = 2
        @inbounds for j ∈ 2:nelement_local
            scale_factor = element_scale[j]
            shift = element_shift[j]
            # reverse the order of the original chebyshev_grid (ran from [1,-1])
            # and apply the scale factor and shift
            grid[imin[j]:imax[j]] .= (reverse(chebyshev_grid)[k:ngrid] * scale_factor) .+ shift
        end
        wgts = clenshaw_curtis_radau_weights(ngrid, nelement_local, n, imin, imax, element_scale)
    else
        # account for the fact that the minimum index needed for the chebyshev_grid
        # within each element changes from 1 to 2 in going from the first element
        # to the remaining elements
        k = 1
        @inbounds for j ∈ 1:nelement_local
            scale_factor = element_scale[j]
            shift = element_shift[j]
            # reverse the order of the original chebyshev_grid (ran from [1,-1])
            # and apply the scale factor and shift
            grid[imin[j]:imax[j]] .= (reverse(chebyshev_grid)[k:ngrid] * scale_factor) .+ shift
            # after first element, increase minimum index for chebyshev_grid to 2
            # to avoid double-counting boundary element
            k = 2
        end
        wgts = clenshaw_curtis_weights(ngrid, nelement_local, n, imin, imax, element_scale)
    end
    return grid, wgts
end

"""
    elementwise_derivative!(coord, ff, chebyshev::chebyshev_info)

Chebyshev transform f to get Chebyshev spectral coefficients and use them to calculate f'.
"""
function elementwise_derivative!(coord, ff, chebyshev::chebyshev_info)
    df = coord.scratch_2d
    # define local variable nelement for convenience
    nelement = coord.nelement_local
    # check array bounds
    @boundscheck nelement == size(chebyshev.lobatto.f,2) || throw(BoundsError(chebyshev.lobatto.f))
    @boundscheck nelement == size(chebyshev.radau.f,2) || throw(BoundsError(chebyshev.radau.f))
    @boundscheck nelement == size(df,2) && coord.ngrid == size(df,1) || throw(BoundsError(df))
    # note that one must multiply by a coordinate transform factor 1/element_scale[j]
    # for each element j to get derivative on the extended grid
    
    if coord.cheb_option == "matrix"
        # variable k will be used to avoid double counting of overlapping point
        # at element boundaries (see below for further explanation)
        k = 0
        j = 1 # the first element
        imin = coord.imin[j]-k
        # imax is the maximum index on the full grid for this (jth) element
        imax = coord.imax[j]        
        if coord.name == "vperp" && coord.irank == 0 # differentiate this element with the Radau scheme
            @views mul!(df[:,j],chebyshev.radau.Dmat[:,:],ff[imin:imax])
        else #differentiate using the Lobatto scheme
            @views mul!(df[:,j],chebyshev.lobatto.Dmat[:,:],ff[imin:imax])
        end
        for i ∈ 1:coord.ngrid
            df[i,j] /= coord.element_scale[j]
        end
        # calculate the Chebyshev derivative on each element
        @inbounds for j ∈ 2:nelement
            # imin is the minimum index on the full grid for this (jth) element
            # the 'k' below accounts for the fact that the first element includes
            # both boundary points, while each additional element shares a boundary
            # point with neighboring elements.  the choice was made when defining
            # coord.imin to exclude the lower boundary point in each element other
            # than the first so that no point is double-counted
            k = 1 
            imin = coord.imin[j]-k
            # imax is the maximum index on the full grid for this (jth) element
            imax = coord.imax[j]
            @views mul!(df[:,j],chebyshev.lobatto.Dmat[:,:],ff[imin:imax])
            for i ∈ 1:coord.ngrid
                df[i,j] /= coord.element_scale[j]
            end
        end
    elseif coord.cheb_option == "FFT"   
        # note that one must multiply by  1/element_scale[j] get derivative
        # in scaled coordinate on element j
        
        # variable k will be used to avoid double counting of overlapping point
        # at element boundaries (see below for further explanation)
        k = 0
        j = 1 # the first element
        if coord.name == "vperp" && coord.irank == 0 # differentiate this element with the Radau scheme
            imin = coord.imin[j]-k
            # imax is the maximum index on the full grid for this (jth) element
            imax = coord.imax[j]
            @views chebyshev_radau_derivative_single_element!(df[:,j], ff[imin:imax],
                chebyshev.radau.f[:,j], chebyshev.radau.df, chebyshev.radau.fext, chebyshev.radau.forward, coord)
            # and multiply by scaling factor needed to go
            # from Chebyshev z coordinate to actual z
            for i ∈ 1:coord.ngrid
                df[i,j] /= coord.element_scale[j]
            end
        else #differentiate using the Lobatto scheme
            imin = coord.imin[j]-k
            # imax is the maximum index on the full grid for this (jth) element
            imax = coord.imax[j]
            @views chebyshev_derivative_single_element!(df[:,j], ff[imin:imax],
                chebyshev.lobatto.f[:,j], chebyshev.lobatto.df, chebyshev.lobatto.fext, chebyshev.lobatto.forward, coord)
            # and multiply by scaling factor needed to go
            # from Chebyshev z coordinate to actual z
            for i ∈ 1:coord.ngrid
                df[i,j] /= coord.element_scale[j]
            end
        end
        # calculate the Chebyshev derivative on each element
        @inbounds for j ∈ 2:nelement
            # imin is the minimum index on the full grid for this (jth) element
            # the 'k' below accounts for the fact that the first element includes
            # both boundary points, while each additional element shares a boundary
            # point with neighboring elements.  the choice was made when defining
            # coord.imin to exclude the lower boundary point in each element other
            # than the first so that no point is double-counted
            k = 1 
            imin = coord.imin[j]-k
            # imax is the maximum index on the full grid for this (jth) element
            imax = coord.imax[j]
            @views chebyshev_derivative_single_element!(df[:,j], ff[imin:imax],
                chebyshev.lobatto.f[:,j], chebyshev.lobatto.df, chebyshev.lobatto.fext, chebyshev.lobatto.forward, coord)
            # and multiply by scaling factor needed to go
            # from Chebyshev z coordinate to actual z
            for i ∈ 1:coord.ngrid
                df[i,j] /= coord.element_scale[j]
            end        
        end
    else
        println("ERROR: ", coord.cheb_option, " NOT SUPPORTED")
    end
    return nothing
end

"""
    elementwise_derivative!(coord, ff, adv_fac, spectral::chebyshev_info)

Chebyshev transform f to get Chebyshev spectral coefficients and use them to calculate f'.

Note: Chebyshev derivative does not make use of upwinding information within each element.
"""
function elementwise_derivative!(coord, ff, adv_fac, spectral::chebyshev_info)
    return elementwise_derivative!(coord, ff, spectral)
end

"""
"""
function chebyshev_derivative_single_element!(df, ff, cheby_f, cheby_df, cheby_fext,
        forward, coord)
    # calculate the Chebyshev coefficients of the real-space function ff and return
    # as cheby_f
    chebyshev_forward_transform!(cheby_f, cheby_fext, ff, forward, coord.ngrid)
    # calculate the Chebyshev coefficients of the derivative of ff with respect to coord.grid
    chebyshev_spectral_derivative!(cheby_df, cheby_f)
    # inverse Chebyshev transform to get df/dcoord
    chebyshev_backward_transform!(df, cheby_fext, cheby_df, forward, coord.ngrid)
end

"""
Chebyshev transform f to get Chebyshev spectral coefficients
"""
function update_fcheby!(cheby, ff, coord)
    k = 0
    # loop over the different elements and perform a Chebyshev transform
    # using the grid within each element
    @inbounds for j ∈ 1:coord.nelement
        # imin is the minimum index on the full grid for this (jth) element
        # the 'k' below accounts for the fact that the first element includes
        # both boundary points, while each additional element shares a boundary
        # point with neighboring elements.  the choice was made when defining
        # coord.imin to exclude the lower boundary point in each element other
        # than the first so that no point is double-counted
        imin = coord.imin[j]-k
        # imax is the maximum index on the full grid for this (jth) element
        imax = coord.imax[j]
        chebyshev_forward_transform!(view(cheby.f,:,j),
            cheby.fext, view(ff,imin:imax), cheby.forward, coord.ngrid)
        k = 1
    end
    return nothing
end

"""
compute the Chebyshev spectral coefficients of the spatial derivative of f
"""
function update_df_chebyshev!(df, chebyshev, coord)
    ngrid = coord.ngrid
    nelement = coord.nelement
    L = coord.L
    @boundscheck nelement == size(chebyshev.f,2) || throw(BoundsError(chebyshev.f))
    @boundscheck nelement == size(df,2) && ngrid == size(df,1) || throw(BoundsError(df))
    # obtain Chebyshev spectral coefficients of f'[z]
    # note that must multiply by 2/Lz to get derivative
    # in scaled coordinate
    scale_factor = 2*nelement/L
    # scan over elements
    @inbounds for j ∈ 1:nelement
        chebyshev_spectral_derivative!(chebyshev.df,view(chebyshev.f,:,j))
        # inverse Chebyshev transform to get df/dz
        # and multiply by scaling factor needed to go
        # from Chebyshev z coordinate to actual z
        chebyshev_backward_transform!(view(df,:,j), chebyshev.fext, chebyshev.df, chebyshev.forward, coord.ngrid)
        for i ∈ 1:ngrid
            df[i,j] *= scale_factor
        end
    end
    return nothing
end

"""
use Chebyshev basis to compute the first derivative of f
"""
function chebyshev_spectral_derivative!(df,f)
    m = length(f)
    @boundscheck m == length(df) || throw(BoundsError(df))
    @inbounds begin
        df[m] = 0.
        df[m-1] = 2*(m-1)*f[m]
        df[m-2] = 2*(m-2)*f[m-1]
        for i ∈ m-3:-1:2
            df[i] = 2*i*f[i+1] + df[i+2]
        end
        df[1] = f[2] + 0.5*df[3]
    end
end
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
chebyshev : chebyshev_info
    struct containing information for Chebyshev transforms

Note that this routine does not support Gauss-Chebyshev-Radau elements
"""
function interpolate_to_grid_1d!(result, newgrid, f, coord, chebyshev::chebyshev_info)
    # define local variable nelement for convenience
    nelement = coord.nelement_local
    # check array bounds
    @boundscheck nelement == size(chebyshev.lobatto.f,2) || throw(BoundsError(chebyshev.lobatto.f))

    n_new = size(newgrid)[1]
    # Find which points belong to which element.
    # kstart[j] contains the index of the first point in newgrid that is within element
    # j, and kstart[nelement+1] is n_new if the last point is within coord.grid, or the
    # index of the first element outside coord.grid otherwise.
    # Assumes points in newgrid are sorted.
    # May not be the most efficient algorithm.
    # Find start/end points for each element, storing their indices in kstart
    kstart = Vector{mk_int}(undef, nelement+1)
    # set the starting index by finding the start of coord.grid
    kstart[1] = searchsortedfirst(newgrid, coord.grid[1])
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

    # First element includes both boundary points, while all others have only one (to
    # avoid duplication), so calculate the first element outside the loop.
    if kstart[1] < kstart[2]
        imin = coord.imin[1]
        imax = coord.imax[1]
        kmin = kstart[1]
        kmax = kstart[2] - 1
        @views chebyshev_interpolate_single_element!(result[kmin:kmax],
                                                     newgrid[kmin:kmax],
                                                     f[imin:imax],
                                                     imin, imax, coord, chebyshev.lobatto)
    end
    @inbounds for j ∈ 2:nelement
        kmin = kstart[j]
        kmax = kstart[j+1] - 1
        if kmin <= kmax
            imin = coord.imin[j] - 1
            imax = coord.imax[j]
            @views chebyshev_interpolate_single_element!(result[kmin:kmax],
                                                         newgrid[kmin:kmax],
                                                         f[imin:imax],
                                                         imin, imax, coord, chebyshev.lobatto)
        end
    end

    for k ∈ kstart[nelement+1]:n_new
        result[k] = f[end] * exp(-(newgrid[k] - coord.grid[end])^2)
    end

    return nothing
end

"""
"""
function chebyshev_interpolate_single_element!(result, newgrid, f, imin, imax, coord, chebyshev::chebyshev_base_info)
    # Temporary buffer to store Chebyshev coefficients
    cheby_f = chebyshev.df

    # Need to transform newgrid values to a scaled z-coordinate associated with the
    # Chebyshev coefficients to get the interpolated function values. Transform is a
    # shift and scale so that the element coordinate goes from -1 to 1
    shift = 0.5 * (coord.grid[imin] + coord.grid[imax])
    scale = 2.0 / (coord.grid[imax] - coord.grid[imin])

    # Get Chebyshev coefficients
    chebyshev_forward_transform!(cheby_f, chebyshev.fext, f, chebyshev.forward, coord.ngrid)

    for i ∈ 1:length(newgrid)
        x = newgrid[i]
        z = scale * (x - shift)
        # Evaluate sum of Chebyshev polynomials at z using recurrence relation
        cheb1 = 1.0
        cheb2 = z
        result[i] = cheby_f[1] * cheb1 + cheby_f[2] * cheb2
        for coef ∈ @view(cheby_f[3:end])
            cheb1, cheb2 = cheb2, 2.0 * z * cheb2 - cheb1
            result[i] += coef * cheb2
        end
    end

    return nothing
end

"""
returns wgts array containing the integration weights associated
with all grid points for Clenshaw-Curtis quadrature
"""
function clenshaw_curtis_weights(ngrid, nelement_local, n, imin, imax, element_scale)
    # create array containing the integration weights
    wgts = zeros(mk_float, n)
    # calculate the modified Chebshev moments of the first kind
    μ = chebyshevmoments(ngrid)
    # calculate the raw weights for a normalised grid on [-1,1]
    w = clenshawcurtisweights(μ)
    @inbounds begin
        # calculate the weights within a single element and
        # scale to account for modified domain (not [-1,1])
        wgts[1:ngrid] = w*element_scale[1]
        if nelement_local > 1
            for j ∈ 2:nelement_local
                wgts[imin[j]-1:imax[j]] .+= w*element_scale[j]
            end
        end
    end
    return wgts
end

function clenshaw_curtis_radau_weights(ngrid, nelement_local, n, imin, imax, element_scale)
    # create array containing the integration weights
    wgts = zeros(mk_float, n)
    # calculate the modified Chebshev moments of the first kind
    μ = chebyshevmoments(ngrid)
    wgts_lobatto = clenshawcurtisweights(μ)
    wgts_radau = chebyshev_radau_weights(μ, ngrid)
    @inbounds begin
        # calculate the weights within a single element and
        # scale to account for modified domain (not [-1,1])
        wgts[1:ngrid] .= wgts_radau[1:ngrid]*element_scale[1]
        if nelement_local > 1
            for j ∈ 2:nelement_local
                # account for double-counting of points at inner element boundaries
                wgts[imin[j]-1] += wgts_lobatto[1]*element_scale[j]
                # assign weights for interior of elements and one boundary point
                wgts[imin[j]:imax[j]] .= wgts_lobatto[2:ngrid]*element_scale[j]
            end
        end
    end
    return wgts
end

"""
compute and return modified Chebyshev moments of the first kind:
∫dx Tᵢ(x) over range [-1,1]
"""
function chebyshevmoments(N)
    μ = zeros(N)
    @inbounds for i = 0:2:N-1
        μ[i+1] = 2/(1-i^2)
    end
    return μ
end

"""
returns the Chebyshev-Gauss-Lobatto grid points on an n point grid
"""
function chebyshevpoints(n)
    grid = allocate_float(n)
    nfac = 1/(n-1)
    @inbounds begin
        # calculate z = cos(θ) ∈ [1,-1]
        for j ∈ 1:n
            grid[j] = cospi((j-1)*nfac)
        end
    end
    return grid
end

function chebyshev_radau_points(n)
    grid = allocate_float(n)
    nfac = 1.0/(n-0.5)
    @inbounds begin
        # calculate z = cos(θ) ∈ (-1,1]
        for j ∈ 1:n
            grid[j] = cospi((n-j)*nfac)
        end
    end
    return grid
end

function chebyshev_radau_weights(moments::Array{mk_float,1}, n)
    # input should have values moments[j] = (cos(pi j) + 1)/(1-j^2) for j >= 0
    nfft = 2*n - 1
    # create array for moments on extended [0,2π] domain in theta = ArcCos[z]
    fext = allocate_complex(nfft)
    # make fft plan
    forward_transform = plan_fft!(fext, flags=FFTW.MEASURE)
    # assign values of fext from moments 
    @inbounds begin
        for j ∈ 1:n
            fext[j] = complex(moments[j],0.0)
        end
        for j ∈ 1:n-1
            fext[n+j] = fext[n-j+1]
        end
    end
    # perform the forward, complex-to-complex FFT in-place (fext is overwritten)
    forward_transform*fext
    # use reality + evenness of moments to eliminate unncessary information
    # also sort out normalisation and order of array
    # note that fft order output is reversed compared to the order of 
    # the grid chosen, which runs from (-1,1]
    wgts = allocate_float(n)
    @inbounds begin
        for j ∈ 2:n
            wgts[n-j+1] = 2.0*real(fext[j])/nfft
        end
        wgts[n] = real(fext[1])/nfft
    end
    return wgts
end

"""
takes the real function ff on a Chebyshev grid in z (domain [-1, 1]),
which corresponds to the domain [π, 2π] in variable theta = ArcCos(z).
interested in functions of form f(z) = sum_n c_n T_n(z)
using T_n(cos(theta)) = cos(n*theta) and z = cos(theta) gives
f(z) = sum_n c_n cos(n*theta)
thus a Chebyshev transform is equivalent to a discrete cosine transform
doing this directly turns out to be slower than extending the domain
from [0, 2pi] and using the fact that f(z) must be even (as cosines are all even)
on this extended domain, can do a standard complex-to-complex fft
fext is an array used to store f(theta) on the extended grid theta ∈ [0,2π)
ff is f(theta) on the grid [π,2π]
the Chebyshev coefficients of ff are calculated and stored in chebyf
n is the number of grid points on the Chebyshev-Gauss-Lobatto grid
transform is the plan for the complex-to-complex, in-place fft
"""
function chebyshev_forward_transform!(chebyf, fext, ff, transform, n)
    # ff as input is f(z) on the domain [-1,1]
    # corresponding to f(theta) on the domain [π,2π]
    # must extend f(theta) using even-ness about theta=π onto domain [0,2π]
    @inbounds begin
        # first, fill in values for f on domain θ ∈ [0,π]
        # using even-ness of f about θ = π
        for j ∈ 0:n-1
            fext[n-j] = complex(ff[j+1],0.0)
        end
        # next, fill in values for f on domain θ ∈ (π,2π)
        for j ∈ 1:n-2
            fext[n+j] = fext[n-j]
        end
    end
    # perform the forward, complex-to-complex FFT in-place (cheby.fext is overwritten)
    transform*fext
    # use reality + evenness of f to eliminate unncessary information
    # and obtain Chebyshev spectral coefficients for this element
    # also sort out normalisation
    @inbounds begin
        nm = n-1
        nfac = 1/nm
        for j ∈ 2:nm
            chebyf[j] = real(fext[j])*nfac
        end
        nfac *= 0.5
        chebyf[1] = real(fext[1])*nfac
        chebyf[n] = real(fext[n])*nfac
    end
    return nothing
end

"""
"""
function chebyshev_backward_transform!(ff, fext, chebyf, transform, n)
    # chebyf as input contains Chebyshev spectral coefficients
    # need to use reality condition to extend onto negative frequency domain
    @inbounds begin
        # first, fill in values for fext corresponding to positive frequencies
        for j ∈ 2:n-1
            fext[j] = chebyf[j]*0.5
        end
        # next, fill in values for fext corresponding to negative frequencies
        # using fext(-k) = conjg(fext(k)) = fext(k)
        # usual FFT ordering with j=1 <-> k=0, followed by ascending k up to kmax
        # and then descending from -kmax down to -dk
        for j ∈ 1:n-2
            fext[n+j] = fext[n-j]
        end
        # fill in zero frequency mode, which is special in that it does not require
        # the 1/2 scale factor
        fext[1] = chebyf[1]
        fext[n] = chebyf[n]
    end
    # perform the backward, complex-to-complex FFT in-place (fext is overwritten)
    transform*fext
    # fext now contains a real function on θ [0,2π)
    # all we need is the real(fext) on [π,2π]
    # also sort out normalisation
    @inbounds begin
        nm = n-1
        # fill in entries for ff on θ ∈ [π,2π)
        for j ∈ 1:nm
            ff[j] = real(fext[j+nm])
        end
        # fill in ff[2π] and normalise
        ff[n] = real(fext[1])
    end
    return nothing
end

function chebyshev_radau_forward_transform!(chebyf, fext, ff, transform, n)
        @inbounds begin
            for j ∈ 1:n
                fext[j] = complex(ff[n-j+1],0.0)
            end
            for j ∈ 1:n-1
                fext[n+j] = fext[n-j+1]
            end
        end
        #println("ff",ff)
        #println("fext",fext)
        # perform the forward, complex-to-complex FFT in-place (cheby.fext is overwritten)
        transform*fext
        #println("fext",fext)
        # use reality + evenness of f to eliminate unncessary information
        # and obtain Chebyshev spectral coefficients for this element
        # also sort out normalisation
        @inbounds begin
            nfft = 2*n - 1
            for j ∈ 2:n
                chebyf[j] = 2.0*real(fext[j])/nfft
            end
            chebyf[1] = real(fext[1])/nfft
        end
        return nothing
    end
    
    """
    """
    function chebyshev_radau_backward_transform!(ff, fext, chebyf, transform, n)
        # chebyf as input contains Chebyshev spectral coefficients
        # need to use reality condition to extend onto negative frequency domain
        @inbounds begin
            # first, fill in values for fext corresponding to positive frequencies
            for j ∈ 2:n
                fext[j] = chebyf[j]*0.5
            end
            # next, fill in values for fext corresponding to negative frequencies
            # using fext(-k) = conjg(fext(k)) = fext(k)
            # usual FFT ordering with j=1 <-> k=0, followed by ascending k up to kmax
            # and then descending from -kmax down to -dk
            for j ∈ 1:n-1
                fext[n+j] = fext[n-j+1]
            end
            # fill in zero frequency mode, which is special in that it does not require
            # the 1/2 scale factor
            fext[1] = chebyf[1]
        end
        #println("chebyf",chebyf)
        #println("fext",fext)
        # perform the backward, complex-to-complex FFT in-place (fext is overwritten)
        transform*fext
        #println("fext",fext)
        
        @inbounds begin
            for j ∈ 1:n
                ff[j] = real(fext[n-j+1])
            end
        end
        return nothing
    end
    function chebyshev_radau_derivative_single_element!(df, ff, cheby_f, cheby_df, cheby_fext, forward, coord)
        # calculate the Chebyshev coefficients of the real-space function ff and return
        # as cheby_f
        chebyshev_radau_forward_transform!(cheby_f, cheby_fext, ff, forward, coord.ngrid)
        # calculate the Chebyshev coefficients of the derivative of ff with respect to coord.grid
        chebyshev_spectral_derivative!(cheby_df, cheby_f)
        # inverse Chebyshev transform to get df/dcoord
        chebyshev_radau_backward_transform!(df, cheby_fext, cheby_df, forward, coord.ngrid)
    end
    function chebyshev_radau_derivative_lower_endpoint(ff, cheby_f, cheby_df, cheby_fext, forward, coord)
        # calculate the Chebyshev coefficients of the real-space function ff and return
        # as cheby_f
        chebyshev_radau_forward_transform!(cheby_f, cheby_fext, ff, forward, coord.ngrid)
        # calculate the Chebyshev coefficients of the derivative of ff with respect to coord.grid
        chebyshev_spectral_derivative!(cheby_df, cheby_f)
        # form the derivative at x = - 1 using that T_n(-1) = (-1)^n
        # and converting the normalisation factors to undo the normalisation in the FFT
        # df = d0 + sum_n=1 (-1)^n d_n with d_n the coeffs
        # of the Cheb derivative in the Fourier representation
        # df = sum_n=0,N-1 d_n T_n(x)
        df = cheby_df[1]
        for i in 2:coord.ngrid
            df += ((-1)^(i-1))*cheby_df[i]
        end
        return df
    end


"""
derivative matrix for Gauss-Lobatto points using the analytical specification from 
Chapter 8.2 from Trefethen 1994 
https://people.maths.ox.ac.uk/trefethen/8all.pdf
full list of Chapters may be obtained here 
https://people.maths.ox.ac.uk/trefethen/pdetext.html
"""
    function cheb_derivative_matrix_elementwise!(D::Array{Float64,2},n::Int64)
        
        # define Gauss-Lobatto Chebyshev points in reversed order x_j = { -1, ... , 1}
        # consistent with use in elements of the grid
        x = Array{Float64,1}(undef,n)
        for j in 1:n
            x[j] = cospi((n-j)/(n-1))
        end
        
        # zero matrix before allocating values
        D[:,:] .= 0.0
        
        # top row 
        j = 1
        c_j = 2.0 
        c_k = 1.0
        for k in 2:n-1
            D[j,k] = Djk(x,j,k,c_j,c_k)
        end
        k = n 
        c_k = 2.0
        D[j,k] = Djk(x,j,k,c_j,c_k)
        
        # bottom row 
        j = n
        c_j = 2.0 
        c_k = 1.0
        for k in 2:n-1
            D[j,k] = Djk(x,j,k,c_j,c_k)
        end
        k = 1
        c_k = 2.0
        D[j,k] = Djk(x,j,k,c_j,c_k)
        
        #left column
        k = 1
        c_j = 1.0 
        c_k = 2.0
        for j in 2:n-1
            D[j,k] = Djk(x,j,k,c_j,c_k)
        end
        
        #right column
        k = n
        c_j = 1.0 
        c_k = 2.0
        for j in 2:n-1
            D[j,k] = Djk(x,j,k,c_j,c_k)
        end
        
        
        # top left, bottom right
        #D[n,n] = (2.0*(n - 1.0)^2 + 1.0)/6.0
        #D[1,1] = -(2.0*(n - 1.0)^2 + 1.0)/6.0        
        # interior rows and columns
        for j in 2:n-1
            #D[j,j] = Djj(x,j)
            for k in 2:n-1
                if j == k 
                    continue
                end
                c_k = 1.0
                c_j = 1.0
                D[j,k] = Djk(x,j,k,c_j,c_k)
            end
        end
        
        # calculate diagonal entries to guarantee that
        # D * (1, 1, ..., 1, 1) = (0, 0, ..., 0, 0)
        for j in 1:n
            D[j,j] = -sum(D[j,:])
        end
    end
    function Djk(x::Array{Float64,1},j::Int64,k::Int64,c_j::Float64,c_k::Float64)
        return  (c_j/c_k)*((-1)^(k+j))/(x[j] - x[k])
    end
 """
 Derivative matrix for Chebyshev-Radau grid using the FFT.
 Note that a similar function could be constructed for the 
 Chebyshev-Lobatto grid, if desired.
 """
    function cheb_derivative_matrix_elementwise_radau_by_FFT!(D::Array{Float64,2}, coord, f, df, fext, forward)
        ff_buffer = Array{Float64,1}(undef,coord.ngrid)
        df_buffer = Array{Float64,1}(undef,coord.ngrid)
        # use response matrix approach to calculate derivative matrix D 
        for j in 1:coord.ngrid 
            ff_buffer .= 0.0 
            ff_buffer[j] = 1.0
            @views chebyshev_radau_derivative_single_element!(df_buffer[:], ff_buffer[:],
                f[:,1], df, fext, forward, coord)
            @. D[:,j] = df_buffer[:] # assign appropriate column of derivative matrix 
        end
        # correct diagonal elements to gurantee numerical stability
        # gives D*[1.0, 1.0, ... 1.0] = [0.0, 0.0, ... 0.0]
        for j in 1:coord.ngrid
            D[j,j] = 0.0
            D[j,j] = -sum(D[j,:])
        end
    end
    
    function cheb_lower_endpoint_derivative_vector_elementwise_radau_by_FFT!(D::Array{Float64,1}, coord, f, df, fext, forward)
        ff_buffer = Array{Float64,1}(undef,coord.ngrid)
        df_buffer = Array{Float64,1}(undef,coord.ngrid)
        # use response matrix approach to calculate derivative vector D 
        for j in 1:coord.ngrid 
            ff_buffer .= 0.0 
            ff_buffer[j] = 1.0
            @views df_buffer = chebyshev_radau_derivative_lower_endpoint(ff_buffer[:],
                f[:,1], df, fext, forward, coord)
            D[j] = df_buffer # assign appropriate value of derivative vector 
        end
        # correct diagonal elements to gurantee numerical stability
        # gives D*[1.0, 1.0, ... 1.0] = [0.0, 0.0, ... 0.0]
        D[1] = 0.0
        D[1] = -sum(D[:])
    end

end
