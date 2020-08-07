module chebyshev

using FastTransforms: chebyshevpoints, clenshawcurtisweights
using FastTransforms: ChebyshevTransformPlan, IChebyshevTransformPlan
using FastTransforms: plan_chebyshevtransform, plan_ichebyshevtransform
using array_allocation: allocate_float

export update_fcheby
export setup_chebyshev_pseudospectral
export scaled_chebyshev_grid
export chebyshev_spectral_derivative!

struct chebyshev_info
    # Chebyshev spectral coefficients of distribution function f
    f::Array{Float64,2}
    # Chebyshev spectral coefficients of derivative of f
    df::Array{Float64,1}
    # plan for the forward Chebyshev transform on Chebyshev Gauss Lobatto grid
    forward::ChebyshevTransformPlan
    # plan for the backward Chebyshev transform on Chebyshev Gauss Lobatto grid
    backward::IChebyshevTransformPlan
end
# create arrays needed for explicit Chebyshev pseudospectral treatment
# and create the plans for the forward and backward fast Chebyshev transforms
function setup_chebyshev_pseudospectral(f, coord)
    # create arrays for storing Chebyshev spectral coefficients of f and f'
    fcheby = allocate_float(coord.ngrid, coord.nelement)
    dcheby = allocate_float(coord.ngrid)
    # setup the plans for the forward and backward Chebyshev transforms
    forward_transform = plan_chebyshevtransform(f,kind=2)
    #backward_transform = plan_ichebyshevtransform(fcheby[:,1],kind=2)
    backward_transform = plan_ichebyshevtransform(dcheby,kind=2)
    # return a structure containing the information needed to carry out
    # a 1D Chebyshev transform
    return chebyshev_info(fcheby, dcheby, forward_transform, backward_transform)
end
# initialize chebyshev grid scaled to interval [-box_length/2, box_length/2]
function scaled_chebyshev_grid(ngrid, nelement, n, box_length, imin, imax)
    # initialize chebyshev grid defined on [1,-1]
    # with n grid points chosen to facilitate
    # the fast Chebyshev transform (aka the discrete cosine transform)
    # needed to obtain Chebyshev spectral coefficients
    # this grid goes from ~ +1 to ~ -1
    chebyshev_grid = chebyshevpoints(ngrid,kind=2)
    # create array for the full grid and associated weights
    grid = allocate_float(n)
    # setup the scale factor by which the Chebyshev grid on [-1,1]
    # is to be multiplied to account for the full domain [-L/2,L/2]
    # and the splitting into nelement elements with ngrid grid points
    scale_factor = 0.5*box_length/nelement
    # account for the fact that the minimum index needed for the chebyshev_grid
    # within each element changes from 1 to 2 in going from the first element
    # to the remaining elements
    k = 1
    @inbounds for j ∈ 1:nelement
        #wgts[imin[j]:imax[j]] .= sqrt.(1.0 .- reverse(chebyshev_grid)[k:ngrid].^2) * scale_factor
        # amount by which to shift the centre of this element from zero
        shift = box_length*((j-0.5)/nelement - 0.5)
        # reverse the order of the original chebyshev_grid (ran from [1,-1])
        # and apply the scale factor and shift
        grid[imin[j]:imax[j]] .= (reverse(chebyshev_grid)[k:ngrid] * scale_factor) .+ shift
        # after first element, increase minimum index for chebyshev_grid to 2
        # to avoid double-counting boundary element
        k = 2
    end
    wgts = clenshaw_curtis_weights(ngrid, nelement, n, imin, imax, scale_factor)
    return grid, wgts
end
# Chebyshev transform f to get Chebyshev spectral coefficients
function update_fcheby!(cheby, ff, coord)
    k = 0
    @inbounds for j ∈ 1:coord.nelement
        imin = coord.imin[j]-k
        imax = coord.imax[j]
        #cheby.f[:,j] .= cheby.forward*reverse(ff)[imin:imax]
        cheby.f[:,j] .= cheby.forward*reverse(ff[imin:imax])
        k = 1
    end
    return nothing
end
# compute the Chebyshev spectral coefficients of the spatial derivative of f
function update_df_chebyshev!(df, chebyshev, coord)
    ngrid = coord.ngrid
    nelement = coord.nelement
    L = coord.L
    @boundscheck nelement == size(chebyshev.f,2) || throw(BoundsError(chebyshev.f))
    # obtain Chebyshev spectral coefficients of f'[z]
    # note that must multiply by 2/Lz to get derivative
    # in scaled coordinate
    k = 1
    @inbounds for j ∈ 1:nelement
        chebyshev_spectral_derivative!(chebyshev.df,chebyshev.f[:,j])
        # inverse Chebyshev transform to get df/dz
        # and multiply by scaling factor needed to go
        # from Chebyshev z coordinate to actual z
        imin = coord.imin[j]
        imax = coord.imax[j]
        df[imin:imax] .= 2*nelement*reverse(chebyshev.backward*chebyshev.df)[k:ngrid]/L
        #df[imin:imax] .= 2*nelement*(chebyshev.backward*chebyshev.df)[k:ngrid]/L
        #df[imin:imax] .= reverse(chebyshev.backward*chebyshev.f[:,j])[k:ngrid]
        #df[imin:imax] .= (chebyshev.backward*chebyshev.f[:,j])[k:ngrid]
        k = 2
    end
    #df .= reverse(df)
    return nothing
end
#=
# compute the Chebyshev spectral coefficients of the spatial derivative of f
function update_df_chebyshev!(df, fcheb, dfcheb, backward, ngrid, nelement, L, imin, imax, to)
    @timeit to "@boundscheck" @boundscheck nelement == size(fcheb,2) || throw(BoundsError(fcheb))
    # obtain Chebyshev spectral coefficients of f'[z]
    # note that must multiply by 2/Lz to get derivative
    # in scaled coordinate
    k = 1
    tmp = 2*nelement/L
    dum = allocate_float(ngrid)
    @inbounds for j ∈ 1:nelement
        @timeit to "chebyshev_spectral_derivative!" chebyshev_spectral_derivative!(dfcheb,fcheb[:,j])
        # inverse Chebyshev transform to get df/dz
        # and multiply by scaling factor needed to go
        # from Chebyshev z coordinate to actual z
        #@timeit to "df" df[imin[j]:imax[j]] .= 2*nelement*reverse(backward*dfcheb)[k:ngrid]/L
        #@timeit to "df" df[imin[j]:imax[j]] .= tmp*reverse(backward*dfcheb)[k:ngrid]
        @timeit to "transform" dum = reverse(backward*dfcheb)
        for i ∈ imin[j]:imax[j]
            @timeit to "ii" ii = i-imin[j]+k
            @timeit to "df" df[i] = 2*nelement*dum[ii]/L
        end
        #df[imin:imax] .= 2*nelement*(chebyshev.backward*chebyshev.df)[k:ngrid]/L
        #df[imin:imax] .= reverse(chebyshev.backward*chebyshev.f[:,j])[k:ngrid]
        #df[imin:imax] .= (chebyshev.backward*chebyshev.f[:,j])[k:ngrid]
        k = 2
    end
    #df .= reverse(df)
    return nothing
end
=#
# use Chebyshev basis to compute the derivative of f
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
# returns wgts array containing the integration weights associated
# with all grid points for Clenshaw-Curtis quadrature
function clenshaw_curtis_weights(ngrid, nelement, n, imin, imax, scale_factor)
    # create array containing the integration weights
    wgts = zeros(n)
    # calculate the modified Chebshev moments of the first kind
    μ = chebyshevmoments(ngrid)
    @inbounds begin
        # calculate the weights within a single element and
        # scale to account for modified domain (not [-1,1])
        wgts[1:ngrid] = clenshawcurtisweights(μ)*scale_factor
        if nelement > 1
            # account for double-counting of points at inner element boundaries
            wgts[ngrid] *= 2
            for j ∈ 2:nelement
                wgts[imin[j]:imax[j]] .= wgts[2:ngrid]
            end
            # remove double-counting of outer element boundary for last element
            wgts[n] *= 0.5
        end
    end
    return wgts
end
# compute and return modified Chebyshev moments of the first kind:
# ∫dx Tᵢ(x) over range [-1,1]
function chebyshevmoments(N)
    μ = zeros(N)
    @inbounds for i = 0:2:N-1
        μ[i+1] = 2/(1-i^2)
    end
    return μ
end

end
