"""
"""
module fourier

export setup_fourier_pseudospectral
export scaled_fourier_grid
export fourier_spectral_derivative!
export fourier_info

using FFTW
using MPI
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float, allocate_complex
import ..calculus: elementwise_derivative!
using ..communication
using ..moment_kinetics_structs: discretization_info

"""
Fourier pseudospectral discretization
"""
struct fourier_info{TForward <: FFTW.cFFTWPlan, TBackward <: AbstractFFTs.ScaledPlan}
    # fext is an array for storing f(z) on the extended domain needed
    # to perform complex-to-complex FFT using the fact that f(theta) is even in theta
    fext::Array{Complex{mk_float},1}
    # Chebyshev spectral coefficients of distribution function f
    # first dimension contains location within element
    # second dimension indicates the element
    f::Array{Complex{mk_float},2}
    # Chebyshev spectral coefficients of derivative of f
    df::Array{Complex{mk_float},1}
    # plan for the complex-to-complex, in-place, forward Fourier transform on Chebyshev-Gauss-Lobatto/Radau grid
    forward::TForward
    # plan for the complex-to-complex, in-place, backward Fourier transform on Chebyshev-Gauss-Lobatto/Radau grid
    # backward_transform::FFTW.cFFTWPlan
    backward::TBackward
    # midpoint integer, highest integer of postive physical wavenumbers
    imidm::mk_int
    imidp::mk_int
end


"""
create arrays needed for explicit Fourier pseudospectral treatment
and create the plans for the forward and backward fast Fourier transforms
"""
function setup_fourier_pseudospectral(coord, run_directory; ignore_MPI=false)
    # First set up the FFTW plans on the (global) root process, then save the 'FFTW
    # wisdom' and load it on all other processes, to ensure that we use the exact same
    # FFT algorithms on all processes for consistency.
    if run_directory === nothing
        if global_size[] != 1 && !ignore_MPI
            error("run_directory is required by setup_fourier_pseudospectral() when "
                  * "running in parallel, in order to save FFTW wisdom.")
        end
        wisdom_filename = nothing
    else
        wisdom_filename = joinpath(run_directory, "fftw_wisdom_fourier.save")
    end

    # When using FFTW.WISDOM_ONLY, the flag should be combined with the flag that was
    # originally used to generate the 'wisdom' otherwise if the original flag was 'lower
    # effort' (i.e. was FFTW.ESTIMATE) then the default (FFTW.MEASURE) will be used
    # instead. 
    base_flag = FFTW.MEASURE

    function this_barrier()
        if !ignore_MPI
            # Normal case, all processors are creating the coordinate
            MPI.Barrier(comm_world)
        elseif run_directory !== nothing && comm_inter_block[] != MPI.COMM_NULL
            # ignore_MPI=true was passed, but non-null communicator exists. This happens
            # in calls from load_restart_coordinates(), which is only called on
            # block_rank[]==0.
            MPI.Barrier(comm_inter_block[])
        else
            # Should be serial (e.g. used in post-processing), so no Barrier
        end
    end

    if global_rank[] != 0
        # Wait for rank-0
        this_barrier()
        if wisdom_filename !== nothing
            # Load wisdom
            FFTW.import_wisdom(wisdom_filename)
            # Flags can be combined with a bitwise-or operation `|`.
            fftw_flags = base_flag | FFTW.WISDOM_ONLY
        else
            fftw_flags = base_flag
        end
    else
        fftw_flags = base_flag
    end

    fourier_spectral = setup_fourier_pseudospectral_grids(coord, fftw_flags)

    if global_rank[] == 0
        if wisdom_filename !== nothing
            FFTW.export_wisdom(wisdom_filename)
        end
        this_barrier()
    end

    # Ensure root does not start modifying 'wisdom file' while other processes are still
    # reading it - root waits here for all other processes.
    this_barrier()

    return fourier_spectral
end

function setup_fourier_pseudospectral_grids(coord, fftw_flags)
    ngrid_fft = coord.ngrid
    # create array for f on extended [0,2π] domain in theta = ArcCos[z]
    fext = allocate_complex(ngrid_fft)
    # create arrays for storing spectral coefficients of f and f'
    fhat = allocate_complex(coord.ngrid, coord.nelement_local)
    dhat = allocate_complex(coord.ngrid)
    # setup the plans for the forward and backward Fourier transforms
    forward_transform = plan_fft!(fext, flags=fftw_flags)
    backward_transform = plan_ifft!(fext, flags=fftw_flags)
    if mod(ngrid_fft,2) == 0
      imidm = mk_int((ngrid_fft/2) - 1 + 1)
      imidp = mk_int((ngrid_fft/2) - 1 + 1)
    else
      imidm = mk_int(((ngrid_fft - 1)/2))
      imidp = mk_int(((ngrid_fft - 1)/2) + 1)
    end    
    # return a structure containing the information needed to carry out
    # a 1D Fourier transform
    return fourier_info(fext, fhat, dhat, forward_transform, backward_transform, imidm, imidp)
end


"""
initialize uniform Fourier grid scaled to interval [-box_length/2, box_length/2]
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
function scaled_fourier_grid(ngrid, nelement_local, n,
			element_scale, element_shift, imin, imax)
    # this uniform grid goes from -1 to 1(-)
    uniform_grid = uniformpoints(ngrid)
    # create array for the full grid
    grid = allocate_float(n)
    
    if nelement_local > 1
       error("Fourier grids should have nelement = 1")
    end
    @inbounds for j ∈ 1:1
        scale_factor = element_scale[j]
        shift = element_shift[j]
        # apply the scale factor and shift
        grid[imin[j]:imax[j]] .= (uniform_grid[1:ngrid] * scale_factor) .+ shift

    end
    # get the weights
    uniform_wgts = allocate_float(n)
    @. uniform_wgts = 2/n
    wgts = uniform_wgts*element_scale[1]
    
    return grid, wgts
end

"""
    elementwise_derivative!(coord, ff, fourier::fourier_info)

Fourier transform f to get Fourier spectral coefficients and use them to calculate f'.
"""
function elementwise_derivative!(coord, ff, fourier::fourier_info)
   df = coord.scratch_2d
   # define local variable nelement for convenience
   nelement = coord.nelement_local
   # check array bounds
   @boundscheck nelement == size(df,2) && coord.ngrid == size(df,1) || throw(BoundsError(df))
   # note that one must multiply by a coordinate transform factor 1/element_scale[j]
   # for each element j to get derivative on the extended grid
    
   # note that one must multiply by  1/element_scale[j] get derivative
   # in scaled coordinate on element j
  
   j = 1 # the first and only element
   imin = coord.imin[j]
   # imax is the maximum index on the full grid for this (jth) element
   imax = coord.imax[j]
   #println("elementwise_derivative!: ",ff)
   @views fourier_derivative_single_element!(df[:,j], ff[imin:imax],
       fourier.f[:,j], fourier.df, fourier.fext, fourier.forward, fourier.backward, fourier.imidm, fourier.imidp, coord)
   # and multiply by scaling factor needed to go
   # from Fourier z coordinate to actual z
   for i ∈ 1:coord.ngrid
       df[i,j] /= 2.0*coord.element_scale[j]
   end
  
   return nothing
end

"""
    elementwise_derivative!(coord, ff, adv_fac, spectral::chebyshev_info)

Fourier transform f to get Chebyshev spectral coefficients and use them to calculate f'.

Note: Fourier derivative does not make use of upwinding information. This function 
is only provided for compatibility
"""
function elementwise_derivative!(coord, ff, adv_fac, spectral::fourier_info)
    return elementwise_derivative!(coord, ff, spectral)
end

"""
"""
function fourier_derivative_single_element!(df, ff, fhat, dfhat, fext,
        forward, backward, imidm, imidp, coord)
    # calculate the fourier coefficients of the real-space function ff and return
    fourier_forward_transform!(fhat, fext, ff, forward, imidm, imidp, coord.ngrid)
    # calculate the fourier coefficients of the derivative of ff with respect to coord.grid
    fourier_spectral_derivative!(dfhat, fhat, imidm, imidp)
    # inverse fourier transform to get df/dcoord
    fourier_backward_transform!(df, fext, dfhat, backward, imidm, imidp, coord.ngrid)
end

"""
use Fourier basis to compute the first derivative of f
"""
function fourier_spectral_derivative!(df,f,imidm,imidp)
    m = length(f)
    @boundscheck m == length(df) || throw(BoundsError(df))
    @inbounds begin
        for i in 1:imidm
            df[i] = 2*pi*im*(i-1)*f[i]
        end
        for i in imidm+1:m
            df[i] = 2*pi*im*((i-1)-m)*f[i]
        end
    end
end

"""
returns the uniform grid points [-1,1) on an n point grid
"""
function uniformpoints(n)
    grid = allocate_float(n)
    nfac = 1/n
    @inbounds begin
        for j ∈ 1:n
            grid[j] = -1.0 + 2.0*(j-1)*nfac
        end
    end
    return grid
end


"""
"""
function fourier_forward_transform!(fhat, fext, ff, transform, imidm, imidp, n)
    # put ff into fft order
    @inbounds begin
        # first, fill in values for f into complex array function
        for j in 1:imidm
            fext[j] = complex(ff[j+imidp],0.0)
        end
        for j in imidm+1:n
            fext[j] = complex(ff[j-imidm],0.0) 
        end
    end
    # perform the forward, complex-to-complex FFT in-place (cheby.fext is overwritten)
    transform*fext
    # set out data
    fhat .= fext
    return nothing
end

"""
"""
function fourier_backward_transform!(ff, fext, fhat, transform, imidm, imidp, n)
    fext .= fhat
    # perform the backward, complex-to-complex FFT in-place (fext is overwritten)
    transform*fext
    @inbounds begin
        # fill in entries for ff
        # put ff out of  fft order
       @inbounds begin
           # fill in values for f into complex array function
           # normalisation appears to be handled by the transform
           for j in 1:imidp
               ff[j] = real(fext[j+imidm])
           end
           for j in imidp+1:n
               ff[j] = real(fext[j-imidp])
           end
       end
    end
    return nothing
end

    



end
