"""
"""
module fourier

export setup_fourier_pseudospectral
export scaled_fourier_grid
export fourier_info

using FFTW
using MPI
using ..type_definitions: mk_float, mk_int
using ..array_allocation: allocate_float, allocate_complex
import ..calculus: elementwise_derivative!
import ..calculus: elementwise_indefinite_integration!
using ..communication
import ..interpolation: single_element_interpolate!
using ..moment_kinetics_structs: discretization_info

"""
Fourier pseudospectral discretization
"""
struct fourier_base_info{TForward <: FFTW.cFFTWPlan, TBackward <: AbstractFFTs.ScaledPlan}
    # fbuffer is an array for storing f(z) on the extended domain needed
    # to perform complex-to-complex FFT using the fact that f(theta) is even in theta
    fbuffer::Vector{Complex{mk_float}}
    # flag for whether we are Radau or Lobatto
    is_lobatto::Bool
    # Wavenumbers for each point on the Fourier-space grid
    # second dimension indicates the element
    k::Vector{mk_float}
    # plan for the complex-to-complex, in-place, forward Fourier transform
    forward::TForward
    # plan for the complex-to-complex, in-place, backward Fourier transform
    # backward_transform::FFTW.cFFTWPlan
    backward::TBackward
end

struct fourier_info{TForward <: FFTW.cFFTWPlan, TBackward <: AbstractFFTs.ScaledPlan} <: discretization_info
    lobatto::fourier_base_info{TForward, TBackward}
end

"""
create arrays needed for explicit Fourier pseudospectral treatment
and create the plans for the forward and backward fast Fourier transforms
"""
function setup_fourier_pseudospectral(coord, run_directory; ignore_MPI=false)
    if coord.nelement_global > 1
        error("fourier_pseudospectral discretization requires a single element. "
              * "Got nelement_global=$(coord.nelement_globa)")
    end
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
        wisdom_filename = joinpath(run_directory, "fftw_wisdom.save")
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

    lobatto = setup_fourier_pseudospectral_lobatto(coord, fftw_flags)

    if global_rank[] == 0
        if wisdom_filename !== nothing
            FFTW.export_wisdom(wisdom_filename)
        end
        this_barrier()
    end

    # Ensure root does not start modifying 'wisdom file' while other processes are still
    # reading it - root waits here for all other processes.
    this_barrier()

    return fourier_info(lobatto)
end

function setup_fourier_pseudospectral_lobatto(coord, fftw_flags)
    # This Fourier implementation only allows one element.
    # The repeated point at the periodic boundary is only included once in the Fourier
    # transform, so the internal arrays all have size ngrid-1.
    fbuffer = allocate_complex(coord.ngrid-1)
    # k values corresponding to Fourier-space grid
    k = allocate_float(coord.ngrid-1)
    k .= fftfreq(coord.ngrid-1, (coord.ngrid-1) * π)
    # setup the plans for the forward and backward Fourier transforms
    forward_transform = plan_fft!(fbuffer, flags=fftw_flags)
    backward_transform = plan_ifft!(fbuffer, flags=fftw_flags)
    # return a structure containing the information needed to carry out
    # a 1D Chebyshev transform
    return fourier_base_info(fbuffer, true, k, forward_transform, backward_transform)
end

"""
    elementwise_derivative!(coord, ff, fourier::fourier_info)

Fourier transform f to get spectral coefficients and use them to calculate f'.
"""
function elementwise_derivative!(coord, ff, fourier::fourier_info)
    df = @view coord.scratch_2d[:,1]
    # check array bounds
    @boundscheck coord.ngrid == size(df,1) || throw(BoundsError(df))

    # note that one must multiply by  1/element_scale[j] get derivative
    # in scaled coordinate on element j
    
    fbuffer = fourier.lobatto.fbuffer

    # Final point is duplicte of first point - do not include in Fourier transforms
    @views fbuffer .= ff[1:end-1]

    # perform the forward, complex-to-complex FFT in-place (fbuffer is overwritten)
    fourier.lobatto.forward * fbuffer

    # Multiply by wavenumber i*k to get the derivative
    fbuffer .*= 1im * fourier.lobatto.k

    # perform the backward transform to get the result back in real space
    fourier.lobatto.backward * fbuffer

    @views @. df[1:end-1] = real(fbuffer)
    df[end] = df[1,1]

    # and multiply by scaling factor needed to go from derivative on [-1,1] to actual
    # derivative
    df ./= coord.element_scale[1]

    return nothing
end

"""
    elementwise_derivative!(coord, ff, adv_fac, spectral::fourier_info)

Fourier transform f to get spectral coefficients and use them to calculate f'.

Note: Fourier derivative does not make use of upwinding information.
"""
function elementwise_derivative!(coord, ff, adv_fac, spectral::fourier_info)
    return elementwise_derivative!(coord, ff, spectral)
end

function single_element_interpolate!(result, newgrid, f, imin, imax, ielement, coord,
                                     fourier::fourier_base_info, derivative)
    error("Fourier discretization does not support interpolation yet.")
end

end
