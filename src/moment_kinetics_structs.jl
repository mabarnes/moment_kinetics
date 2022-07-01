"""
`struct`s used within `moment_kinetics` that should be defined early to avoid dependency
cycles when they are used by several other modules.
"""
module moment_kinetics_structs

using FFTW
using ..communication
using ..type_definitions: mk_float

"""
"""
struct scratch_pdf{n_distribution, n_moment}
    pdf::MPISharedArray{mk_float, n_distribution}
    density::MPISharedArray{mk_float, n_moment}
    upar::MPISharedArray{mk_float, n_moment}
    ppar::MPISharedArray{mk_float, n_moment}
    temp_z_s::MPISharedArray{mk_float, n_moment}
end

"""
"""
struct em_fields_struct
    # phi is the electrostatic potential
    phi::MPISharedArray{mk_float,2}
    # phi0 is the initial electrostatic potential
    phi0::MPISharedArray{mk_float,2}
    # if including an external forcing for phi, it is of the form
    # phi_external = phi0*drive_amplitude*sinpi(t*drive_frequency)
    force_phi::Bool
    drive_amplitude::mk_float
    drive_frequency::mk_float
end

"""
"""
struct chebyshev_info{TForward <: FFTW.cFFTWPlan, TBackward <: AbstractFFTs.ScaledPlan}
    # fext is an array for storing f(z) on the extended domain needed
    # to perform complex-to-complex FFT using the fact that f(theta) is even in theta
    fext::Array{Complex{mk_float},1}
    # Chebyshev spectral coefficients of distribution function f
    # first dimension contains location within element
    # second dimension indicates the element
    f::Array{mk_float,2}
    # Chebyshev spectral coefficients of derivative of f
    df::Array{mk_float,1}
    # plan for the complex-to-complex, in-place, forward Fourier transform on Chebyshev-Gauss-Lobatto grid
    forward::TForward
    # plan for the complex-to-complex, in-place, backward Fourier transform on Chebyshev-Gauss-Lobatto grid
    #backward_transform::FFTW.cFFTWPlan
    backward::TBackward
end

end
