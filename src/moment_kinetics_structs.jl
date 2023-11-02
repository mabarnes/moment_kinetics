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
struct scratch_pdf{n_distribution_charged, n_moment, n_distribution_neutral,n_moment_neutral}
    # charged particles
    pdf::MPISharedArray{mk_float, n_distribution_charged}
    density::MPISharedArray{mk_float, n_moment}
    upar::MPISharedArray{mk_float, n_moment}
    ppar::MPISharedArray{mk_float, n_moment}
    pperp::MPISharedArray{mk_float, n_moment}
    temp_z_s::MPISharedArray{mk_float, n_moment}
    # neutral particles 
    pdf_neutral::MPISharedArray{mk_float, n_distribution_neutral}
    density_neutral::MPISharedArray{mk_float, n_moment_neutral}
    uz_neutral::MPISharedArray{mk_float, n_moment_neutral}
    pz_neutral::MPISharedArray{mk_float, n_moment_neutral}
end

"""
"""
struct em_fields_struct
    # phi is the electrostatic potential
    phi::MPISharedArray{mk_float,2}
    # phi0 is the initial electrostatic potential
    phi0::MPISharedArray{mk_float,2}
    # Er is the radial electric field
    Er::MPISharedArray{mk_float,2}
    # Ez is the parallel electric field
    Ez::MPISharedArray{mk_float,2}
    # if including an external forcing for phi, it is of the form
    # phi_external = phi0*drive_amplitude*sinpi(t*drive_frequency)
    force_phi::Bool
    drive_amplitude::mk_float
    drive_frequency::mk_float
    # if true, force Er = 0 at wall plates
    force_Er_zero_at_wall::Bool
end

end
