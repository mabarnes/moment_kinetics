"""
`struct`s used within `moment_kinetics` that should be defined early to avoid dependency
cycles when they are used by several other modules.
"""
module moment_kinetics_structs

using ..communication
using ..type_definitions: mk_float

export ndim_pdf_ion, ndim_pdf_neutral, ndim_pdf_electron
export ndim_field, ndim_moment, ndim_moment_electron
export ndim_v, ndim_v_neutral

# variables to define the number of dimensions in arrays
const ndim_pdf_ion = 5 #(vpa + vperp + z + r + s)
const ndim_pdf_neutral = 6 #(vz + vr + vzeta + z + r + s)
const ndim_pdf_electron = 4 #(vpa + vperp + z + r)
const ndim_field = 2 #(z + r)
const ndim_gyrofield = 4 #(vperp + z + r + s)
const ndim_moment = 3 #(z + r + s)
const ndim_moment_electron = 2 #(z + r)
const ndim_v = 2 #(vpa + vperp)
const ndim_v_neutral = 3 #(vz + vr + vzeta)


"""
"""
struct scratch_pdf
    # ions
    pdf::MPISharedArray{mk_float, ndim_pdf_ion}
    density::MPISharedArray{mk_float, ndim_moment}
    upar::MPISharedArray{mk_float, ndim_moment}
    ppar::MPISharedArray{mk_float, ndim_moment}
    pperp::MPISharedArray{mk_float, ndim_moment}
    ion_external_source_controller_integral::MPISharedArray{mk_float, 3}
    temp_z_s::MPISharedArray{mk_float, ndim_moment}
    # electrons
    pdf_electron::MPISharedArray{mk_float, ndim_pdf_electron}
    electron_density::MPISharedArray{mk_float, ndim_moment_electron}
    electron_upar::MPISharedArray{mk_float, ndim_moment_electron}
    electron_ppar::MPISharedArray{mk_float, ndim_moment_electron}
    electron_pperp::MPISharedArray{mk_float, ndim_moment_electron}
    electron_temp::MPISharedArray{mk_float, ndim_moment_electron}
    #electron_external_source_controller_integral::MPISharedArray{mk_float, 3} # Not implemented yet
    # neutral particles 
    pdf_neutral::MPISharedArray{mk_float, ndim_pdf_neutral}
    density_neutral::MPISharedArray{mk_float, ndim_moment}
    uz_neutral::MPISharedArray{mk_float, ndim_moment}
    pz_neutral::MPISharedArray{mk_float, ndim_moment}
    neutral_external_source_controller_integral::MPISharedArray{mk_float, 3}
end

"""
"""
struct scratch_electron_pdf
    # electrons
    pdf_electron::MPISharedArray{mk_float, ndim_pdf_electron}
    electron_ppar::MPISharedArray{mk_float, ndim_moment_electron}
end

"""
"""
struct em_fields_struct
    # phi is the electrostatic potential
    phi::MPISharedArray{mk_float,ndim_field}
    # phi0 is the initial electrostatic potential
    phi0::MPISharedArray{mk_float,ndim_field}
    # Er is the radial electric field
    Er::MPISharedArray{mk_float,ndim_field}
    # Ez is the parallel electric field
    Ez::MPISharedArray{mk_float,ndim_field}
    # gphi is the gyroaveraged electrostatic potential
    gphi::MPISharedArray{mk_float,ndim_gyrofield}
    # gEr is the gyroaveraged radial electric field
    gEr::MPISharedArray{mk_float,ndim_gyrofield}
    # gEz is the gyroaveraged parallel electric field
    gEz::MPISharedArray{mk_float,ndim_gyrofield}
    # if true, force Er = 0 at wall plates
    force_Er_zero_at_wall::Bool
end

"""
"""
struct moments_ion_substruct{ndim_moment_wall}
    # this is the particle density
    dens::MPISharedArray{mk_float,ndim_moment}
    # flag that keeps track of if the density needs updating before use
    # Note: may not be set for all species on this process, but this process only ever
    # sets/uses the value for the same subset of species. This means dens_update does
    # not need to be a shared memory array.
    dens_updated::Vector{Bool}
    # this is the parallel flow
    upar::MPISharedArray{mk_float,ndim_moment}
    # flag that keeps track of whether or not upar needs updating before use
    # Note: may not be set for all species on this process, but this process only ever
    # sets/uses the value for the same subset of species. This means upar_update does
    # not need to be a shared memory array.
    upar_updated::Vector{Bool}
    # this is the parallel pressure
    ppar::MPISharedArray{mk_float,ndim_moment}
    # flag that keeps track of whether or not ppar needs updating before use
    # Note: may not be set for all species on this process, but this process only ever
    # sets/uses the value for the same subset of species. This means ppar_update does
    # not need to be a shared memory array.
    ppar_updated::Vector{Bool}
    # this is the perpendicular pressure
    pperp::MPISharedArray{mk_float,ndim_moment}
    # this is the parallel heat flux
    qpar::MPISharedArray{mk_float,ndim_moment}
    # flag that keeps track of whether or not qpar needs updating before use
    # Note: may not be set for all species on this process, but this process only ever
    # sets/uses the value for the same subset of species. This means qpar_update does
    # not need to be a shared memory array.
    qpar_updated::Vector{Bool}
    # this is the thermal speed based on the parallel temperature Tpar = ppar/dens: vth = sqrt(2*Tpar/m)
    vth::MPISharedArray{mk_float,ndim_moment}
    # this is the temperature, calculated from 2ppar/dens (the comment above may be out of date?)
    temp::MPISharedArray{mk_float,3}
    # generalised Chodura integrals for the lower and upper plates
    chodura_integral_lower::MPISharedArray{mk_float,ndim_moment_wall}
    chodura_integral_upper::MPISharedArray{mk_float,ndim_moment_wall}
    # if evolve_ppar = true, then the velocity variable is (vpa - upa)/vth, which introduces
    # a factor of vth for each power of wpa in velocity space integrals.
    # v_norm_fac accounts for this: it is vth if using the above definition for the parallel velocity,
    # and it is one otherwise
    v_norm_fac::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the z-derivative of the particle density
    ddens_dz::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the upwinded z-derivative of the particle density
    ddens_dz_upwind::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the second-z-derivative of the particle density
    d2dens_dz2::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the z-derivative of the parallel flow
    dupar_dz::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the upwinded z-derivative of the parallel flow
    dupar_dz_upwind::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the second-z-derivative of the parallel flow
    d2upar_dz2::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the z-derivative of the parallel pressure
    dppar_dz::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the upwinded z-derivative of the parallel pressure
    dppar_dz_upwind::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the second-z-derivative of the parallel pressure
    d2ppar_dz2::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the z-derivative of the parallel heat flux
    dqpar_dz::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the z-derivative of the thermal speed based on the parallel temperature Tpar = ppar/dens: vth = sqrt(2*Tpar/m)
    dvth_dz::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the z-derivative of the temperature
    dT_dz::Union{MPISharedArray{mk_float,3},Nothing}
    # Time derivative of the density
    ddens_dt::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # Time derivative of the parallel flow
    dupar_dt::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # Time derivative of the parallel particle flux
    dnupar_dt::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # Time derivative of the parallel pressure
    dppar_dt::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # Time derivative of the thermal speed
    dvth_dt::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the entropy production dS/dt = - int (ln f sum_s' C_ss' [f_s,f_s']) d^3 v
    dSdt::MPISharedArray{mk_float,ndim_moment}
    # Spatially varying amplitude of the external source term (third index is for different sources)
    external_source_amplitude::MPISharedArray{mk_float,ndim_moment}
    # Spatially varying amplitude of the density moment of the external source term
    external_source_density_amplitude::MPISharedArray{mk_float,ndim_moment}
    # Spatially varying amplitude of the parallel momentum moment of the external source
    # term
    external_source_momentum_amplitude::MPISharedArray{mk_float,ndim_moment}
    # Spatially varying amplitude of the parallel pressure moment of the external source
    # term
    external_source_pressure_amplitude::MPISharedArray{mk_float,ndim_moment}
    # Integral term for the PID controller of the external source term
    external_source_controller_integral::MPISharedArray{mk_float,ndim_moment}
    # Store coefficient 'A' from applying moment constraints so we can write it out as a
    # diagnostic
    constraints_A_coefficient::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # Store coefficient 'B' from applying moment constraints so we can write it out as a
    # diagnostic
    constraints_B_coefficient::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # Store coefficient 'C' from applying moment constraints so we can write it out as a
    # diagnostic
    constraints_C_coefficient::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
end

"""
moments_electron_substruct is a struct that contains moment information for electrons
"""
struct moments_electron_substruct{ndim_moment_electron_source}
    # this is the particle density
    dens::MPISharedArray{mk_float,ndim_moment_electron}
    # flag that keeps track of if the density needs updating before use
    dens_updated::Base.RefValue{Bool}
    # this is the parallel flow
    upar::MPISharedArray{mk_float,ndim_moment_electron}
    # flag that keeps track of whether or not upar needs updating before use
    upar_updated::Base.RefValue{Bool}
    # this is the parallel pressure
    ppar::MPISharedArray{mk_float,ndim_moment_electron}
    # flag that keeps track of whether or not ppar needs updating before use
    ppar_updated::Base.RefValue{Bool}
    # this is the temperature
    temp::MPISharedArray{mk_float,ndim_moment_electron}
    # flag that keeps track of whether or not temp needs updating before use
    temp_updated::Base.RefValue{Bool}
    # this is the parallel heat flux
    qpar::MPISharedArray{mk_float,ndim_moment_electron}
    # flag that keeps track of whether or not qpar needs updating before use
    qpar_updated::Base.RefValue{Bool}
    # this is the thermal speed based on the parallel temperature Tpar = ppar/dens: vth = sqrt(2*Tpar/m)
    vth::MPISharedArray{mk_float,ndim_moment_electron}
    # this is the parallel friction force between ions and electrons
    parallel_friction::MPISharedArray{mk_float,ndim_moment_electron}
    # Spatially varying amplitude of the external source term
    external_source_amplitude::MPISharedArray{mk_float,ndim_moment_electron_source}
    # Spatially varying amplitude of the density moment of the external source term
    external_source_density_amplitude::MPISharedArray{mk_float,ndim_moment_electron_source}
    # Spatially varying amplitude of the parallel momentum moment of the external source
    # term
    external_source_momentum_amplitude::MPISharedArray{mk_float,ndim_moment_electron_source}
    # Spatially varying amplitude of the parallel pressure moment of the external source
    # term
    external_source_pressure_amplitude::MPISharedArray{mk_float,ndim_moment_electron_source}
    # if evolve_ppar = true, then the velocity variable is (vpa - upa)/vth, which introduces
    # a factor of vth for each power of wpa in velocity space integrals.
    # v_norm_fac accounts for this: it is vth if using the above definition for the parallel velocity,
    # and it is one otherwise
    v_norm_fac::Union{MPISharedArray{mk_float,ndim_moment_electron},Nothing}
    # this is the z-derivative of the particle density
    ddens_dz::Union{MPISharedArray{mk_float,ndim_moment_electron},Nothing}
    # this is the z-derivative of the parallel flow
    dupar_dz::Union{MPISharedArray{mk_float,ndim_moment_electron},Nothing}
    # this is the z-derivative of the parallel pressure
    dppar_dz::Union{MPISharedArray{mk_float,ndim_moment_electron},Nothing}
    # this is the upwinded z-derivative of the parallel pressure
    dppar_dz_upwind::Union{MPISharedArray{mk_float,ndim_moment_electron},Nothing}
    # this is the second-z-derivative of the parallel pressure
    d2ppar_dz2::Union{MPISharedArray{mk_float,ndim_moment_electron},Nothing}
    # this is the z-derivative of the parallel heat flux
    dqpar_dz::Union{MPISharedArray{mk_float,ndim_moment_electron},Nothing}
    # this is the z-derivative of the parallel temperature Tpar = ppar/dens
    dT_dz::Union{MPISharedArray{mk_float,ndim_moment_electron},Nothing}
    # this is the upwinded z-derivative of the temperature Tpar = ppar/dens
    dT_dz_upwind::Union{MPISharedArray{mk_float,ndim_moment_electron},Nothing}
    # this is the z-derivative of the electron thermal speed vth = sqrt(2*Tpar/m)
    dvth_dz::Union{MPISharedArray{mk_float,ndim_moment_electron},Nothing}
    # Time derivative of the parallel pressure
    dppar_dt::Union{MPISharedArray{mk_float,ndim_moment_electron},Nothing}
    # Time derivative of the parallel temperature
    dTpar_dt::Union{MPISharedArray{mk_float,ndim_moment_electron},Nothing}
    # Time derivative of the thermal speed
    dvth_dt::Union{MPISharedArray{mk_float,ndim_moment_electron},Nothing}
    # Store coefficient 'A' from applying moment constraints so we can write it out as a
    # diagnostic
    constraints_A_coefficient::Union{MPISharedArray{mk_float,ndim_moment_electron},Nothing}
    # Store coefficient 'B' from applying moment constraints so we can write it out as a
    # diagnostic
    constraints_B_coefficient::Union{MPISharedArray{mk_float,ndim_moment_electron},Nothing}
    # Store coefficient 'C' from applying moment constraints so we can write it out as a
    # diagnostic
    constraints_C_coefficient::Union{MPISharedArray{mk_float,ndim_moment_electron},Nothing}
end

"""
"""
struct moments_neutral_substruct
    # this is the particle density
    dens::MPISharedArray{mk_float,ndim_moment}
    # flag that keeps track of if the density needs updating before use
    # Note: may not be set for all species on this process, but this process only ever
    # sets/uses the value for the same subset of species. This means dens_update does
    # not need to be a shared memory array.
    dens_updated::Vector{Bool}
    # this is the particle mean velocity in z
    uz::MPISharedArray{mk_float,ndim_moment}
    # flag that keeps track of if uz needs updating before use
    uz_updated::Vector{Bool}
    # this is the particle mean velocity in r
    ur::MPISharedArray{mk_float,ndim_moment}
    # flag that keeps track of if ur needs updating before use
    ur_updated::Vector{Bool}
    # this is the particle mean velocity in zeta
    uzeta::MPISharedArray{mk_float,ndim_moment}
    # flag that keeps track of if uzeta needs updating before use
    uzeta_updated::Vector{Bool}
    # this is the zz particle pressure tensor component
    pz::MPISharedArray{mk_float,ndim_moment}
    # flag that keeps track of if pz needs updating before use
    pz_updated::Vector{Bool}
    # this is the rr particle pressure tensor component
    pr::MPISharedArray{mk_float,ndim_moment}
    # flag that keeps track of if pr needs updating before use
    pr_updated::Vector{Bool}
    # this is the zetazeta particle pressure tensor component
    pzeta::MPISharedArray{mk_float,ndim_moment}
    # flag that keeps track of if pzeta needs updating before use
    pzeta_updated::Vector{Bool}
    # this is the total (isotropic) particle pressure
    ptot::MPISharedArray{mk_float,ndim_moment}
    # this is the heat flux along z
    qz::MPISharedArray{mk_float,ndim_moment}
    # flag that keeps track of if qz needs updating before use
    qz_updated::Vector{Bool}
    # this is the thermal speed based on the temperature T = ptot/dens: vth = sqrt(2*T/m)
    vth::MPISharedArray{mk_float,ndim_moment}
    # if evolve_ppar = true, then the velocity variable is (vz - uz)/vth, which introduces
    # a factor of vth for each power of wz in velocity space integrals.
    # v_norm_fac accounts for this: it is vth if using the above definition for the parallel velocity,
    # and it is one otherwise
    v_norm_fac::MPISharedArray{mk_float,ndim_moment}
    # this is the z-derivative of the particle density
    ddens_dz::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the z-derivative of the particle density
    ddens_dz_upwind::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the second-z-derivative of the particle density
    d2dens_dz2::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the z-derivative of the particle mean velocity in z
    duz_dz::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the upwinded z-derivative of the particle mean velocity in z
    duz_dz_upwind::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the second-z-derivative of the particle mean velocity in z
    d2uz_dz2::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the z-derivative of the zz particle pressure tensor component
    dpz_dz::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the upwinded z-derivative of the zz particle pressure tensor component
    dpz_dz_upwind::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the second-z-derivative of the zz particle pressure tensor component
    d2pz_dz2::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the z-derivative of the thermal speed based on the temperature T = ptot/dens: vth = sqrt(2*T/m)
    dvth_dz::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # this is the z-derivative of the heat flux along z
    dqz_dz::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # Time derivative of the density
    ddens_dt::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # Time derivative of the particle mean velocity in z
    duz_dt::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # Time derivative of the particle mean flux in z
    dnuz_dt::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # Time derivative of the zz particle pressure tensor component
    dpz_dt::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # Time derivative of the thermal speed
    dvth_dt::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # Spatially varying amplitude of the external source term
    external_source_amplitude::MPISharedArray{mk_float,ndim_moment}
    # Spatially varying amplitude of the density moment of the external source term
    external_source_density_amplitude::MPISharedArray{mk_float,ndim_moment}
    # Spatially varying amplitude of the parallel momentum moment of the external source
    # term
    external_source_momentum_amplitude::MPISharedArray{mk_float,ndim_moment}
    # Spatially varying amplitude of the parallel pressure moment of the external source
    # term
    external_source_pressure_amplitude::MPISharedArray{mk_float,ndim_moment}
    # Integral term for the PID controller of the external source term
    external_source_controller_integral::MPISharedArray{mk_float,ndim_moment}
    # Store coefficient 'A' from applying moment constraints so we can write it out as a
    # diagnostic
    constraints_A_coefficient::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # Store coefficient 'B' from applying moment constraints so we can write it out as a
    # diagnostic
    constraints_B_coefficient::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
    # Store coefficient 'C' from applying moment constraints so we can write it out as a
    # diagnostic
    constraints_C_coefficient::Union{MPISharedArray{mk_float,ndim_moment},Nothing}
end

"""
"""
struct pdf_substruct{ndim_distribution}
    norm::MPISharedArray{mk_float,ndim_distribution}
    buffer::MPISharedArray{mk_float,ndim_distribution} # for collision operator terms when pdfs must be interpolated onto different velocity space grids, and for gyroaveraging
end

"""
"""
struct electron_pdf_substruct
    norm::MPISharedArray{mk_float,ndim_pdf_electron}
    buffer::MPISharedArray{mk_float,ndim_pdf_electron} # for collision operator terms when pdfs must be interpolated onto different velocity space grids
    pdf_before_ion_timestep::MPISharedArray{mk_float,ndim_pdf_electron}
end

# struct of structs neatly contains i+n info?
"""
"""
struct pdf_struct
    #ion particles: s + r + z + vperp + vpa
    ion::pdf_substruct{ndim_pdf_ion}
    # electron particles: r + z + vperp + vpa
    electron::Union{electron_pdf_substruct,Nothing}
    #neutral particles: s + r + z + vzeta + vr + vz
    neutral::pdf_substruct{ndim_pdf_neutral}
end

"""
"""
struct moments_struct{ndim_moment_wall, ndim_moment_electron_source}
    ion::moments_ion_substruct{ndim_moment_wall}
    electron::moments_electron_substruct{ndim_moment_electron_source}
    neutral::moments_neutral_substruct
    # flag that indicates if the density should be evolved via continuity equation
    evolve_density::Bool
    # flag that indicates if particle number should be conserved for each species
    # effects like ionisation or net particle flux from the domain would lead to
    # non-conservation
    particle_number_conserved::Bool
    # flag that indicates if exact particle conservation should be enforced
    enforce_conservation::Bool
    # flag that indicates if the parallel flow should be evolved via force balance
    evolve_upar::Bool
    # flag that indicates if the parallel pressure should be evolved via the energy equation
    evolve_ppar::Bool
end

"""
"""
struct boundary_distributions_struct
    # knudsen cosine distribution for imposing the neutral wall boundary condition
    knudsen::MPISharedArray{mk_float,ndim_v_neutral}
    # ion particle r boundary values (vpa,vperp,z,r,s)
    pdf_rboundary_ion::MPISharedArray{mk_float,ndim_pdf_ion}
    # neutral particle r boundary values (vz,vr,vzeta,z,r,s)
    pdf_rboundary_neutral::MPISharedArray{mk_float,ndim_pdf_neutral}
end

"""
discretization_info for one dimension

All the specific discretizations in moment_kinetics are subtypes of this type.
"""
abstract type discretization_info end

"""
discretization_info for a discretization that supports 'weak form' methods, for one
dimension
"""
abstract type weak_discretization_info <: discretization_info end

"""
Type representing a spatial dimension with only one grid point
"""
struct null_spatial_dimension_info <: discretization_info end

"""
Type representing a velocity space dimension with only one grid point
"""
struct null_velocity_dimension_info <: discretization_info end

end
