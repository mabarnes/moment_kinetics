"""
"""
module em_fields

export setup_em_fields
export update_phi!

using ..type_definitions: mk_float
using ..array_allocation: allocate_shared_float
using ..communication: _block_synchronize
using ..input_structs
using ..looping
using ..moment_kinetics_structs: em_fields_struct
using ..velocity_moments: update_density!
#using ..calculus: derivative!
using ..derivatives: derivative_r!, derivative_z!
using ..gyroaverages: gyro_operators, gyroaverage_field!
"""
"""
function setup_em_fields(nvperp, nz, nr, n_ion_species, force_phi, drive_amplitude, drive_frequency, force_Er_zero)
    phi = allocate_shared_float(nz,nr)
    phi0 = allocate_shared_float(nz,nr)
    Er = allocate_shared_float(nz,nr)
    Ez = allocate_shared_float(nz,nr)
    gphi = allocate_shared_float(nvperp,nz,nr,n_ion_species)
    gEr = allocate_shared_float(nvperp,nz,nr,n_ion_species)
    gEz = allocate_shared_float(nvperp,nz,nr,n_ion_species)
    return em_fields_struct(phi, phi0, Er, Ez, gphi, gEr, gEz, force_phi, drive_amplitude, drive_frequency, force_Er_zero)
end

"""
update_phi updates the electrostatic potential, phi
"""
function update_phi!(fields, fvec, vperp, z, r, composition, geometry, z_spectral, r_spectral, scratch_dummy, gyroavs::gyro_operators)
    n_ion_species = composition.n_ion_species
    @boundscheck size(fields.phi,1) == z.n || throw(BoundsError(fields.phi))
    @boundscheck size(fields.phi,2) == r.n || throw(BoundsError(fields.phi))
    @boundscheck size(fields.phi0,1) == z.n || throw(BoundsError(fields.phi0))
    @boundscheck size(fields.phi0,2) == r.n || throw(BoundsError(fields.phi0))
    @boundscheck size(fields.Er,1) == z.n || throw(BoundsError(fields.Er))
    @boundscheck size(fields.Er,2) == r.n || throw(BoundsError(fields.Er))
    @boundscheck size(fields.Ez,1) == z.n || throw(BoundsError(fields.Ez))
    @boundscheck size(fields.Ez,2) == r.n || throw(BoundsError(fields.Ez))
    @boundscheck size(fvec.density,1) == z.n || throw(BoundsError(fvec.density))
    @boundscheck size(fvec.density,2) == r.n || throw(BoundsError(fvec.density))
    @boundscheck size(fvec.density,3) == composition.n_ion_species || throw(BoundsError(fvec.density))
    # Update phi using the set of processes that handles the first ion species
    # Means we get at least some parallelism, even though we have to sum
    # over species, and reduces number of _block_synchronize() calls needed
    # when there is only one species.
    
    begin_serial_region()#(no_synchronize=true)
    # in serial as both s, r and z required locally
    if (composition.n_ion_species > 1 || true ||
        composition.electron_physics == boltzmann_electron_response_with_simple_sheath)
        # If there is more than 1 ion species, the ranks that handle species 1 have to
        # read density for all the other species, so need to synchronize here.
        # If composition.electron_physics ==
        # boltzmann_electron_response_with_simple_sheath, all ranks need to read
        # fvec.density at iz=1, so need to synchronize here.
        # Use _block_synchronize() directly because we stay in a z_s type region, even
        # though synchronization is needed here.
        _block_synchronize()
    end
    #@loop_r ir begin # radial locations uncoupled so perform boltzmann solve 
                           # for each radial position in parallel if possible 
        
    # TODO: parallelise this...
    @serial_region begin
        ne = @view scratch_dummy.dummy_zrs[:,:,1]
        jpar_i = @view scratch_dummy.buffer_rs_1[:,:,1]
        N_e = @view scratch_dummy.buffer_rs_2[:,:,1]
        ne .= 0.0
        for is in 1:n_ion_species
            @loop_r_z ir iz begin
                 ne[iz,ir] += fvec.density[iz,ir,is]
            end
        end
        if composition.electron_physics == boltzmann_electron_response
            N_e .= 1.0
        elseif composition.electron_physics == boltzmann_electron_response_with_simple_sheath
            # calculate Sum_{i} Z_i n_i u_i = J_||i at z = 0
            jpar_i .= 0.0
            for is in 1:n_ion_species
                @loop_r ir begin
                     jpar_i[ir] +=  fvec.density[1,ir,is]*fvec.upar[1,ir,is]
                end
            end
            # Calculate N_e using J_||e at sheath entrance at z = 0 (lower boundary).
            # Assuming pdf is a half maxwellian with boltzmann factor at wall, we have
            # J_||e = e N_e v_{th,e} exp[ e phi_wall / T_e ] / 2 sqrt{pi},
            # where positive sign above (and negative sign below)
            # is due to the fact that electrons reaching the wall flow towards more negative z.
            # Using J_||e + J_||i = 0, and rearranging for N_e, we have
            #N_e = - 2.0 * sqrt( pi * composition.me_over_mi) * jpar_i * exp( - composition.phi_wall / composition.T_e)
            @loop_r ir begin
                 N_e[ir] = - 2.0 * sqrt( pi * composition.me_over_mi) * jpar_i[ir] * exp( - composition.phi_wall / composition.T_e)
                 # N_e must be positive, so force this in case a numerical error or something
                 # made jpar_i negative
                 N_e[ir] = max(N_e[ir], 1.e-16)
            end
            # See P.C. Stangeby, The Plasma Boundary of Magnetic Fusion Devices, IOP Publishing, Chpt 2, p75
        end
        if composition.electron_physics ∈ (boltzmann_electron_response, boltzmann_electron_response_with_simple_sheath)
            @loop_r_z ir iz begin
                 fields.phi[iz,ir] = composition.T_e * log(ne[iz,ir] / N_e[ir])
            end
        end
    end
    ## can calculate phi at z = L and hence phi_wall(z=L) using jpar_i at z =L if needed
    _block_synchronize()

    ## calculate the electric fields after obtaining phi
    #Er = - d phi / dr 
    if r.n > 1
        derivative_r!(fields.Er,-fields.phi,
                scratch_dummy.buffer_z_1, scratch_dummy.buffer_z_2,
                scratch_dummy.buffer_z_3, scratch_dummy.buffer_z_4,
                r_spectral,r)
        if z.irank == 0 && fields.force_Er_zero_at_wall
            fields.Er[1,:] .= 0.0
        end
        if z.irank == z.nrank - 1 && fields.force_Er_zero_at_wall
            fields.Er[z.n,:] .= 0.0
        end
    else
        @loop_r_z ir iz begin
            fields.Er[iz,ir] = geometry.input.Er_constant
            # Er_constant defaults to 0.0 in geo.jl
        end
    end
    #Ez = - d phi / dz
    if z.n > 1
        derivative_z!(fields.Ez,-fields.phi,
                scratch_dummy.buffer_r_1, scratch_dummy.buffer_r_2,
                scratch_dummy.buffer_r_3, scratch_dummy.buffer_r_4,
                z_spectral,z)
    else
        @loop_r_z ir iz begin
            fields.Ez[iz,ir] = geometry.input.Ez_constant
            # Ez_constant defaults to 0.0 in geo.jl
        end
    end
    
    # get gyroaveraged field arrays for distribution function advance
    gkions = composition.gyrokinetic_ions
    if gkions
        gyroaverage_field!(fields.gphi,fields.phi,gyroavs,vperp,z,r,composition)
        gyroaverage_field!(fields.gEz,fields.Ez,gyroavs,vperp,z,r,composition)
        gyroaverage_field!(fields.gEr,fields.Er,gyroavs,vperp,z,r,composition)
    else # use the drift-kinetic form of the fields in the kinetic equation
        @loop_s_r_z_vperp is ir iz ivperp begin
            fields.gphi[ivperp,iz,ir,is] = fields.phi[iz,ir]
            fields.gEz[ivperp,iz,ir,is] = fields.Ez[iz,ir]
            fields.gEr[ivperp,iz,ir,is] = fields.Er[iz,ir]
        end
    end

end

end
