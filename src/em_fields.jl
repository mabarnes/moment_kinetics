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
using ..calculus: derivative!
using ..manufactured_solns: manufactured_electric_fields

"""
"""
function setup_em_fields(nz, nr, force_phi, drive_amplitude, drive_frequency)
    phi = allocate_shared_float(nz,nr)
    phi0 = allocate_shared_float(nz,nr)
    Er = allocate_shared_float(nz,nr)
    Ez = allocate_shared_float(nz,nr)
    return em_fields_struct(phi, phi0, Er, Ez, force_phi, drive_amplitude, drive_frequency)
end

"""
update_phi updates the electrostatic potential, phi
"""
function update_phi!(fields, fvec, z, r, composition, z_spectral, r_spectral, use_manufactured_electric_fields, t)
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
    
    begin_serial_region(no_synchronize=true)
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
    @loop_r ir begin # radial locations uncoupled so perform boltzmann solve 
                           # for each radial position in parallel if possible 
        if 1 ∈ loop_ranges[].s
            @loop_z iz begin
                z.scratch[iz] = fvec.density[iz,ir,1]
            end
            @inbounds for is ∈ 2:composition.n_ion_species
                @loop_z iz begin
                    z.scratch[iz] += fvec.density[iz,ir,is]
                end
            end
            if composition.electron_physics == boltzmann_electron_response
                N_e = 1.0
            elseif composition.electron_physics == boltzmann_electron_response_with_simple_sheath
                # calculate Sum_{i} Z_i n_i u_i = J_||i at z = 0
                jpar_i = 0.0
                @inbounds for is ∈ 1:composition.n_ion_species
                    jpar_i +=  fvec.density[1,ir,is]*fvec.upar[1,ir,is]
                end
                # Calculate N_e using J_||e at sheath entrance at z = 0 (lower boundary).
                # Assuming pdf is a half maxwellian with boltzmann factor at wall, we have
                # J_||e = e N_e v_{th,e} exp[ e phi_wall / T_e ] / 2 sqrt{pi},
                # where positive sign above (and negative sign below)
                # is due to the fact that electrons reaching the wall flow towards more negative z.
                # Using J_||e + J_||i = 0, and rearranging for N_e, we have
                N_e = - 2.0 * sqrt( pi * composition.me_over_mi) * jpar_i * exp( - composition.phi_wall / composition.T_e)
                # N_e must be positive, so force this in case a numerical error or something
                # made jpar_i negative
                N_e = max(N_e, 1.e-16)
                # See P.C. Stangeby, The Plasma Boundary of Magnetic Fusion Devices, IOP Publishing, Chpt 2, p75
            end

            if composition.electron_physics ∈ (boltzmann_electron_response, boltzmann_electron_response_with_simple_sheath)
                @loop_z iz begin
                    fields.phi[iz,ir] = composition.T_e * log(z.scratch[iz] / N_e)
                end
                # if fields.force_phi
                #     @inbounds for iz ∈ 1:z.n
                #         fields.phi[iz] += fields.phi0[iz]*fields.drive_amplitude*sin(t*fields.drive_frequency)
                #     end
                # end
            end
        end

   
    ## can calculate phi at z = L and hence phi_wall(z=L) using jpar_i at z =L if needed
    end # end of r loop
    
    ## calculate the electric fields after obtaining phi
    #Er = - d phi / dr 
    if 1 ∈ loop_ranges[].s
        if r.n > 1
            @loop_z iz begin
                derivative!(r.scratch, view(fields.phi,iz,:), r, r_spectral)
                @loop_r ir begin
                    fields.Er[iz,ir] = -r.scratch[ir]
                end
            end
        else
            fields.Er[:,:] .= 0.0
        end
        #Ez = - d phi / dz
        @loop_r ir begin
            derivative!(z.scratch, view(fields.phi,:,ir), z, z_spectral)
            @loop_z iz begin
                fields.Ez[iz,ir] = -z.scratch[iz]
            end
        end
        
        if use_manufactured_electric_fields
            electric_fields = manufactured_electric_fields(r.L,z.L,r.bc,z.bc)
            Er_func = electric_fields.Er_func
            Ez_func = electric_fields.Ez_func
            if r.n > 1
                @loop_r_z ir iz begin
                    fields.Er[iz,ir] = Er_func(z.grid[iz],r.grid[ir],t)
                end
            else
                fields.Er[:,:] .= 0.0
            end
            @loop_r_z ir iz begin
                fields.Ez[iz,ir] = Ez_func(z.grid[iz],r.grid[ir],t)
            end
        end     
        
    end
end

end
