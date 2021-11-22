module em_fields

export setup_em_fields
export update_phi!

using ..type_definitions: mk_float
using ..array_allocation: allocate_shared_float
using ..communication: block_rank, block_synchronize, MPISharedArray
using ..input_structs
using ..velocity_moments: update_density!

struct fields{dims}
    # phi is the electrostatic potential
    phi::MPISharedArray{dims,mk_float}
    # phi0 is the initial electrostatic potential
    phi0::MPISharedArray{dims,mk_float}
    # if including an external forcing for phi, it is of the form
    # phi_external = phi0*drive_amplitude*sinpi(t*drive_frequency)
    force_phi::Bool
    drive_amplitude::mk_float
    drive_frequency::mk_float
end

function setup_em_fields(m, force_phi, drive_amplitude, drive_frequency)
    phi = allocate_shared_float(z=m)
    phi0 = allocate_shared_float(z=m)
    return fields(phi, phi0, force_phi, drive_amplitude, drive_frequency)
end

# update_phi updates the electrostatic potential, phi
function update_phi!(fields, fvec, z, composition, z_range)
    n_ion_species = composition.n_ion_species
    @boundscheck size(fields.phi,1) == z.n || throw(BoundsError(fields.phi))
    @boundscheck size(fields.phi0,1) == z.n || throw(BoundsError(fields.phi0))
    @boundscheck size(fvec.density,1) == z.n || throw(BoundsError(fvec.density))
    @boundscheck size(fvec.density,2) == composition.n_species || throw(BoundsError(fvec.density))
    # Update phi using the set of processes that handles the first ion species
    # Means we get at least some parallelism, even though we have to sum
    # over species, and reduces number of block_synchronize() calls needed
    # when there is only one species.
    if 1 ∈ composition.species_local_range
        for iz ∈ z_range
            z.scratch[iz] = fvec.density[iz,1]
        end
    end
    composition.n_ion_species > 1 && block_synchronize()
    if 1 ∈ composition.species_local_range
        @inbounds for is ∈ 2:composition.n_ion_species
            for iz ∈ z_range
                z.scratch[iz] += fvec.density[iz,is]
            end
        end
        if composition.electron_physics == boltzmann_electron_response
            N_e = 1.0
        elseif composition.electron_physics == boltzmann_electron_response_with_simple_sheath
            #  calculate Sum_{i} Z_i n_i u_i = J_||i at z = 0
            jpar_i = 0.0
            @inbounds for is ∈ 1:composition.n_ion_species
                jpar_i +=  fvec.density[1,is]*fvec.upar[1,is]
            end
            # Calculate N_e using J_||e at sheath entrance at z = 0 (lower boundary).
            # Assuming pdf is a half maxwellian with boltzmann factor at wall, we have
            # J_||e = e N_e v_{th,e} exp[ e phi_wall / T_e ] / 2 sqrt{pi},
            # where positive sign above (and negative sign below)
            # is due to the fact that electrons reaching the wall flow towards more negative z.
            # Using J_||e + J_||i = 0, and rearranging for N_e, we have
            N_e = - 2.0 * sqrt( pi * composition.me_over_mi) * jpar_i * exp( - composition.phi_wall / composition.T_e)
            # See P.C. Stangeby, The Plasma Boundary of Magnetic Fusion Devices, IOP Publishing, Chpt 2, p75
        end

        if composition.electron_physics ∈ (boltzmann_electron_response, boltzmann_electron_response_with_simple_sheath)
            @inbounds for iz ∈ z_range
                fields.phi[iz] = composition.T_e * log(z.scratch[iz])
            end
            # if fields.force_phi
            #     @inbounds for iz ∈ 1:z.n
            #         fields.phi[iz] += fields.phi0[iz]*fields.drive_amplitude*sin(t*fields.drive_frequency)
            #     end
            # end
        end
    end

    ## can calculate phi at z = L and hence phi_wall(z=L) using jpar_i at z =L if needed

end

end
