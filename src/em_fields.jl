module em_fields

export setup_em_fields
export update_phi!

using ..type_definitions: mk_float
using ..array_allocation: allocate_float
using ..velocity_moments: update_density!

struct fields
    # phi is the electrostatic potential
    phi::Array{mk_float}
    # phi0 is the initial electrostatic potential
    phi0::Array{mk_float}
    # if including an external forcing for phi, it is of the form
    # phi_external = phi0*drive_amplitude*sinpi(t*drive_frequency)
    force_phi::Bool
    drive_amplitude::mk_float
    drive_frequency::mk_float
end

function setup_em_fields(m, force_phi, drive_amplitude, drive_frequency)
    phi = allocate_float(m)
    phi0 = allocate_float(m)
    return fields(phi, phi0, force_phi, drive_amplitude, drive_frequency)
end

# update_phi updates the electrostatic potential, phi
function update_phi!(fields, fvec, z::Vector, composition)
    update_phi!(fields, fvec, z[1], composition)
end
function update_phi!(fields, fvec, z, composition)
    n_ion_species = composition.n_ion_species
    @boundscheck size(fields.phi,1) == z.n || throw(BoundsError(fields.phi))
    @boundscheck size(fields.phi0,1) == z.n || throw(BoundsError(fields.phi0))
    @boundscheck size(fvec.density,1) == z.n || throw(BoundsError(fvec.density))
    @boundscheck size(fvec.density,2) == composition.n_species || throw(BoundsError(fvec.density))
    if composition.boltzmann_electron_response
        z.scratch .= @view(fvec.density[:,1])
        @inbounds for is ∈ 2:composition.n_ion_species
            for iz ∈ 1:z.n
                z.scratch[iz] += fvec.density[iz,is]
            end
        end
        @inbounds for iz ∈ 1:z.n
            fields.phi[iz] = composition.T_e * log(z.scratch[iz])
        end
        # if fields.force_phi
        #     @inbounds for iz ∈ 1:z.n
        #         fields.phi[iz] += fields.phi0[iz]*fields.drive_amplitude*sin(t*fields.drive_frequency)
        #     end
        # end
    end
end

end
