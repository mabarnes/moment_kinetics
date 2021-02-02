module em_fields

export setup_em_fields
export update_phi!

using type_definitions: mk_float
using array_allocation: allocate_float
using velocity_moments: update_density!

struct fields
    phi::Array{mk_float}
end

function setup_em_fields(m)
    phi = allocate_float(m)
    return fields(phi)
end

# update_phi updates the electrostatic potential, phi
function update_phi!(phi, moments, ff, vpa, nz, composition)
    n_ion_species = composition.n_ion_species
    @boundscheck size(phi,1) == nz || throw(BoundsError(phi))
    @boundscheck size(moments.dens,1) == nz || throw(BoundsError(moments.dens))
    @boundscheck size(moments.dens,2) == composition.n_species || throw(BoundsError(moments.dens))
    if composition.boltzmann_electron_response
        for is ∈ 1:composition.n_ion_species
            if moments.dens_updated[is] == false
                @views update_density!(moments.dens[:,is], vpa.scratch, ff[:,:,is], vpa, nz)
                moments.dens_updated[is] = true
            end
        end
        @inbounds begin
            for iz ∈ 1:nz
                total_density = 0.0
                for is ∈ 1:composition.n_ion_species
                    total_density += moments.dens[iz,is]
                end
                phi[iz] = log(total_density)
            end
        end
    end
end

end
