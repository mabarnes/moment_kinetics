module em_fields

export setup_em_fields
export update_phi!

using type_definitions: mk_float
using array_allocation: allocate_float
using moment_kinetics_input: boltzmann_electron_response
using velocity_moments: update_density!

struct fields
    phi::Array{mk_float}
end

function setup_em_fields(m)
    phi = allocate_float(m)
    return fields(phi)
end

# update_phi updates the electrostatic potential, phi
function update_phi!(phi, moments, ff, vpa, nz)
    @boundscheck size(phi,1) == nz || throw(BoundsError(phi))
    @boundscheck size(moments.dens,1) == nz || throw(BoundsError(moments.dens))
    if boltzmann_electron_response
        if moments.dens_updated == false
            update_density!(moments.dens, vpa.scratch, ff, vpa, nz)
            moments.dens_updated = true
        end
        @inbounds begin
            for iz ∈ 1:nz
                phi[iz] = log(moments.dens[iz])
            end
        end
    end
end

end
