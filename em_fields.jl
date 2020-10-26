module em_fields

export setup_em_fields
export update_phi!

using array_allocation: allocate_float
using moment_kinetics_input: boltzmann_electron_response
using velocity_moments: update_density!

struct fields
    phi::Array{Float64}
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
            update_density!(moments.dens, moments.scratch, ff, vpa, nz)
            moments.dens_updated = true
        end
        @inbounds begin
            for iz ∈ 1:nz
                phi[iz] = moments.dens[iz]
            end
        end
    end
end

end
