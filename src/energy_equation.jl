module energy_equation

export energy_equation!

using ..calculus: derivative!

function energy_equation!(ppar, fvec, moments, collisions, z, dt, spectral, composition)
    for is ∈ 1:composition.n_species
        @views energy_equation_collisionless!(ppar[:,is], fvec.upar[:,is], fvec.ppar[:,is],
                                     moments.qpar[:,is], dt, z, spectral)
    end
    # add in contributions due to collisions
    if composition.n_neutral_species > 0
        # add in contribution from charge exchange collisions
        if abs(collisions.charge_exchange) > 0.0
            @views energy_equation_CX!(ppar, fvec.density, fvec.ppar, composition, collisions.charge_exchange, dt)
        end
        # add in contribution from ionization collisions
        if abs(collisions.ionization) > 0.0
            @views energy_equation_ionization!(ppar, fvec.density, fvec.ppar, composition, collisions.ionization, dt)
        end
    end
end
function energy_equation_collisionless!(ppar_out, upar, ppar, qpar, dt, z, spectral)
    # calculate dppar/dz and store in z.scratch
    derivative!(z.scratch, ppar, z, spectral)
    # update ppar to account for contribution from parallel pressure gradient
    @. ppar_out -= dt*upar*z.scratch
    # calculate dqpar/dz and store in z.scratch
    derivative!(z.scratch, qpar, z, spectral)
    # update ppar to account for contribution from parallel heat flux gradient
    @. ppar_out -= dt*z.scratch
    # calculate dupar/dz and store in z.scratch
    derivative!(z.scratch, upar, z, spectral)
    # update ppar to account for contribution from parallel flow gradient
    @. ppar_out -= 3.0*dt*ppar*z.scratch
end
function energy_equation_CX!(ppar_out, dens, ppar, composition, CX_frequency, dt)
    for is ∈ 1:composition.n_ion_species
        for isp ∈ composition.n_ion_species+1:composition.n_species
            @views @. ppar_out[:,is] -= dt*CX_frequency*(dens[:,isp]*ppar[:,is]-dens[:,is]*ppar[:,isp])
        end
    end
    for is ∈ composition.n_ion_species+1:composition.n_species
        for isp ∈ 1:composition.n_ion_species
            @views @. ppar_out[:,is] -= dt*CX_frequency*(dens[:,isp]*ppar[:,is]-dens[:,is]*ppar[:,isp])
        end
    end
end
function energy_equation_ionization!(ppar_out, dens, ppar, composition, ionization_frequency, dt)
    for is ∈ 1:composition.n_ion_species
        for isp ∈ composition.n_ion_species+1:composition.n_species
            @views @. ppar_out[:,is] += dt*ionization_frequency*dens[:,is]*ppar[:,isp]
        end
    end
    for is ∈ composition.n_ion_species+1:composition.n_species
        for isp ∈ 1:composition.n_ion_species
            @views @. ppar_out[:,is] -= dt*ionization_frequency*dens[:,isp]*ppar[:,is]
        end
    end
end

end
