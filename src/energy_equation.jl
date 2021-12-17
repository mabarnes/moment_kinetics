module energy_equation

export energy_equation!

using ..calculus: derivative!

function energy_equation!(ppar, fvec, moments, collisions, z, dt, spectral, composition)
    for is ∈ composition.species_local_range
        if composition.first_proc_in_group
            @views energy_equation_noCX!(ppar[:,is], fvec.upar[:,is], fvec.ppar[:,is],
                                         moments.qpar[:,is], dt, z, spectral)
        end
    end
    # add in contribution due to charge exchange
    if composition.n_neutral_species > 0 && abs(collisions.charge_exchange) > 0.0
        @views energy_equation_CX!(ppar, fvec.density, fvec.ppar, composition, collisions.charge_exchange, dt)
    end

end
function energy_equation_noCX!(ppar_out, upar, ppar, qpar, dt, z, spectral)
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
    for is ∈ composition.species_local_range
        if composition.first_proc_in_group
            if is ∈ composition.ion_species_range
                for isp ∈ composition.neutral_species_range
                    @views @. ppar_out[:,is] -= dt*CX_frequency*(dens[:,isp]*ppar[:,is]-dens[:,is]*ppar[:,isp])
                end
            end
            if is ∈ composition.neutral_species_range
                for isp ∈ composition.ion_species_range
                    @views @. ppar_out[:,is] -= dt*CX_frequency*(dens[:,isp]*ppar[:,is]-dens[:,is]*ppar[:,isp])
                end
            end
        end
    end
end

end
