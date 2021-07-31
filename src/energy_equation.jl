module energy_equation

export energy_equation!

using ..calculus: derivative!
using ..optimization

function energy_equation!(ppar, fvec, moments, CX_frequency, z, dt, spectral, composition)
    for is ∈ 1:composition.n_species
        @views energy_equation_noCX!(ppar[:,is], fvec.upar[:,is], fvec.ppar[:,is],
                                     moments.qpar[:,is], dt, z, spectral)
    end
    # add in contribution due to charge exchange
    if composition.n_neutral_species > 0 && abs(CX_frequency) > 0.0
        @views energy_equation_CX!(ppar, fvec.density, fvec.ppar, composition, CX_frequency, dt)
    end

end
function energy_equation_noCX!(ppar_out, upar, ppar, qpar, dt, z, spectral)
    ithread = Base.Threads.threadid()
    scratch = @view(z.scratch[:,ithread])
    # calculate dppar/dz and store in scratch
    derivative!(scratch, ppar, z, spectral)
    # update ppar to account for contribution from parallel pressure gradient
    @. ppar_out -= dt*upar*scratch
    # calculate dqpar/dz and store in scratch
    derivative!(scratch, qpar, z, spectral)
    # update ppar to account for contribution from parallel heat flux gradient
    @. ppar_out -= dt*scratch
    # calculate dupar/dz and store in scratch
    derivative!(scratch, upar, z, spectral)
    # update ppar to account for contribution from parallel flow gradient
    @. ppar_out -= 3.0*dt*ppar*scratch
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

end
