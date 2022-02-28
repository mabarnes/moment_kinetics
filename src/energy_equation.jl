"""
"""
module energy_equation

export energy_equation!

using ..calculus: derivative!
using ..looping

"""
"""
function energy_equation!(ppar, fvec, moments, collisions, z, r, dt, spectral, composition)
    @loop_s is begin
        @loop_r ir begin
            @views energy_equation_noCX!(ppar[:,ir,is], fvec.upar[:,ir,is], fvec.ppar[:,ir,is],
                                         moments.qpar[:,ir,is], dt, z, spectral)
        end
    end
    # add in contribution due to charge exchange
    if composition.n_neutral_species > 0 && abs(collisions.charge_exchange) > 0.0
        @views energy_equation_CX!(ppar, fvec.density, fvec.ppar, composition, collisions.charge_exchange, dt)
    end

end

"""
"""
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

"""
"""
function energy_equation_CX!(ppar_out, dens, ppar, composition, CX_frequency, dt)
    @loop_s is begin
        @loop_r ir begin
            if is ∈ composition.ion_species_range
                for isp ∈ composition.neutral_species_range
                    @views @. ppar_out[:,ir,is] -= dt*CX_frequency*(dens[:,ir,isp]*ppar[:,ir,is]-dens[:,ir,is]*ppar[:,ir,isp])
                end
            end
            if is ∈ composition.neutral_species_range
                for isp ∈ composition.ion_species_range
                    @views @. ppar_out[:,ir,is] -= dt*CX_frequency*(dens[:,ir,isp]*ppar[:,ir,is]-dens[:,ir,is]*ppar[:,ir,isp])
                end
            end
        end
    end
end

end
