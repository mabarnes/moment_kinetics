"""
"""
module energy_equation

export energy_equation!

using ..calculus: derivative!
using ..looping

"""
evolve the parallel pressure by solving the energy equation
"""
function energy_equation!(ppar, fvec, moments, collisions, z, r, dt, spectral, composition)

    begin_s_r_region()

    @loop_s is begin
        @loop_r ir begin
            @views energy_equation_no_collisions!(ppar[:,ir,is], fvec.upar[:,ir,is], fvec.ppar[:,ir,is],
                                                  moments.qpar[:,ir,is], dt, z, spectral)
        end
    end
    # add in contributions due to charge exchange/ionization collisions
    if composition.n_neutral_species > 0
        if abs(collisions.charge_exchange) > 0.0
            @views energy_equation_CX!(ppar, fvec.density, fvec.ppar, composition, collisions.charge_exchange, dt)
        end
        if abs(collisions.ionization) > 0.0
            @views energy_equation_ionization!(ppar, fvec.density, fvec.ppar, composition, collisions.ionization, dt)
        end
    end

end

"""
include all contributions to the energy equation aside from collisions
"""
function energy_equation_no_collisions!(ppar_out, upar, ppar, qpar, dt, z, spectral)
    # calculate dppar/dz and store in z.scratch
    # Use as 'adv_fac' for upwinding
    @. z.scratch3 = -upar
    derivative!(z.scratch, ppar, z, z.scratch3, spectral)
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
include the contribution to the energy equation from charge exchange collisions
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

"""
include the contribution to the energy equation from ionization collisions
"""
function energy_equation_ionization!(ppar_out, dens, ppar, composition, ionization_frequency, dt)
    @loop_s is begin
        @loop_r ir begin
            if is ∈ composition.ion_species_range
                for isp ∈ composition.neutral_species_range
                    @views @. ppar_out[:,ir,is] += dt*ionization_frequency*dens[:,ir,is]*ppar[:,ir,isp]
                end
            end
            if is ∈ composition.neutral_species_range
                for isp ∈ composition.ion_species_range
                    @views @. ppar_out[:,ir,is] -= dt*ionization_frequency*dens[:,ir,isp]*ppar[:,ir,is]
                end
            end
        end
    end
end

end
