"""
"""
module energy_equation

export energy_equation!

using ..calculus: derivative!
using ..looping

"""
evolve the parallel pressure by solving the energy equation
"""
function energy_equation!(ppar, fvec, moments, collisions, z, r, dt, spectral,
                          composition, num_diss_params)

    begin_s_r_region()

    @loop_s is begin
        @loop_r ir begin
            @views energy_equation_no_collisions!(ppar[:,ir,is], fvec.upar[:,ir,is], fvec.ppar[:,ir,is],
                                                  moments.qpar[:,ir,is], dt, z, spectral, num_diss_params)
        end
    end
    # add in contributions due to charge exchange/ionization collisions
    if composition.n_neutral_species > 0
        if abs(collisions.charge_exchange) > 0.0
            @views energy_equation_CX!(ppar, fvec.density, fvec.upar, fvec.ppar, z,
                                       composition, collisions.charge_exchange, dt)
        end
        if abs(collisions.ionization) > 0.0
            @views energy_equation_ionization!(ppar, fvec.density, fvec.upar, fvec.ppar,
                                               z, composition, collisions.ionization,
                                               dt)
        end
    end

end

"""
include all contributions to the energy equation aside from collisions
"""
function energy_equation_no_collisions!(ppar_out, upar, ppar, qpar, dt, z, spectral,
                                        num_diss_params)
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

    # Ad-hoc diffusion to stabilise numerics...
    diffusion_coefficient = num_diss_params.moment_dissipation_coefficient
    if diffusion_coefficient > 0.0
        derivative!(z.scratch, ppar, z, spectral, Val(2))
        @. ppar_out += dt*diffusion_coefficient*z.scratch
    end

    return nothing
end

"""
include the contribution to the energy equation from charge exchange collisions
"""
function energy_equation_CX!(ppar_out, dens, upar, ppar, z, composition, CX_frequency, dt)
    @loop_s is begin
        @loop_r ir begin
            if is ∈ composition.ion_species_range
                for isp ∈ composition.neutral_species_range
                    @loop_z iz begin
                        ppar_out[iz,ir,is] -=
                            dt*CX_frequency*(dens[iz,ir,isp]*ppar[iz,ir,is] -
                                             dens[iz,ir,is]*ppar[iz,ir,isp] -
                                             dens[iz,ir,is]*dens[iz,ir,isp] *
                                             (upar[iz,ir,is] - upar[iz,ir,isp])^2)
                    end
                end
            end
            if is ∈ composition.neutral_species_range
                for isp ∈ composition.ion_species_range
                    @loop_z iz begin
                        ppar_out[iz,ir,is] -=
                            dt*CX_frequency*(dens[iz,ir,isp]*ppar[iz,ir,is] -
                                             dens[iz,ir,is]*ppar[iz,ir,isp] -
                                             dens[iz,ir,is]*dens[iz,ir,isp] *
                                             (upar[iz,ir,is] - upar[iz,ir,isp])^2)
                    end
                end
            end
        end
    end
end

"""
include the contribution to the energy equation from ionization collisions
"""
function energy_equation_ionization!(ppar_out, dens, upar, ppar, z, composition,
                                     ionization_frequency, dt)
    @loop_s is begin
        @loop_r ir begin
            if is ∈ composition.ion_species_range
                for isp ∈ composition.neutral_species_range
                    @loop_z iz begin
                        ppar_out[iz,ir,is] +=
                            dt*ionization_frequency*dens[iz,ir,is] *
                                (ppar[iz,ir,isp] +
                                 dens[iz,ir,isp] * (upar[iz,ir,is] - upar[iz,ir,isp])^2)
                    end
                end
            end
            if is ∈ composition.neutral_species_range
                for isp ∈ composition.ion_species_range
                    @loop_z iz begin
                        ppar_out[iz,ir,is] -=
                            dt*ionization_frequency*dens[iz,ir,isp]*ppar[iz,ir,is]
                    end
                end
            end
        end
    end
end

end
