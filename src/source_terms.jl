"""
"""
module source_terms

export source_terms!

using ..calculus: derivative!
using ..looping

"""
calculate the source terms due to redefinition of the pdf to split off density,
flow and/or pressure, and use them to update the pdf
"""
function source_terms!(pdf_out, fvec_in, moments, vpa, z, r, dt, spectral, composition, collisions)

    begin_s_r_z_region()

    #n_species = size(pdf_out,3)
    if moments.evolve_ppar
        @loop_s is begin
            @views source_terms_evolve_ppar_no_collisions!(pdf_out[:,:,:,is], fvec_in.pdf[:,:,:,is],
                                             fvec_in.density[:,:,is], fvec_in.upar[:,:,is], fvec_in.ppar[:,:,is],
                                             moments.vth[:,:,is], moments.qpar[:,:,is], z, r, dt, spectral)
        end
        if composition.n_neutral_species > 0
            if abs(collisions.charge_exchange) > 0.0 || abs(collisions.ionization) > 0.0
                @views source_terms_evolve_ppar_collisions!(pdf_out[:,:,:,:], fvec_in.pdf[:,:,:,:],
                                                fvec_in.density, fvec_in.upar,
                                                fvec_in.ppar, composition, collisions,
                                                dt, z, r)
            end
        end
    elseif moments.evolve_density
        @loop_s is begin
            @views source_terms_evolve_density!(pdf_out[:,:,:,is], fvec_in.pdf[:,:,:,is],
                                                fvec_in.density[:,:,is], fvec_in.upar[:,:,is], z, r, dt, spectral)
        end
    end
    return nothing
end

"""
"""
function source_terms_evolve_density!(pdf_out, pdf_in, dens, upar, z, r, dt, spectral)
    # update the density
    nvpa = size(pdf_out, 1)
    @loop_r ir begin
        # calculate d(n*upar)/dz
        @views @. z.scratch = dens[:,ir]*upar[:,ir]
        derivative!(z.scratch, z.scratch, z, spectral)
        @views @. z.scratch *= dt/dens[:,ir]
        #derivative!(z.scratch, z.scratch, z, -upar, spectral)
        @loop_z_vpa iz ivpa begin
            pdf_out[ivpa,iz,ir] += pdf_in[ivpa,iz,ir]*z.scratch[iz]
        end
    end
    return nothing
end

"""
update the evolved pdf to account for the collisionless source terms in the kinetic equation
arising due to the re-normalization of the pdf as g = f * vth / n
"""
function source_terms_evolve_ppar_no_collisions!(pdf_out, pdf_in, dens, upar, ppar, vth, qpar, z, r, dt, spectral)
    nvpa = size(pdf_out, 1)
    @loop_r ir begin
        # calculate dn/dz
        derivative!(z.scratch, view(dens,:,ir), z, spectral)
        # update the pdf to account for the density gradient contribution to the source
        @views @. z.scratch *= upar[:,ir]/dens[:,ir]
        # calculate dvth/dz
        derivative!(z.scratch2, view(vth,:,ir), z, spectral)
        # update the pdf to account for the -g*upar/vth * dvth/dz contribution to the source
        @views @. z.scratch -= z.scratch2*upar[:,ir]/vth[:,ir]
        # calculate dqpar/dz
        derivative!(z.scratch2, view(qpar,:,ir), z, spectral)
        # update the pdf to account for the parallel heat flux contribution to the source
        @views @. z.scratch -= 0.5*z.scratch2/ppar[:,ir]

        @loop_z_vpa iz ivpa begin
            pdf_out[ivpa,iz,ir] += dt*pdf_in[ivpa,iz,ir]*z.scratch[iz]
        end
    end
    return nothing
end

"""
update the evolved pdf to account for the charge exchange and ionization source terms in the
kinetic equation arising due to the re-normalization of the pdf as g = f * vth / n
"""
function source_terms_evolve_ppar_collisions!(pdf_out, pdf_in, dens, upar, ppar,
                                              composition, collisions, dt, z, r)
    @loop_s is begin
        if is ∈ composition.ion_species_range
            for isp ∈ composition.neutral_species_range
                @loop_r_z ir iz begin
                    @views @. pdf_out[:,iz,ir,is] -= 0.5*dt*pdf_in[:,iz,ir,is] *
                    (collisions.charge_exchange
                       * (dens[iz,ir,isp]*ppar[iz,ir,is] - dens[iz,ir,is]*ppar[iz,ir,isp]
                          - dens[iz,ir,is]*dens[iz,ir,isp]
                            * (upar[iz,ir,is] - upar[iz,ir,isp])^2)
                       / ppar[iz,ir,is]
                     + collisions.ionization
                       * (3.0*dens[iz,ir,isp]
                          - dens[iz,ir,is]*(ppar[iz,ir,isp]
                                            + dens[iz,ir,isp]*(upar[iz,ir,is] - upar[iz,ir,isp])^2)
                            / ppar[iz,ir,is]))
                end
            end
        end
        if is ∈ composition.neutral_species_range
            for isp ∈ composition.ion_species_range
                @loop_r_z ir iz begin
                    @views @. pdf_out[:,iz,ir,is] -= 0.5*dt*pdf_in[:,iz,ir,is] *
                    (collisions.charge_exchange
                       * (dens[iz,ir,isp]*ppar[iz,ir,is] - dens[iz,ir,is]*ppar[iz,ir,isp]
                          - dens[iz,ir,is]*dens[iz,ir,isp]
                            * (upar[iz,ir,is] - upar[iz,ir,isp])^2)/ppar[iz,ir,is]
                     - 2.0*collisions.ionization*dens[iz,ir,isp])
                end
            end
        end
    end
    return nothing
end

end
