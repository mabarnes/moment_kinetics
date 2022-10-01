"""
"""
module source_terms

export source_terms!

using ..array_allocation: allocate_float
using ..calculus: derivative!
using ..coordinates: coordinate
using ..input_structs: species_composition
using ..looping
using ..type_definitions: mk_float

"""
Parameters for external source terms, e.g. ion heating
"""
struct source_info
    # Include ion heating?
    ion_heating::Bool
    # Heat source
    S_heat::Array{mk_float,2}
end

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

"""
Add external sources, e.g. ion heating.
"""
function external_sources!(pdf_out, fvec_in, moments, sources, vpa, vpa_spectral,
                           composition, dt)
    if sources.ion_heating
        if moments.evolve_ppar
            # Ion heating does not affect normalized distribution function
        elseif moments.evolve_upar
            # Heating source does not enter force_balance equation, so compared to
            # evolve_density version just need to account for vpa.grid being
            # w_parallel=(v_parallel-u_parallel)
            begin_s_r_z_region()
            @loop_s_r_z is ir iz begin
                if is ∈ composition.ion_species_range
                    @views derivative!(vpa.scratch, fvec_in.pdf[:,iz,ir,is], vpa, vpa_spectral)
                    @views @. pdf_out[:,iz,ir,is] -=
                        dt * sources.S_heat[iz,is] *
                        (fvec_in.pdf[:,iz,ir,is] + vpa.grid * vpa.scratch) /
                        (fvec_in.density[iz,ir,is] * moments.vth[iz,ir,is])^2
                end
            end
        elseif moments.evolve_density
            # Heating source does not enter continuity equation, so just need to divide
            # full-f source term by n_i to account for normalized g_i.
            begin_s_r_z_region()
            @loop_s_r_z is ir iz begin
                if is ∈ composition.ion_species_range
                    @views derivative!(vpa.scratch, fvec_in.pdf[:,iz,ir,is], vpa, vpa_spectral)
                    @views @. pdf_out[:,iz,ir,is] -=
                        dt * sources.S_heat[iz,is] *
                        (fvec_in.pdf[:,iz,ir,is]
                         + (vpa.grid - fvec_in.upar[iz,ir,is]) * vpa.scratch) /
                        (fvec_in.density[iz,ir,is] * moments.vth[iz,ir,is])^2
                end
            end
        else
            # 'full-f' case
            begin_s_r_z_region()
            @loop_s_r_z is ir iz begin
                if is ∈ composition.ion_species_range
                    @views derivative!(vpa.scratch, fvec_in.pdf[:,iz,ir,is], vpa, vpa_spectral)
                    @views @. pdf_out[:,iz,ir,is] -=
                        dt * sources.S_heat[iz,is] *
                        (fvec_in.pdf[:,iz,ir,is]
                         + (vpa.grid - fvec_in.upar[iz,ir,is]) * vpa.scratch) /
                        (fvec_in.density[iz,ir,is] * moments.vth[iz,ir,is]^2)
                end
            end
        end
    end

    return nothing
end

"""
Create arrays and parameters for external sources, e.g. ion heating
"""
function init_external_sources(source_input::Dict, z::coordinate,
                               composition::species_composition)

    ion_heating_amplitude = source_input["ion_heating_amplitude"]
    S_heat = allocate_float(z.n, composition.n_ion_species)
    if ion_heating_amplitude <= 0.0
        ion_heating = false
        S_heat .= 0.0
    else
        ion_heating = true
        ion_heating_width = source_input["ion_heating_width"]

        for is ∈ 1:composition.n_ion_species, iz ∈ 1:z.n
            S_heat[iz,is] = ion_heating_amplitude * exp(-(z.grid[iz]/ion_heating_width)^2)
        end
    end

    return source_info(ion_heating, S_heat)
end

end
