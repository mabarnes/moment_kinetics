"""
"""
module source_terms

export source_terms!
export source_terms_manufactured!

using ..calculus: derivative!
using ..looping

"""
calculate the source terms due to redefinition of the pdf to split off density,
flow and/or pressure, and use them to update the pdf
"""
function source_terms!(pdf_out, fvec_in, moments, vpa, z, r, dt, spectral, composition, collisions)

    begin_s_r_z_vperp_vpa_region()

    #n_species = size(pdf_out,3)
    if moments.evolve_ppar
        @loop_s is begin
            @views source_terms_evolve_ppar_no_collisions!(
                pdf_out[:,:,:,:,is], fvec_in.pdf[:,:,:,:,is], fvec_in.density[:,:,is],
                fvec_in.upar[:,:,is], fvec_in.ppar[:,:,is], moments.ion.vth[:,:,is],
                moments.ion.qpar[:,:,is], z, r, dt, spectral)
            if composition.n_neutral_species > 0
                if abs(collisions.charge_exchange) > 0.0 || abs(collisions.ionization) > 0.0
                    @views source_terms_evolve_ppar_collisions!(
                        pdf_out[:,:,:,:,is], fvec_in.pdf[:,:,:,:,is],
                        fvec_in.density[:,:,is], fvec_in.upar[:,:,is],
                        fvec_in.ppar[:,:,is], fvec_in.density_neutral[:,:,is],
                        fvec_in.uz_neutral[:,:,is], fvec_in.pz_neutral[:,:,is],
                        composition, collisions, dt, z, r)
                end
            end
        end
    elseif moments.evolve_density
        @loop_s is begin
            @views source_terms_evolve_density!(
                pdf_out[:,:,:,:,is], fvec_in.pdf[:,:,:,:,is], fvec_in.density[:,:,is],
                fvec_in.upar[:,:,is], z, r, dt, spectral)
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
        @loop_z_vperp_vpa iz ivperp ivpa begin
            pdf_out[ivpa,ivperp,iz,ir] += pdf_in[ivpa,ivperp,iz,ir]*z.scratch[iz]
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

        @loop_z_vperp_vpa iz ivperp ivpa begin
            pdf_out[ivpa,ivperp,iz,ir] += dt*pdf_in[ivpa,ivperp,iz,ir]*z.scratch[iz]
        end
    end
    return nothing
end

"""
update the evolved pdf to account for the charge exchange and ionization source terms in the
kinetic equation arising due to the re-normalization of the pdf as g = f * vth / n
"""
function source_terms_evolve_ppar_collisions!(pdf_out, pdf_in, dens, upar, ppar,
                                              dens_neutral, upar_neutral, ppar_neutral,
                                              composition, collisions, dt, z, r)
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        @views pdf_out[ivpa,ivperp,iz,ir] -= 0.5*dt*pdf_in[ivpa,ivperp,iz,ir] *
            (collisions.charge_exchange
               * (dens_neutral[iz,ir]*ppar[iz,ir] - dens[iz,ir]*ppar_neutral[iz,ir]
                  - dens[iz,ir]*dens_neutral[iz,ir]
                    * (upar[iz,ir] - upar_neutral[iz,ir])^2)
               / ppar[iz,ir]
             + collisions.ionization
               * (3.0*dens_neutral[iz,ir]
                  - dens[iz,ir]*(ppar_neutral[iz,ir]
                                    + dens_neutral[iz,ir]*(upar[iz,ir] - upar_neutral[iz,ir])^2)
                    / ppar[iz,ir]))
    end
    return nothing
end

"""
calculate the source terms due to redefinition of the pdf to split off density,
flow and/or pressure, and use them to update the pdf
"""
function source_terms_neutral!(pdf_out, fvec_in, moments, vpa, z, r, dt, spectral, composition, collisions)

    begin_sn_r_z_vzeta_vr_vz_region()

    #n_species = size(pdf_out,3)
    if moments.evolve_ppar
        @loop_sn isn begin
            @views source_terms_evolve_ppar_no_collisions_neutral!(
                pdf_out[:,:,:,:,:,isn], fvec_in.pdf_neutral[:,:,:,:,:,isn],
                fvec_in.density_neutral[:,:,isn], fvec_in.uz_neutral[:,:,isn],
                fvec_in.pz_neutral[:,:,isn], moments.neutral.vth[:,:,isn],
                moments.neutral.qz[:,:,isn], z, r, dt, spectral)
            if abs(collisions.charge_exchange) > 0.0 || abs(collisions.ionization) > 0.0
                @views source_terms_evolve_ppar_collisions_neutral!(
                    pdf_out[:,:,:,:,:,isn], fvec_in.pdf_neutral[:,:,:,:,:,isn],
                    fvec_in.density_neutral[:,:,isn], fvec_in.uz_neutral[:,:,isn],
                    fvec_in.pz_neutral[:,:,isn],fvec_in.density[:,:,isn],
                    fvec_in.upar[:,:,isn], fvec_in.ppar[:,:,isn], composition, collisions,
                    dt, z, r)
            end
        end
    elseif moments.evolve_density
        @loop_sn isn begin
            @views source_terms_evolve_density_neutral!(
                pdf_out[:,:,:,:,:,isn], fvec_in.pdf_neutral[:,:,:,:,:,isn],
                fvec_in.density_neutral[:,:,isn], fvec_in.uz_neutral[:,:,isn], z, r, dt,
                spectral)
        end
    end
    return nothing
end

"""
"""
function source_terms_evolve_density_neutral!(pdf_out, pdf_in, dens, upar, z, r, dt,
                                              spectral)
    # update the density
    nvpa = size(pdf_out, 1)
    @loop_r ir begin
        # calculate d(n*upar)/dz
        @views @. z.scratch = dens[:,ir]*upar[:,ir]
        derivative!(z.scratch, z.scratch, z, spectral)
        @views @. z.scratch *= dt/dens[:,ir]
        #derivative!(z.scratch, z.scratch, z, -upar, spectral)
        @loop_z_vzeta_vr_vz iz ivzeta ivr ivz begin
            pdf_out[ivz,ivr,ivzeta,iz,ir] += pdf_in[ivz,ivr,ivzeta,iz,ir]*z.scratch[iz]
        end
    end
    return nothing
end

"""
update the evolved pdf to account for the collisionless source terms in the kinetic equation
arising due to the re-normalization of the pdf as g = f * vth / n
"""
function source_terms_evolve_ppar_no_collisions_neutral!(pdf_out, pdf_in, dens, upar,
                                                         ppar, vth, qpar, z, r, dt,
                                                         spectral)
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

        @loop_z_vzeta_vr_vz iz ivzeta ivr ivz begin
            pdf_out[ivz,ivr,ivzeta,iz,ir] += dt*pdf_in[ivz,ivr,ivzeta,iz,ir]*z.scratch[iz]
        end
    end
    return nothing
end

"""
update the evolved pdf to account for the charge exchange and ionization source terms in the
kinetic equation arising due to the re-normalization of the pdf as g = f * vth / n
"""
function source_terms_evolve_ppar_collisions_neutral!(pdf_out, pdf_in, dens, upar, ppar,
                                                      dens_ion, upar_ion, ppar_ion,
                                                      composition, collisions, dt, z, r)
    @loop_r_z_vzeta_vr_vz ir iz ivzeta ivr ivz begin
        @views pdf_out[ivz,ivr,ivzeta,iz,ir] -= 0.5*dt*pdf_in[ivz,ivr,ivzeta,iz,ir] *
        (collisions.charge_exchange
           * (dens_ion[iz,ir]*ppar[iz,ir] - dens[iz,ir]*ppar_ion[iz,ir]
              - dens[iz,ir]*dens_ion[iz,ir]
                * (upar[iz,ir] - upar_ion[iz,ir])^2)/ppar[iz,ir]
         - 2.0*collisions.ionization*dens_ion[iz,ir])
    end
    return nothing
end

"""
advance the dfn with an arbitrary source function 
"""
function source_terms_manufactured!(pdf_ion_out, pdf_neutral_out, vz, vr, vzeta, vpa, vperp, z, r, t, dt, composition, manufactured_source_list)
    if manufactured_source_list.time_independent_sources
        # the (time-independent) manufactured source arrays
        Source_i = manufactured_source_list.Source_i_array
        Source_n = manufactured_source_list.Source_n_array

        begin_s_r_z_region()

        @loop_s is begin
            @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                pdf_ion_out[ivpa,ivperp,iz,ir,is] += dt*Source_i[ivpa,ivperp,iz,ir]
            end
        end

        if composition.n_neutral_species > 0
            begin_sn_r_z_region()
            @loop_sn isn begin
                @loop_r_z_vzeta_vr_vz ir iz ivzeta ivr ivz begin
                    pdf_neutral_out[ivz,ivr,ivzeta,iz,ir,isn] += dt*Source_n[ivz,ivr,ivzeta,iz,ir]
                end
            end
        end
    else
        # the manufactured source functions
        Source_i_func = manufactured_source_list.Source_i_func
        Source_n_func = manufactured_source_list.Source_n_func

        begin_s_r_z_region()

        @loop_s is begin
            @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                pdf_ion_out[ivpa,ivperp,iz,ir,is] += dt*Source_i_func(vpa.grid[ivpa],vperp.grid[ivperp],z.grid[iz],r.grid[ir],t)
            end
        end

        if composition.n_neutral_species > 0
            begin_sn_r_z_region()
            @loop_sn isn begin
                @loop_r_z_vzeta_vr_vz ir iz ivzeta ivr ivz begin
                    pdf_neutral_out[ivz,ivr,ivzeta,iz,ir,isn] += dt*Source_n_func(vz.grid[ivz],vr.grid[ivr],vzeta.grid[ivzeta],z.grid[iz],r.grid[ir],t)
                end
            end
        end
    end
    return nothing
end

end
