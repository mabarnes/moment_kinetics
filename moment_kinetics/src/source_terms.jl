"""
"""
module source_terms

export source_terms!
export source_terms_manufactured!

using ..calculus: derivative!
using ..looping
using ..timer_utils

"""
calculate the source terms due to redefinition of the pdf to split off density,
flow and/or pressure, and use them to update the pdf
"""
@timeit global_timer source_terms!(
                         pdf_out, fvec_in, moments, vpa, z, r, dt, spectral, composition,
                         collisions, ion_source_settings) = begin

    @begin_s_r_z_vperp_vpa_region()

    #n_species = size(pdf_out,3)
    if moments.evolve_ppar
        @loop_s is begin
            @views source_terms_evolve_ppar_no_collisions!(
                pdf_out[:,:,:,:,is], fvec_in.pdf[:,:,:,:,is], fvec_in.density[:,:,is],
                fvec_in.upar[:,:,is], fvec_in.ppar[:,:,is], moments.ion.vth[:,:,is],
                moments.ion.qpar[:,:,is], moments.ion.ddens_dz[:,:,is],
                moments.ion.dvth_dz[:,:,is], moments.ion.dqpar_dz[:,:,is],
                moments, z, r, dt, spectral, ion_source_settings)
            if composition.n_neutral_species > 0
                if abs(collisions.reactions.charge_exchange_frequency) > 0.0 || abs(collisions.reactions.ionization_frequency) > 0.0
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
                fvec_in.upar[:,:,is], moments.ion.ddens_dz[:,:,is],
                moments.ion.dupar_dz[:,:,is], moments, z, r, dt, spectral,
                ion_source_settings)
        end
    end
    return nothing
end

"""
"""
function source_terms_evolve_density!(pdf_out, pdf_in, dens, upar, ddens_dz, dupar_dz,
                                      moments, z, r, dt, spectral, ion_source_settings)
    # update the density
    nvpa = size(pdf_out, 1)
    @loop_r_z ir iz begin
        # calculate dt * d(n*upar)/dz / n
        factor = dt * (dens[iz,ir] * dupar_dz[iz,ir] + upar[iz,ir] * ddens_dz[iz,ir]) /
                 dens[iz,ir]
        @loop_vperp_vpa ivperp ivpa begin
            pdf_out[ivpa,ivperp,iz,ir] += pdf_in[ivpa,ivperp,iz,ir] * factor
        end
    end

    for index ∈ eachindex(ion_source_settings)
        if ion_source_settings[index].active
            @views source_density_amplitude = moments.ion.external_source_density_amplitude[:, :, index]
            @loop_r_z ir iz begin
                term = dt * source_density_amplitude[iz,ir] / dens[iz,ir]
                @loop_vperp_vpa ivperp ivpa begin
                    pdf_out[ivpa,ivperp,iz,ir] -= term * pdf_in[ivpa,ivperp,iz,ir]
                end
            end
        end
    end

    return nothing
end

"""
update the evolved pdf to account for the collisionless source terms in the kinetic equation
arising due to the re-normalization of the pdf as g = f * vth / n
"""
function source_terms_evolve_ppar_no_collisions!(pdf_out, pdf_in, dens, upar, ppar, vth,
                                                 qpar, ddens_dz, dvth_dz, dqpar_dz,
                                                 moments, z, r, dt, spectral,
                                                 ion_source_settings)
    nvpa = size(pdf_out, 1)
    @loop_r_z ir iz begin
        factor = dt * (ddens_dz[iz,ir] * upar[iz,ir] / dens[iz,ir] -
                       dvth_dz[iz,ir] * upar[iz,ir] / vth[iz,ir] -
                       0.5 * dqpar_dz[iz,ir] / ppar[iz,ir])

        @loop_vperp_vpa ivperp ivpa begin
            pdf_out[ivpa,ivperp,iz,ir] += pdf_in[ivpa,ivperp,iz,ir] * factor
        end
    end

    for index ∈ eachindex(ion_source_settings)
        if ion_source_settings[index].active
            @views source_density_amplitude = moments.ion.external_source_density_amplitude[:, :, index]
            @views source_momentum_amplitude = moments.ion.external_source_momentum_amplitude[:, :, index]
            @views source_pressure_amplitude = moments.ion.external_source_pressure_amplitude[:, :, index]
            @loop_r_z ir iz begin
                term = dt * (1.5 * source_density_amplitude[iz,ir] / dens[iz,ir] -
                             (0.5 * source_pressure_amplitude[iz,ir] +
                              source_momentum_amplitude[iz,ir]) / ppar[iz,ir])
                @loop_vperp_vpa ivperp ivpa begin
                    pdf_out[ivpa,ivperp,iz,ir] -= term * pdf_in[ivpa,ivperp,iz,ir]
                end
            end
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
    charge_exchange = collisions.reactions.charge_exchange_frequency
    ionization = collisions.reactions.ionization_frequency
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        @views pdf_out[ivpa,ivperp,iz,ir] -= 0.5*dt*pdf_in[ivpa,ivperp,iz,ir] *
            (charge_exchange
               * (dens_neutral[iz,ir]*ppar[iz,ir] - dens[iz,ir]*ppar_neutral[iz,ir]
                  - dens[iz,ir]*dens_neutral[iz,ir]
                    * (upar[iz,ir] - upar_neutral[iz,ir])^2)
               / ppar[iz,ir]
             + ionization
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
@timeit global_timer source_terms_neutral!(
                         pdf_out, fvec_in, moments, vpa, z, r, dt, spectral, composition,
                         collisions, neutral_source_settings) = begin

    @begin_sn_r_z_vzeta_vr_vz_region()

    #n_species = size(pdf_out,3)
    if moments.evolve_ppar
        @loop_sn isn begin
            @views source_terms_evolve_ppar_no_collisions_neutral!(
                pdf_out[:,:,:,:,:,isn], fvec_in.pdf_neutral[:,:,:,:,:,isn],
                fvec_in.density_neutral[:,:,isn], fvec_in.uz_neutral[:,:,isn],
                fvec_in.pz_neutral[:,:,isn], moments.neutral.vth[:,:,isn],
                moments.neutral.qz[:,:,isn], moments.neutral.ddens_dz[:,:,isn],
                moments.neutral.dvth_dz[:,:,isn], moments.neutral.dqz_dz[:,:,isn],
                moments, z, r, dt, spectral, neutral_source_settings)
            if abs(collisions.reactions.charge_exchange_frequency) > 0.0 || abs(collisions.reactions.ionization_frequency) > 0.0
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
                fvec_in.density_neutral[:,:,isn], fvec_in.uz_neutral[:,:,isn],
                moments.neutral.ddens_dz[:,:,isn], moments.neutral.duz_dz[:,:,isn],
                moments, z, r, dt, spectral, neutral_source_settings)
        end
    end
    return nothing
end

"""
"""
function source_terms_evolve_density_neutral!(pdf_out, pdf_in, dens, upar, ddens_dz,
                                              dupar_dz, moments, z, r, dt, spectral,
                                              neutral_source_settings)
    # update the density
    nvpa = size(pdf_out, 1)
    @loop_r_z ir iz begin
        # calculate dt * d(n*upar)/dz / n
        factor = dt * (dens[iz,ir] * dupar_dz[iz,ir] + upar[iz,ir] * ddens_dz[iz,ir]) /
                 dens[iz,ir]
        @loop_vzeta_vr_vz ivzeta ivr ivz begin
            pdf_out[ivz,ivr,ivzeta,iz,ir] += pdf_in[ivz,ivr,ivzeta,iz,ir] * factor
        end
    end

    for index ∈ eachindex(neutral_source_settings)
        if neutral_source_settings[index].active
            @views source_density_amplitude = moments.neutral.external_source_density_amplitude[:, :, index]
            @loop_r_z ir iz begin
                term = dt * source_density_amplitude[iz,ir] / dens[iz,ir]
                @loop_vzeta_vr_vz ivzeta ivr ivz begin
                    pdf_out[ivz,ivr,ivzeta,iz,ir] -= term * pdf_in[ivz,ivr,ivzeta,iz,ir]
                end
            end
        end
    end

    return nothing
end

"""
update the evolved pdf to account for the collisionless source terms in the kinetic equation
arising due to the re-normalization of the pdf as g = f * vth / n
"""
function source_terms_evolve_ppar_no_collisions_neutral!(pdf_out, pdf_in, dens, upar,
                                                         ppar, vth, qpar, ddens_dz,
                                                         dvth_dz, dqpar_dz, moments, z, r,
                                                         dt, spectral,
                                                         neutral_source_settings)
    nvpa = size(pdf_out, 1)
    @loop_r_z ir iz begin
        factor = dt * (ddens_dz[iz,ir] * upar[iz,ir] / dens[iz,ir] - dvth_dz[iz,ir] *
                       upar[iz,ir] / vth[iz,ir] - 0.5 * dqpar_dz[iz,ir] / ppar[iz,ir])
        @loop_vzeta_vr_vz ivzeta ivr ivz begin
            pdf_out[ivz,ivr,ivzeta,iz,ir] += pdf_in[ivz,ivr,ivzeta,iz,ir] * factor
        end
    end

    for index ∈ eachindex(neutral_source_settings)
        if neutral_source_settings[index].active
            @views source_density_amplitude = moments.neutral.external_source_density_amplitude[:, :, index]
            @views source_momentum_amplitude = moments.neutral.external_source_momentum_amplitude[:, :, index]
            @views source_pressure_amplitude = moments.neutral.external_source_pressure_amplitude[:, :, index]
            @loop_r_z ir iz begin
                term = dt * (1.5 * source_density_amplitude[iz,ir] / dens[iz,ir] -
                            (0.5 * source_pressure_amplitude[iz,ir] +
                            source_momentum_amplitude[iz,ir]) / ppar[iz,ir])
                @loop_vzeta_vr_vz ivzeta ivr ivz begin
                    pdf_out[ivz,ivr,ivzeta,iz,ir] -= term * pdf_in[ivz,ivr,ivzeta,iz,ir]
                end
            end
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
    charge_exchange = collisions.reactions.charge_exchange_frequency
    ionization = collisions.reactions.ionization_frequency
    @loop_r_z_vzeta_vr_vz ir iz ivzeta ivr ivz begin
        @views pdf_out[ivz,ivr,ivzeta,iz,ir] -= 0.5*dt*pdf_in[ivz,ivr,ivzeta,iz,ir] *
        (charge_exchange
           * (dens_ion[iz,ir]*ppar[iz,ir] - dens[iz,ir]*ppar_ion[iz,ir]
              - dens[iz,ir]*dens_ion[iz,ir]
                * (upar[iz,ir] - upar_ion[iz,ir])^2)/ppar[iz,ir]
         - 2.0*ionization*dens_ion[iz,ir])
    end
    return nothing
end

"""
advance the dfn with an arbitrary source function 
"""
@timeit global_timer source_terms_manufactured!(
                         pdf_ion_out, pdf_neutral_out, vz, vr, vzeta, vpa, vperp, z, r, t,
                         dt, composition, manufactured_source_list) = begin
    if manufactured_source_list.time_independent_sources
        # the (time-independent) manufactured source arrays
        Source_i = manufactured_source_list.Source_i_array
        Source_n = manufactured_source_list.Source_n_array

        @begin_s_r_z_region()

        @loop_s is begin
            @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                pdf_ion_out[ivpa,ivperp,iz,ir,is] += dt*Source_i[ivpa,ivperp,iz,ir]
            end
        end

        if composition.n_neutral_species > 0
            @begin_sn_r_z_region()
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

        @begin_s_r_z_region()

        @loop_s is begin
            @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
                pdf_ion_out[ivpa,ivperp,iz,ir,is] += dt*Source_i_func(vpa.grid[ivpa],vperp.grid[ivperp],z.grid[iz],r.grid[ir],t)
            end
        end

        if composition.n_neutral_species > 0
            @begin_sn_r_z_region()
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
