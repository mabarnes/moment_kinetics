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

    pdf_in = fvec_in.pdf
    n = fvec_in.density
    upar = fvec_in.upar
    vth = moments.ion.vth
    dn_dt = moments.ion.ddens_dt
    dn_dz = moments.ion.ddens_dz
    vpa_grid = vpa.grid
    if moments.evolve_ppar
        dvth_dt = moments.ion.dvth_dt
        dvth_dz = moments.ion.dvth_dz
        @loop_s_r_z is ir iz begin
            coefficient1 = -(dn_dt[iz,ir,is] + upar[iz,ir,is] * dn_dz[iz,ir,is]) / n[iz,ir,is] +
                           (dvth_dt[iz,ir,is] + upar[iz,ir,is] * dvth_dz[iz,ir,is]) / vth[iz,ir,is]
            coefficient2 = -vth[iz,ir,is] * dn_dz[iz,ir,is] / n[iz,ir,is] + dvth_dz[iz,ir,is]
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] +=
                    dt * (coefficient1 + vpa_grid[ivpa] * coefficient2) *
                    pdf_in[ivpa,ivperp,iz,ir,is]
            end
        end
    elseif moments.evolve_upar
        @loop_s_r_z is ir iz begin
            coefficient1 = -(dn_dt[iz,ir,is] + upar[iz,ir,is] * dn_dz[iz,ir,is]) / n[iz,ir,is]
            coefficient2 = -dn_dz[iz,ir,is] / n[iz,ir,is]
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] +=
                    dt * (coefficient1 + vpa_grid[ivpa] * coefficient2) *
                    pdf_in[ivpa,ivperp,iz,ir,is]
            end
        end
    elseif moments.evolve_density
        @loop_s_r_z is ir iz begin
            coefficient1 = -dn_dt[iz,ir,is] / n[iz,ir,is]
            coefficient2 = -dn_dz[iz,ir,is] / n[iz,ir,is]
            @loop_vperp_vpa ivperp ivpa begin
                pdf_out[ivpa,ivperp,iz,ir,is] +=
                    dt * (coefficient1 + vpa_grid[ivpa] * coefficient2) *
                    pdf_in[ivpa,ivperp,iz,ir,is]
            end
        end
    end
    return nothing
end

"""
calculate the source terms due to redefinition of the pdf to split off density,
flow and/or pressure, and use them to update the pdf
"""
@timeit global_timer source_terms_neutral!(
                         pdf_out, fvec_in, moments, vz, z, r, dt, spectral, composition,
                         collisions, neutral_source_settings) = begin

    @begin_sn_r_z_vzeta_vr_vz_region()

    pdf_in = fvec_in.pdf_neutral
    n = fvec_in.density_neutral
    uz = fvec_in.uz_neutral
    vth = moments.neutral.vth
    dn_dt = moments.neutral.ddens_dt
    dn_dz = moments.neutral.ddens_dz
    vz_grid = vz.grid
    if moments.evolve_ppar
        dvth_dt = moments.neutral.dvth_dt
        dvth_dz = moments.neutral.dvth_dz
        @loop_sn_r_z isn ir iz begin
            coefficient1 = -(dn_dt[iz,ir,isn] + uz[iz,ir,isn] * dn_dz[iz,ir,isn]) / n[iz,ir,isn] +
                           (dvth_dt[iz,ir,isn] + uz[iz,ir,isn] * dvth_dz[iz,ir,isn]) / vth[iz,ir,isn]
            coefficient2 = -vth[iz,ir,isn] * dn_dz[iz,ir,isn] / n[iz,ir,isn] + dvth_dz[iz,ir,isn]
            @loop_vzeta_vr_vz ivzeta ivr ivz begin
                pdf_out[ivz,ivr,ivzeta,iz,ir,isn] +=
                    dt * (coefficient1 + vz_grid[ivz] * coefficient2) *
                    pdf_in[ivz,ivr,ivzeta,iz,ir,isn]
            end
        end
    elseif moments.evolve_upar
        @loop_sn_r_z isn ir iz begin
            coefficient1 = -(dn_dt[iz,ir,isn] + uz[iz,ir,isn] * dn_dz[iz,ir,isn]) / n[iz,ir,isn]
            coefficient2 = -dn_dz[iz,ir,isn] / n[iz,ir,isn]
            @loop_vzeta_vr_vz ivzeta ivr ivz begin
                pdf_out[ivz,ivr,ivzeta,iz,ir,isn] +=
                    dt * (coefficient1 + vz_grid[ivz] * coefficient2) *
                    pdf_in[ivz,ivr,ivzeta,iz,ir,isn]
            end
        end
    elseif moments.evolve_density
        @loop_sn_r_z isn ir iz begin
            coefficient1 = -dn_dt[iz,ir,isn] / n[iz,ir,isn]
            coefficient2 = -dn_dz[iz,ir,isn] / n[iz,ir,isn]
            @loop_vzeta_vr_vz ivzeta ivr ivz begin
                pdf_out[ivz,ivr,ivzeta,iz,ir,isn] +=
                    dt * (coefficient1 + vz_grid[ivz] * coefficient2) *
                    pdf_in[ivz,ivr,ivzeta,iz,ir,isn]
            end
        end
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
