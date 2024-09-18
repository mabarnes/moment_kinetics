"""
"""
module electron_vpa_advection

export electron_vpa_advection!
export update_electron_speed_vpa!

using ..looping
using ..calculus: derivative!, second_derivative!

"""
calculate the wpa-advection term for the electron kinetic equation 
= (vthe / 2 ppare * dppare/dz + wpa / 2 ppare * dqpare/dz - wpa^2 * dvthe/dz) * df/dwpa
"""
function electron_vpa_advection!(pdf_out, pdf_in, density, upar, ppar, moments, advect,
                                 vpa, spectral, scratch_dummy, dt,
                                 electron_source_settings)
    begin_r_z_vperp_region()

    # create a reference to a scratch_dummy array to store the wpa-derivative of the electron pdf
    dpdf_dvpa = scratch_dummy.buffer_vpavperpzr_1
    #d2pdf_dvpa2 = scratch_dummy.buffer_vpavperpzr_2
    begin_r_z_vperp_region()
    # get the updated speed along the wpa direction using the current pdf
    @views update_electron_speed_vpa!(advect[1], density, upar, ppar, moments, vpa.grid,
                                      electron_source_settings)
    # update adv_fac -- note that there is no factor of dt here because
    # in some cases the electron kinetic equation is solved as a steady-state equation iteratively
    @loop_r_z_vperp ir iz ivperp begin
        @views @. advect[1].adv_fac[:,ivperp,iz,ir] = -advect[1].speed[:,ivperp,iz,ir]
    end
    #calculate the upwind derivative of the electron pdf w.r.t. wpa
    @loop_r_z_vperp ir iz ivperp begin
        @views derivative!(dpdf_dvpa[:,ivperp,iz,ir], pdf_in[:,ivperp,iz,ir], vpa,
                           advect[1].adv_fac[:,ivperp,iz,ir], spectral)
    end
    #@loop_r_z_vperp ir iz ivperp begin
    #    @views second_derivative!(d2pdf_dvpa2[:,ivperp,iz,ir], pdf_in[:,ivperp,iz,ir], vpa, spectral)
    #end
    # calculate the advection term
    @loop_r_z_vperp ir iz ivperp begin
        @. pdf_out[:,ivperp,iz,ir] += dt * advect[1].adv_fac[:,ivperp,iz,ir] * dpdf_dvpa[:,ivperp,iz,ir]
        #@. pdf_out[:,ivperp,iz,ir] -= advect[1].adv_fac[:,ivperp,iz,ir] * dpdf_dvpa[:,ivperp,iz,ir] + 0.0001*d2pdf_dvpa2[:,ivperp,iz,ir]
    end
    #@loop_vpa ivpa begin
    #    println("electron_vpa_advection: ", pdf_out[ivpa,1,10,1], " vpa: ", vpa.grid[ivpa], " dpdf_dvpa: ", dpdf_dvpa[ivpa,1,10,1],
    #        " pdf: ", pdf[ivpa,1,10,1])
    #end
    #exit()
    return nothing
end

"""
calculate the electron advection speed in the wpa-direction at each grid point
"""
function update_electron_speed_vpa!(advect, density, upar, ppar, moments, vpa,
                                    electron_source_settings)
    vth = moments.electron.vth
    dppar_dz = moments.electron.dppar_dz
    dqpar_dz = moments.electron.dqpar_dz
    dvth_dz = moments.electron.dvth_dz
    # calculate the advection speed in wpa
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        # TMP FOR TESTING
        #advect.speed[ivpa,ivperp,iz,ir] = vth[iz,ir] * dppar_dz[iz,ir] / (2 * ppar[iz,ir])
        advect.speed[ivpa,ivperp,iz,ir] = ((vth[iz,ir] * dppar_dz[iz,ir] + vpa[ivpa] * dqpar_dz[iz,ir]) 
                                           / (2 * ppar[iz,ir]) - vpa[ivpa]^2 * dvth_dz[iz,ir])
    end

    for index ∈ eachindex(electron_source_settings)
        if electron_source_settings[index].active
            @views source_density_amplitude = moments.electron.external_source_density_amplitude[:, :, index]
            @views source_momentum_amplitude = moments.electron.external_source_momentum_amplitude[:, :, index]
            @views source_pressure_amplitude = moments.electron.external_source_pressure_amplitude[:, :, index]
            @loop_r_z ir iz begin
                term1 = source_density_amplitude[iz,ir] * upar[iz,ir]/(density[iz,ir]*vth[iz,ir])
                term2_over_vpa =
                    -0.5 * (source_pressure_amplitude[iz,ir] +
                            2.0 * upar[iz,ir] * source_momentum_amplitude[iz,ir]) /
                        ppar[iz,ir] +
                    0.5 * source_density_amplitude[iz,ir] / density[iz,ir]
                @loop_vperp_vpa ivperp ivpa begin
                    advect.speed[ivpa,ivperp,iz,ir] += term1 + vpa[ivpa] * term2_over_vpa
                end
            end
        end
    end
    return nothing
end

end
