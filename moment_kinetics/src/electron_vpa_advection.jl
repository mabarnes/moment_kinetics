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
                                 electron_source_settings, ir)
    begin_z_vperp_region()

    # create a reference to a scratch_dummy array to store the wpa-derivative of the electron pdf
    dpdf_dvpa = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
    #d2pdf_dvpa2 = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]

    # get the updated speed along the wpa direction using the current pdf
    @views update_electron_speed_vpa!(advect[1], density, upar, ppar, moments, vpa.grid,
                                      electron_source_settings, ir)
    # update adv_fac
    @loop_z_vperp iz ivperp begin
        @views @. advect[1].adv_fac[:,ivperp,iz,ir] = -advect[1].speed[:,ivperp,iz,ir]
    end
    #calculate the upwind derivative of the electron pdf w.r.t. wpa
    @loop_z_vperp iz ivperp begin
        @views derivative!(dpdf_dvpa[:,ivperp,iz], pdf_in[:,ivperp,iz], vpa,
                           advect[1].adv_fac[:,ivperp,iz,ir], spectral)
    end
    #@loop_z_vperp iz ivperp begin
    #    @views second_derivative!(d2pdf_dvpa2[:,ivperp,iz], pdf_in[:,ivperp,iz], vpa, spectral)
    #end
    # calculate the advection term
    @loop_z_vperp iz ivperp begin
        @. pdf_out[:,ivperp,iz] += dt * advect[1].adv_fac[:,ivperp,iz,ir] * dpdf_dvpa[:,ivperp,iz]
        #@. pdf_out[:,ivperp,iz] -= advect[1].adv_fac[:,ivperp,iz,ir] * dpdf_dvpa[:,ivperp,iz] + 0.0001*d2pdf_dvpa2[:,ivperp,iz]
    end
    return nothing
end

"""
calculate the electron advection speed in the wpa-direction at each grid point
"""
function update_electron_speed_vpa!(advect, density, upar, ppar, moments, vpa,
                                    electron_source_settings, ir)
    vth = @view moments.electron.vth[:,ir]
    dppar_dz = @view moments.electron.dppar_dz[:,ir]
    dqpar_dz = @view moments.electron.dqpar_dz[:,ir]
    dvth_dz = @view moments.electron.dvth_dz[:,ir]
    # calculate the advection speed in wpa
    @loop_z_vperp_vpa iz ivperp ivpa begin
        advect.speed[ivpa,ivperp,iz,ir] = ((vth[iz] * dppar_dz[iz] + vpa[ivpa] * dqpar_dz[iz])
                                           / (2 * ppar[iz]) - vpa[ivpa]^2 * dvth_dz[iz])
    end
    if electron_source_settings.active
        source_density_amplitude = @view moments.electron.external_source_density_amplitude[:,ir]
        source_momentum_amplitude = @view moments.electron.external_source_momentum_amplitude[:,ir]
        source_pressure_amplitude = @view moments.electron.external_source_pressure_amplitude[:,ir]
        @loop_z iz begin
            term1 = source_density_amplitude[iz] * upar[iz]/(density[iz]*vth[iz])
            term2_over_vpa =
                -0.5 * (source_pressure_amplitude[iz] +
                        2.0 * upar[iz] * source_momentum_amplitude[iz]) /
                       ppar[iz] +
                0.5 * source_density_amplitude[iz] / density[iz]
            @loop_vperp_vpa ivperp ivpa begin
                advect.speed[ivpa,ivperp,iz,ir] += term1 + vpa[ivpa] * term2_over_vpa
            end
        end
    end
    return nothing
end
# Alternative version with loop over r is used for adaptive timestep update
function update_electron_speed_vpa!(advect, density, upar, ppar, moments, vpa,
                                    electron_source_settings)
    @loop_r ir begin
        @views update_electron_speed_vpa!(advect, density[:,ir], upar[:,ir], ppar[:,ir],
                                          moments, vpa, electron_source_settings, ir)
    end
    return nothing
end

end
