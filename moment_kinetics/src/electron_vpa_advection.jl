"""
"""
module electron_vpa_advection

export electron_vpa_advection!
export update_electron_speed_vpa!
export get_electron_vpa_advection_term

using ..looping
using ..calculus: derivative!, second_derivative!
using ..debugging
using ..gauss_legendre: gausslegendre_info
using ..jacobian_matrices
using ..moment_kinetics_structs
using ..timer_utils
using ..type_definitions

"""
calculate the wpa-advection term for the electron kinetic equation 
= (vthe / 2 pe * dpe/dz + wpa / 3 pe * dqpare/dz - wpa^2 * dvthe/dz) * df/dwpa
"""
@timeit global_timer electron_vpa_advection!(
                         pdf_out, pdf_in, density, upar, p, moments, composition, advect,
                         vpa, spectral, scratch_dummy, dt, electron_source_settings,
                         ir) = begin
    @begin_anyzv_z_vperp_region()

    # create a reference to a scratch_dummy array to store the wpa-derivative of the electron pdf
    dpdf_dvpa = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
    #d2pdf_dvpa2 = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]

    # get the updated speed along the wpa direction using the current pdf
    @views update_electron_speed_vpa!(advect, density, upar, p, moments,
                                      composition.me_over_mi, vpa.grid,
                                      electron_source_settings, ir)
    #calculate the upwind derivative of the electron pdf w.r.t. wpa
    @loop_z_vperp iz ivperp begin
        @views derivative!(dpdf_dvpa[:,ivperp,iz], pdf_in[:,ivperp,iz], vpa,
                           advect[:,ivperp,iz,ir], spectral)
    end
    #@loop_z_vperp iz ivperp begin
    #    @views second_derivative!(d2pdf_dvpa2[:,ivperp,iz], pdf_in[:,ivperp,iz], vpa, spectral)
    #end
    # calculate the advection term
    @loop_z_vperp iz ivperp begin
        @views @. pdf_out[:,ivperp,iz] += -dt * advect[:,ivperp,iz,ir] * dpdf_dvpa[:,ivperp,iz]
        #@. pdf_out[:,ivperp,iz] -= -advect[:,ivperp,iz,ir] * dpdf_dvpa[:,ivperp,iz] + 0.0001*d2pdf_dvpa2[:,ivperp,iz]
    end
    return nothing
end

"""
calculate the electron advection speed in the wpa-direction at each grid point
"""
function update_electron_speed_vpa!(advect, density, upar, p, moments, me_over_mi, vpa,
                                    electron_source_settings, ir)
    vth = @view moments.electron.vth[:,ir]
    dppar_dz = @view moments.electron.dppar_dz[:,ir]
    dqpar_dz = @view moments.electron.dqpar_dz[:,ir]
    dvth_dz = @view moments.electron.dvth_dz[:,ir]
    speed = @view advect[:,:,:,ir]
    # calculate the advection speed in wpa
    @loop_z_vperp_vpa iz ivperp ivpa begin
        speed[ivpa,ivperp,iz] = ((0.5 * vth[iz] * dppar_dz[iz] + vpa[ivpa] * dqpar_dz[iz] / 3.0) / p[iz]
                                 - vpa[ivpa]^2 * dvth_dz[iz])
    end

    for index ∈ eachindex(electron_source_settings)
        if electron_source_settings[index].active
            @views source_density_amplitude = moments.electron.external_source_density_amplitude[:, ir, index]
            @views source_momentum_amplitude = moments.electron.external_source_momentum_amplitude[:, ir, index]
            @views source_pressure_amplitude = moments.electron.external_source_pressure_amplitude[:, ir, index]
            @loop_z iz begin
                term1 = (source_density_amplitude[iz] * upar[iz] -
                         source_momentum_amplitude[iz] / me_over_mi) / (density[iz] * vth[iz])
                term2_over_vpa =
                    -0.5 * source_pressure_amplitude[iz] / p[iz] +
                     0.5 * source_density_amplitude[iz] / density[iz]
                @loop_vperp ivperp begin
                    @. speed[:,ivperp,iz] += term1 + vpa * term2_over_vpa
                end
            end
        end
    end
    return nothing
end
# Alternative version with loop over r is used for adaptive timestep update
function update_electron_speed_vpa!(advect, density, upar, p, moments, me_over_mi, vpa,
                                    electron_source_settings)
    @begin_r_anyzv_region()
    @loop_r ir begin
        @views update_electron_speed_vpa!(advect, density[:,ir], upar[:,ir], p[:,ir],
                                          moments, me_over_mi, vpa,
                                          electron_source_settings, ir)
    end
    return nothing
end

function get_electron_vpa_advection_term(sub_terms::ElectronSubTerms)
    # Get contribution from vpa-advection term
    #   (
    #     (0.5*vth*dppar_dz + wpa*dqpar_dz/3)/p
    #     - wpa^2*dvth_dz
    #     + ∑( (souce_density_amplitude*upar - source_momentum_amplitude/me)/dens/vth
    #           + wpa*(-0.5*source_pressure_amplitude/p + 0.5*source_density_amplitude/dens) )
    #   ) * df/dwpa
    # to Jacobian matrix

    me = sub_terms.me
    n = sub_terms.n
    u = sub_terms.u
    p = sub_terms.p
    vth = sub_terms.vth
    dppar_dz = sub_terms.dppar_dz
    dvth_dz = sub_terms.dvth_dz
    wpa = sub_terms.wpa
    dq_dz = sub_terms.dq_dz
    df_dvpa = sub_terms.df_dvpa

    term = (
            0.5 * vth * dppar_dz + 1.0 / 3.0 * wpa * dq_dz
           ) * p^(-1) -
           wpa^2 * dvth_dz

    for (density_source, momentum_source, pressure_source) ∈ zip(sub_terms.density_source, sub_terms.momentum_source, sub_terms.pressure_source)
        term += (
                 (density_source * u - momentum_source * (1.0/me))
                 * n^(1) * vth^(-1)
                 + wpa * 0.5 * (- pressure_source * p^(-1)
                                + density_source * n^(-1)
                               )
                )
    end

    term *= df_dvpa

    return term
end

end
