"""
"""
module electron_z_advection

export electron_z_advection!
export update_electron_speed_z!

using ..advection: advance_f_df_precomputed!
using ..chebyshev: chebyshev_info
using ..looping
using ..derivatives: derivative_z!
using ..calculus: second_derivative!, derivative!

"""
calculate the z-advection term for the electron kinetic equation = wpa * vthe * df/dz
"""
function electron_z_advection!(pdf_out, pdf_in, upar, vth, advect, z, vpa, spectral,
                               scratch_dummy, dt)
    begin_r_vperp_vpa_region()

    # create a pointer to a scratch_dummy array to store the z-derivative of the electron pdf
    dpdf_dz = scratch_dummy.buffer_vpavperpzr_1
    d2pdf_dz2 = scratch_dummy.buffer_vpavperpzr_2
    begin_r_vperp_vpa_region()
    # get the updated speed along the z direction using the current pdf
    @views update_electron_speed_z!(advect[1], upar, vth, vpa)
    # update adv_fac -- note that there is no factor of dt here because
    # in some cases the electron kinetic equation is solved as a steady-state equation iteratively
    @loop_r_vperp_vpa ir ivperp ivpa begin
        @views advect[1].adv_fac[:,ivpa,ivperp,ir] = -advect[1].speed[:,ivpa,ivperp,ir]
    end
    #calculate the upwind derivative
    derivative_z!(dpdf_dz, pdf_in,
                  advect, scratch_dummy.buffer_vpavperpr_1,
                  scratch_dummy.buffer_vpavperpr_2, scratch_dummy.buffer_vpavperpr_3,
                  scratch_dummy.buffer_vpavperpr_4, scratch_dummy.buffer_vpavperpr_5,
                  scratch_dummy.buffer_vpavperpr_6, spectral, z)
    #@loop_r_vperp_vpa ir ivperp ivpa begin
    #    @views second_derivative!(d2pdf_dz2[ivpa,ivperp,:,ir], pdf[ivpa,ivperp,:,ir], z, spectral)
    #end
    # calculate the advection term
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        pdf_out[ivpa,ivperp,iz,ir] += dt * advect[1].adv_fac[iz,ivpa,ivperp,ir] * dpdf_dz[ivpa,ivperp,iz,ir]
        #pdf_out[ivpa,ivperp,iz,ir] += dt * advect[1].adv_fac[iz,ivpa,ivperp,ir] * dpdf_dz[ivpa,ivperp,iz,ir] + 0.0001*d2pdf_dz2[ivpa,ivperp,iz,ir]
    end
    return nothing
end

"""
calculate the electron advection speed in the z-direction at each grid point
"""
function update_electron_speed_z!(advect, upar, vth, vpa)
    # the electron advection speed in z is v_par = w_par * v_the
    @loop_r_vperp_vpa ir ivperp ivpa begin
        #@. @views advect.speed[:,ivpa,ivperp,ir] = vpa[ivpa] * vth[:,ir]
        @. @views advect.speed[:,ivpa,ivperp,ir] = vpa[ivpa] * vth[:,ir] + upar[:,ir]
    end
    return nothing
end

end
