"""
"""
module electron_z_advection

export electron_z_advection!
export update_electron_speed_z!

using ..advection: advance_f_df_precomputed!
using ..chebyshev: chebyshev_info
using ..looping
using ..derivatives: derivative_z_pdf_vpavperpz!
using ..calculus: second_derivative!

"""
calculate the z-advection term for the electron kinetic equation = wpa * vthe * df/dz
"""
function electron_z_advection!(pdf_out, pdf_in, upar, vth, advect, z, vpa, spectral,
                               scratch_dummy, dt, ir)
    begin_vperp_vpa_region()

    # create a pointer to a scratch_dummy array to store the z-derivative of the electron pdf
    dpdf_dz = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
    d2pdf_dz2 = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]
    begin_r_vperp_vpa_region()
    # get the updated speed along the z direction using the current pdf
    @views update_electron_speed_z!(advect[1], upar, vth, vpa, ir)
    # update adv_fac -- note that there is no factor of dt here because
    # in some cases the electron kinetic equation is solved as a steady-state equation iteratively
    @loop_vperp_vpa ivperp ivpa begin
        @views advect[1].adv_fac[:,ivpa,ivperp,ir] = -advect[1].speed[:,ivpa,ivperp,ir]
    end
    #calculate the upwind derivative
    @views derivative_z_pdf_vpavperpz!(
               dpdf_dz, pdf_in, advect[1].adv_fac[:,:,:,ir],
               scratch_dummy.buffer_vpavperpr_1[:,:,ir],
               scratch_dummy.buffer_vpavperpr_2[:,:,ir],
               scratch_dummy.buffer_vpavperpr_3[:,:,ir],
               scratch_dummy.buffer_vpavperpr_4[:,:,ir],
               scratch_dummy.buffer_vpavperpr_5[:,:,ir],
               scratch_dummy.buffer_vpavperpr_6[:,:,ir], spectral, z)
    #@loop_vperp_vpa ivperp ivpa begin
    #    @views second_derivative!(d2pdf_dz2[ivpa,ivperp,:], pdf_in[ivpa,ivperp,:], z, spectral)
    #end
    # calculate the advection term
    @loop_z_vperp_vpa iz ivperp ivpa begin
        pdf_out[ivpa,ivperp,iz] += dt * advect[1].adv_fac[iz,ivpa,ivperp,ir] * dpdf_dz[ivpa,ivperp,iz]
        #pdf_out[ivpa,ivperp,iz] += dt * advect[1].adv_fac[iz,ivpa,ivperp,ir] * dpdf_dz[ivpa,ivperp,iz] + 0.0001*d2pdf_dz2[ivpa,ivperp,iz]
    end
    return nothing
end

"""
calculate the electron advection speed in the z-direction at each grid point
"""
function update_electron_speed_z!(advect, upar, vth, vpa, ir)
    # the electron advection speed in z is v_par = w_par * v_the
    @loop_vperp_vpa ivperp ivpa begin
        #@. @views advect.speed[:,ivpa,ivperp,ir] = vpa[ivpa] * vth
        @. @views advect.speed[:,ivpa,ivperp,ir] = vpa[ivpa] * vth + upar
    end
    return nothing
end
# Alternative version with loop over r is used for adaptive timestep update
function update_electron_speed_z!(advect, upar, vth, vpa)
    @loop_r ir begin
        @views update_electron_speed_z!(advect, upar[:,ir], vth[:,ir], vpa, ir)
    end
    return nothing
end

end
