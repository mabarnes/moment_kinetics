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
function electron_vpa_advection!(advection_term, pdf, ppar, vth, dppar_dz, dqpar_dz, dvth_dz, 
                                 advect, vpa, spectral, scratch_dummy)
    # create a reference to a scratch_dummy array to store the wpa-derivative of the electron pdf
    dpdf_dvpa = scratch_dummy.buffer_vpavperpzr_1
    d2pdf_dvpa2 = scratch_dummy.buffer_vpavperpzr_2
    begin_r_z_vperp_region()
    # get the updated speed along the wpa direction using the current pdf
    @views update_electron_speed_vpa!(advect[1], ppar[:,:], vth[:,:], dppar_dz[:,:], dqpar_dz[:,:], dvth_dz[:,:], vpa.grid)
    # update adv_fac -- note that there is no factor of dt here because
    # in some cases the electron kinetic equation is solved as a steady-state equation iteratively
    @views @. advect[1].adv_fac[:,:,:,:] = -advect[1].speed[:,:,:,:]
    #calculate the upwind derivative of the electron pdf w.r.t. wpa
    @loop_r_z_vperp ir iz ivperp begin
        @views derivative!(dpdf_dvpa[:,ivperp,iz,ir], pdf[:,ivperp,iz,ir], vpa, advect[1].adv_fac[:,ivperp,iz,ir], spectral)
    end
    #@loop_r_z_vperp ir iz ivperp begin
    #    @views second_derivative!(d2pdf_dvpa2[:,ivperp,iz,ir], pdf[:,ivperp,iz,ir], vpa, spectral)
    #end
    # calculate the advection term
    @loop_vpa ivpa begin
        @. advection_term[ivpa,:,:,:] -= advect[1].adv_fac[ivpa,:,:,:] * dpdf_dvpa[ivpa,:,:,:]
        #@. advection_term[ivpa,:,:,:] -= advect[1].adv_fac[ivpa,:,:,:] * dpdf_dvpa[ivpa,:,:,:] + 0.0001*d2pdf_dvpa2[ivpa,:,:,:]
    end
    #@loop_vpa ivpa begin
    #    println("electron_vpa_advection: ", advection_term[ivpa,1,10,1], " vpa: ", vpa.grid[ivpa], " dpdf_dvpa: ", dpdf_dvpa[ivpa,1,10,1],
    #        " pdf: ", pdf[ivpa,1,10,1])
    #end
    #exit()
    return nothing
end

"""
calculate the electron advection speed in the wpa-direction at each grid point
"""
function update_electron_speed_vpa!(advect, ppar, vth, dppar_dz, dqpar_dz, dvth_dz, vpa)
    # calculate the advection speed in wpa
    @loop_r_z_vperp_vpa ir iz ivperp ivpa begin
        # TMP FOR TESTING
        #advect.speed[ivpa,ivperp,iz,ir] = vth[iz,ir] * dppar_dz[iz,ir] / (2 * ppar[iz,ir])
        advect.speed[ivpa,ivperp,iz,ir] = ((vth[iz,ir] * dppar_dz[iz,ir] + vpa[ivpa] * dqpar_dz[iz,ir]) 
                                           / (2 * ppar[iz,ir]) - vpa[ivpa]^2 * dvth_dz[iz,ir])
    end
    return nothing
end

end
