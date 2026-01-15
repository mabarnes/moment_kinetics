"""
"""
module electron_z_advection

export electron_z_advection!
export update_electron_speed_z!
export get_electron_z_advection_term

using ..advection: advance_f_df_precomputed!
using ..chebyshev: chebyshev_info
using ..debugging
using ..gauss_legendre: gausslegendre_info
using ..jacobian_matrices
using ..looping
using ..moment_kinetics_structs
using ..timer_utils
using ..type_definitions
using ..derivatives: derivative_z_pdf_vpavperpz!
using ..calculus: second_derivative!

"""
calculate the z-advection term for the electron kinetic equation = wpa * vthe * df/dz
"""
@timeit global_timer electron_z_advection!(
                         pdf_out, pdf_in, upar, vth, advect, z, vpa, spectral,
                         scratch_dummy, dt, ir) = begin
    # create a pointer to a scratch_dummy array to store the z-derivative of the electron pdf
    dpdf_dz = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
    #d2pdf_dz2 = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]

    # get the updated speed along the z direction using the current pdf
    @views update_electron_speed_z!(advect[:,:,:,ir], upar, vth, vpa)

    @begin_anyzv_vperp_vpa_region()

    #calculate the upwind derivative
    @views derivative_z_pdf_vpavperpz!(
               dpdf_dz, pdf_in, advect[:,:,:,ir],
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
    @begin_anyzv_z_vperp_vpa_region()
    @loop_z_vperp_vpa iz ivperp ivpa begin
        pdf_out[ivpa,ivperp,iz] += -dt * advect[ivpa,ivperp,iz,ir] * dpdf_dz[ivpa,ivperp,iz]
        #pdf_out[ivpa,ivperp,iz] += -dt * advect[iz,ivpa,ivperp,ir] * dpdf_dz[ivpa,ivperp,iz] + 0.0001*d2pdf_dz2[ivpa,ivperp,iz]
    end
    return nothing
end

"""
calculate the electron advection speed in the z-direction at each grid point
"""
function update_electron_speed_z!(advect::AbstractArray{mk_float,3}, upar, vth, vpa)
    # the electron advection speed in z is v_par = w_par * v_the
    @begin_anyzv_z_vperp_region()
    @loop_z iz begin
        this_vth = vth[iz]
        @loop_vperp ivperp begin
            @. advect[:,ivperp,iz] = vpa * this_vth
        end
    end
    return nothing
end
# Alternative version with loop over r is used for adaptive timestep update
function update_electron_speed_z!(advect::AbstractArray{mk_float,4}, upar, vth, vpa)
    @begin_r_anyzv_region()
    @loop_r ir begin
        @views update_electron_speed_z!(advect[:,:,:,ir], upar[:,ir], vth[:,ir], vpa)
    end
    return nothing
end

function get_electron_z_advection_term(sub_terms::ElectronSubTerms)
    # Get contribution from z-advection term
    #   wpa * sqrt(2/me*p/n) * df/dz
    # to Jacobian matrix
    me = sub_terms.me
    n = sub_terms.n
    p = sub_terms.p
    wpa = sub_terms.wpa
    df_dz = sub_terms.df_dz

    term = sqrt(2.0 / me) * wpa * n^(-0.5) * p^0.5 * df_dz

    return term
end

end
