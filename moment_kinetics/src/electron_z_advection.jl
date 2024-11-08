"""
"""
module electron_z_advection

export electron_z_advection!
export update_electron_speed_z!
export add_electron_z_advection_to_Jacobian!

using ..advection: advance_f_df_precomputed!
using ..boundary_conditions: skip_f_electron_bc_points_in_Jacobian
using ..chebyshev: chebyshev_info
using ..gauss_legendre: gausslegendre_info
using ..looping
using ..timer_utils
using ..derivatives: derivative_z_pdf_vpavperpz!
using ..calculus: second_derivative!

"""
calculate the z-advection term for the electron kinetic equation = wpa * vthe * df/dz
"""
@timeit global_timer electron_z_advection!(
                         pdf_out, pdf_in, upar, vth, advect, z, vpa, spectral,
                         scratch_dummy, dt, ir) = begin
    begin_vperp_vpa_region()

    # create a pointer to a scratch_dummy array to store the z-derivative of the electron pdf
    dpdf_dz = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
    d2pdf_dz2 = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]
    begin_vperp_vpa_region()
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
    begin_z_vperp_vpa_region()
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

function add_electron_z_advection_to_Jacobian!(jacobian_matrix, f, dens, upar, ppar, vth,
                                               me, z, vperp, vpa, z_spectral, z_advect,
                                               scratch_dummy, dt, ir; f_offset=0,
                                               ppar_offset=0)
    if f_offset == ppar_offset
        error("Got f_offset=$f_offset the same as ppar_offset=$ppar_offset. f and ppar "
              * "cannot be in same place in state vector.")
    end
    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2)
    @boundscheck size(jacobian_matrix, 1) ≥ f_offset + z.n * vperp.n * vpa.n
    @boundscheck size(jacobian_matrix, 1) ≥ ppar_offset + z.n

    v_size = vperp.n * vpa.n

    dpdf_dz = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]

    begin_vperp_vpa_region()
    update_electron_speed_z!(z_advect[1], upar, vth, vpa.grid, ir)
    z_speed_array = @view z_advect[1].speed[:,:,:,1]

    @loop_vperp_vpa ivperp ivpa begin
        @views z_advect[1].adv_fac[:,ivpa,ivperp,ir] = -z_speed_array[:,ivpa,ivperp]
    end
    #calculate the upwind derivative
    @views derivative_z_pdf_vpavperpz!(dpdf_dz, f, z_advect[1].adv_fac[:,:,:,ir],
                                       scratch_dummy.buffer_vpavperpr_1[:,:,ir],
                                       scratch_dummy.buffer_vpavperpr_2[:,:,ir],
                                       scratch_dummy.buffer_vpavperpr_3[:,:,ir],
                                       scratch_dummy.buffer_vpavperpr_4[:,:,ir],
                                       scratch_dummy.buffer_vpavperpr_5[:,:,ir],
                                       scratch_dummy.buffer_vpavperpr_6[:,:,ir],
                                       z_spectral, z)

    if !isa(z_spectral, gausslegendre_info)
        error("Only gausslegendre_pseudospectral z-coordinate type is supported by "
              * "add_electron_z_advection_to_Jacobian!() preconditioner because we need "
              * "differentiation matrices.")
    end
    z_Dmat = z_spectral.lobatto.Dmat
    z_element_scale = z.element_scale

    begin_z_vperp_vpa_region()
    @loop_z_vperp_vpa iz ivperp ivpa begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa,
                                                 z_speed_array)
            continue
        end

        # Rows corresponding to pdf_electron
        row = (iz - 1) * v_size + (ivperp - 1) * vpa.n + ivpa + f_offset
        v_remainder = (ivperp - 1) * vpa.n + ivpa + f_offset

        ielement_z = z.ielement[iz]
        igrid_z = z.igrid[iz]
        icolumn_min_z = z.imin[ielement_z] - (ielement_z != 1)
        icolumn_max_z = z.imax[ielement_z]

        z_speed = z_speed_array[iz,ivpa,ivperp]

        # Contributions from (w_∥*vth + upar)*dg/dz
        if ielement_z == 1 && igrid_z == 1
            jacobian_matrix[row,(icolumn_min_z-1)*v_size+v_remainder:v_size:(icolumn_max_z-1)*v_size+v_remainder] .+=
            dt * z_speed * z_Dmat[1,:] ./ z_element_scale[ielement_z]
        elseif ielement_z == z.nelement_local && igrid_z == z.ngrid
            jacobian_matrix[row,(icolumn_min_z-1)*v_size+v_remainder:v_size:(icolumn_max_z-1)*v_size+v_remainder] .+=
            dt * z_speed * z_Dmat[end,:] ./ z_element_scale[ielement_z]
        elseif igrid_z == z.ngrid
            # Note igrid_z is only ever 1 when ielement_z==1, because
            # of the way element boundaries are counted.
            icolumn_min_z_next = z.imin[ielement_z+1] - 1
            icolumn_max_z_next = z.imax[ielement_z+1]
            if z_speed < 0.0
                jacobian_matrix[row,(icolumn_min_z_next-1)*v_size+v_remainder:v_size:(icolumn_max_z_next-1)*v_size+v_remainder] .+=
                dt * z_speed * z_Dmat[1,:] ./ z_element_scale[ielement_z+1]
            elseif z_speed > 0.0
                jacobian_matrix[row,(icolumn_min_z-1)*v_size+v_remainder:v_size:(icolumn_max_z-1)*v_size+v_remainder] .+=
                dt * z_speed * z_Dmat[end,:] ./ z_element_scale[ielement_z]
            else
                jacobian_matrix[row,(icolumn_min_z-1)*v_size+v_remainder:v_size:(icolumn_max_z-1)*v_size+v_remainder] .+=
                dt * z_speed * 0.5 * z_Dmat[end,:] ./ z_element_scale[ielement_z]
                jacobian_matrix[row,(icolumn_min_z_next-1)*v_size+v_remainder:v_size:(icolumn_max_z_next-1)*v_size+v_remainder] .+=
                dt * z_speed * 0.5 * z_Dmat[1,:] ./ z_element_scale[ielement_z+1]
            end
        else
            jacobian_matrix[row,(icolumn_min_z-1)*v_size+v_remainder:v_size:(icolumn_max_z-1)*v_size+v_remainder] .+=
            dt * z_speed * z_Dmat[igrid_z,:] ./ z_element_scale[ielement_z]
        end
        # vth = sqrt(2*p/n/me)
        # so d(vth)/d(ppar) = 1/n/me/sqrt(2*p/n/me) = 1/n/me/vth
        # and d(w_∥*vth*dg/dz)/d(ppar) = 1/n/me/vth*w_∥*dg/dz
        jacobian_matrix[row,ppar_offset+iz] += dt / dens[iz] / me / vth[iz] * vpa.grid[ivpa] * dpdf_dz[ivpa,ivperp,iz]
    end

    return nothing
end

end
