"""
"""
module electron_z_advection

export electron_z_advection!
export update_electron_speed_z!
export add_electron_z_advection_to_Jacobian!

using ..advection: advance_f_df_precomputed!
using ..boundary_conditions: skip_f_electron_bc_points_in_Jacobian
using ..chebyshev: chebyshev_info
using ..debugging
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
    @begin_anyzv_vperp_vpa_region()

    adv_fac = advect[1].adv_fac
    speed = advect[1].speed

    # create a pointer to a scratch_dummy array to store the z-derivative of the electron pdf
    dpdf_dz = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
    d2pdf_dz2 = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]
    @begin_anyzv_vperp_vpa_region()
    # get the updated speed along the z direction using the current pdf
    @views update_electron_speed_z!(advect[1], upar, vth, vpa, ir)
    # update adv_fac -- note that there is no factor of dt here because
    # in some cases the electron kinetic equation is solved as a steady-state equation iteratively
    @loop_vperp_vpa ivperp ivpa begin
        @views @. adv_fac[:,ivpa,ivperp,ir] = -speed[:,ivpa,ivperp,ir]
    end
    #calculate the upwind derivative
    @views derivative_z_pdf_vpavperpz!(
               dpdf_dz, pdf_in, adv_fac[:,:,:,ir],
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
        pdf_out[ivpa,ivperp,iz] += dt * adv_fac[iz,ivpa,ivperp,ir] * dpdf_dz[ivpa,ivperp,iz]
        #pdf_out[ivpa,ivperp,iz] += dt * adv_fac[iz,ivpa,ivperp,ir] * dpdf_dz[ivpa,ivperp,iz] + 0.0001*d2pdf_dz2[ivpa,ivperp,iz]
    end
    return nothing
end

"""
calculate the electron advection speed in the z-direction at each grid point
"""
function update_electron_speed_z!(advect, upar, vth, vpa, ir)
    # the electron advection speed in z is v_par = w_par * v_the
    speed = advect.speed
    @begin_anyzv_vperp_vpa_region()
    @loop_vperp_vpa ivperp ivpa begin
        @. speed[:,ivpa,ivperp,ir] = vpa[ivpa] * vth
    end
    return nothing
end
# Alternative version with loop over r is used for adaptive timestep update
function update_electron_speed_z!(advect, upar, vth, vpa)
    @begin_r_anyzv_region()
    @loop_r ir begin
        @views update_electron_speed_z!(advect, upar[:,ir], vth[:,ir], vpa, ir)
    end
    return nothing
end

function add_electron_z_advection_to_Jacobian!(jacobian_matrix, f, dens, upar, p, vth,
                                               dpdf_dz, me, z, vperp, vpa, z_spectral,
                                               z_advect, z_speed, scratch_dummy, dt, ir,
                                               include=:all; f_offset=0, p_offset=0)
    if f_offset == p_offset
        error("Got f_offset=$f_offset the same as p_offset=$p_offset. f and p "
              * "cannot be in same place in state vector.")
    end
    @debug_consistency_checks size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @debug_consistency_checks size(jacobian_matrix, 1) ≥ f_offset + z.n * vperp.n * vpa.n || error("f_offset=$f_offset is too big")
    @debug_consistency_checks size(jacobian_matrix, 1) ≥ p_offset + z.n || error("p_offset=$p_offset is too big")
    @debug_consistency_checks include ∈ (:all, :explicit_z, :explicit_v) || error("Unexpected value for include=$include")

    v_size = vperp.n * vpa.n

    if !isa(z_spectral, gausslegendre_info)
        error("Only gausslegendre_pseudospectral z-coordinate type is supported by "
              * "add_electron_z_advection_to_Jacobian!() preconditioner because we need "
              * "differentiation matrices.")
    end
    z_Dmat = z_spectral.lobatto.Dmat
    z_element_scale = z.element_scale

    @begin_anyzv_z_vperp_vpa_region()
    @loop_z_vperp_vpa iz ivperp ivpa begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = (iz - 1) * v_size + (ivperp - 1) * vpa.n + ivpa + f_offset
        v_remainder = (ivperp - 1) * vpa.n + ivpa + f_offset

        ielement_z = z.ielement[iz]
        igrid_z = z.igrid[iz]
        icolumn_min_z = z.imin[ielement_z] - (ielement_z != 1)
        icolumn_max_z = z.imax[ielement_z]

        this_z_speed = z_speed[iz,ivpa,ivperp]

        # Contributions from (w_∥*vth + upar)*dg/dz
        if include ∈ (:all, :explicit_z)
            if ielement_z == 1 && igrid_z == 1
                jacobian_matrix[row,(icolumn_min_z-1)*v_size+v_remainder:v_size:(icolumn_max_z-1)*v_size+v_remainder] .+=
                dt * this_z_speed * z_Dmat[1,:] ./ z_element_scale[ielement_z]
            elseif ielement_z == z.nelement_local && igrid_z == z.ngrid
                jacobian_matrix[row,(icolumn_min_z-1)*v_size+v_remainder:v_size:(icolumn_max_z-1)*v_size+v_remainder] .+=
                dt * this_z_speed * z_Dmat[end,:] ./ z_element_scale[ielement_z]
            elseif igrid_z == z.ngrid
                # Note igrid_z is only ever 1 when ielement_z==1, because
                # of the way element boundaries are counted.
                icolumn_min_z_next = z.imin[ielement_z+1] - 1
                icolumn_max_z_next = z.imax[ielement_z+1]
                if this_z_speed < 0.0
                    jacobian_matrix[row,(icolumn_min_z_next-1)*v_size+v_remainder:v_size:(icolumn_max_z_next-1)*v_size+v_remainder] .+=
                    dt * this_z_speed * z_Dmat[1,:] ./ z_element_scale[ielement_z+1]
                elseif this_z_speed > 0.0
                    jacobian_matrix[row,(icolumn_min_z-1)*v_size+v_remainder:v_size:(icolumn_max_z-1)*v_size+v_remainder] .+=
                    dt * this_z_speed * z_Dmat[end,:] ./ z_element_scale[ielement_z]
                else
                    jacobian_matrix[row,(icolumn_min_z-1)*v_size+v_remainder:v_size:(icolumn_max_z-1)*v_size+v_remainder] .+=
                    dt * this_z_speed * 0.5 * z_Dmat[end,:] ./ z_element_scale[ielement_z]
                    jacobian_matrix[row,(icolumn_min_z_next-1)*v_size+v_remainder:v_size:(icolumn_max_z_next-1)*v_size+v_remainder] .+=
                    dt * this_z_speed * 0.5 * z_Dmat[1,:] ./ z_element_scale[ielement_z+1]
                end
            else
                jacobian_matrix[row,(icolumn_min_z-1)*v_size+v_remainder:v_size:(icolumn_max_z-1)*v_size+v_remainder] .+=
                dt * this_z_speed * z_Dmat[igrid_z,:] ./ z_element_scale[ielement_z]
            end
        end
        # vth = sqrt(2*p/n/me)
        # so d(vth)/d(p) = 1/n/me/sqrt(2*p/n/me) = 1/n/me/vth
        # and d(w_∥*vth*dg/dz)/d(p) = 1/n/me/vth*w_∥*dg/dz
        if include ∈ (:all, :explicit_v)
            jacobian_matrix[row,p_offset+iz] += dt / dens[iz] / me / vth[iz] * vpa.grid[ivpa] * dpdf_dz[ivpa,ivperp,iz]
        end
    end

    return nothing
end

function add_electron_z_advection_to_z_only_Jacobian!(
        jacobian_matrix, f, dens, upar, p, vth, dpdf_dz, me, z, vperp, vpa, z_spectral,
        z_advect, z_speed, scratch_dummy, dt, ir, ivperp, ivpa)

    @debug_consistency_checks size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @debug_consistency_checks size(jacobian_matrix, 1) == z.n || error("Jacobian matrix size is wrong")

    if !isa(z_spectral, gausslegendre_info)
        error("Only gausslegendre_pseudospectral z-coordinate type is supported by "
              * "add_electron_z_advection_to_Jacobian!() preconditioner because we need "
              * "differentiation matrices.")
    end
    z_Dmat = z_spectral.lobatto.Dmat
    z_element_scale = z.element_scale

    @loop_z iz begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa,
                                                 z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = iz

        ielement_z = z.ielement[iz]
        igrid_z = z.igrid[iz]
        icolumn_min_z = z.imin[ielement_z] - (ielement_z != 1)
        icolumn_max_z = z.imax[ielement_z]

        this_z_speed = z_speed[iz,ivpa,ivperp]

        # Contributions from (w_∥*vth + upar)*dg/dz
        if ielement_z == 1 && igrid_z == 1
            jacobian_matrix[row,icolumn_min_z:icolumn_max_z] .+=
            dt * this_z_speed * z_Dmat[1,:] ./ z_element_scale[ielement_z]
        elseif ielement_z == z.nelement_local && igrid_z == z.ngrid
            jacobian_matrix[row,icolumn_min_z:icolumn_max_z] .+=
            dt * this_z_speed * z_Dmat[end,:] ./ z_element_scale[ielement_z]
        elseif igrid_z == z.ngrid
            # Note igrid_z is only ever 1 when ielement_z==1, because
            # of the way element boundaries are counted.
            icolumn_min_z_next = z.imin[ielement_z+1] - 1
            icolumn_max_z_next = z.imax[ielement_z+1]
            if this_z_speed < 0.0
                jacobian_matrix[row,icolumn_min_z_next:icolumn_max_z_next] .+=
                dt * this_z_speed * z_Dmat[1,:] ./ z_element_scale[ielement_z+1]
            elseif this_z_speed > 0.0
                jacobian_matrix[row,icolumn_min_z:icolumn_max_z] .+=
                dt * this_z_speed * z_Dmat[end,:] ./ z_element_scale[ielement_z]
            else
                jacobian_matrix[row,icolumn_min_z:icolumn_max_z] .+=
                dt * this_z_speed * 0.5 * z_Dmat[end,:] ./ z_element_scale[ielement_z]
                jacobian_matrix[row,icolumn_min_z_next:icolumn_max_z_next] .+=
                dt * this_z_speed * 0.5 * z_Dmat[1,:] ./ z_element_scale[ielement_z+1]
            end
        else
            jacobian_matrix[row,icolumn_min_z:icolumn_max_z] .+=
            dt * this_z_speed * z_Dmat[igrid_z,:] ./ z_element_scale[ielement_z]
        end
    end

    return nothing
end

function add_electron_z_advection_to_v_only_Jacobian!(
        jacobian_matrix, f, dens, upar, p, vth, dpdf_dz, me, z, vperp, vpa, z_spectral,
        z_advect, z_speed, scratch_dummy, dt, ir, iz)

    @debug_consistency_checks size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @debug_consistency_checks size(jacobian_matrix, 1) == vperp.n * vpa.n + 1 || error("Jacobian matrix size is wrong")

    @loop_vperp_vpa ivperp ivpa begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa,
                                                 z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = (ivperp - 1) * vpa.n + ivpa

        jacobian_matrix[row,end] += dt / dens / me / vth * vpa.grid[ivpa] * dpdf_dz[ivpa,ivperp]
    end

    return nothing
end

end
