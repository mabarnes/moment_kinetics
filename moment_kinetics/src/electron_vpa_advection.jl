"""
"""
module electron_vpa_advection

export electron_vpa_advection!
export update_electron_speed_vpa!
export add_electron_vpa_advection_to_Jacobian!

using ..looping
using ..boundary_conditions: skip_f_electron_bc_points_in_Jacobian
using ..calculus: derivative!, second_derivative!
using ..gauss_legendre: gausslegendre_info
using ..timer_utils

"""
calculate the wpa-advection term for the electron kinetic equation 
= (vthe / 2 ppare * dppare/dz + wpa / 2 ppare * dqpare/dz - wpa^2 * dvthe/dz) * df/dwpa
"""
@timeit global_timer electron_vpa_advection!(
                         pdf_out, pdf_in, density, upar, ppar, moments, advect, vpa,
                         spectral, scratch_dummy, dt, electron_source_settings,
                         ir) = begin
    @begin_z_vperp_region()

    adv_fac = advect[1].adv_fac
    speed = advect[1].speed

    # create a reference to a scratch_dummy array to store the wpa-derivative of the electron pdf
    dpdf_dvpa = @view scratch_dummy.buffer_vpavperpzr_1[:,:,:,ir]
    #d2pdf_dvpa2 = @view scratch_dummy.buffer_vpavperpzr_2[:,:,:,ir]

    # get the updated speed along the wpa direction using the current pdf
    @views update_electron_speed_vpa!(advect[1], density, upar, ppar, moments, vpa.grid,
                                      electron_source_settings, ir)
    # update adv_fac
    @loop_z_vperp iz ivperp begin
        @views @. adv_fac[:,ivperp,iz,ir] = -speed[:,ivperp,iz,ir]
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
        @views @. pdf_out[:,ivperp,iz] += dt * adv_fac[:,ivperp,iz,ir] * dpdf_dvpa[:,ivperp,iz]
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
    speed = @view advect.speed[:,:,:,ir]
    # calculate the advection speed in wpa
    @loop_z_vperp_vpa iz ivperp ivpa begin
        speed[ivpa,ivperp,iz] = ((vth[iz] * dppar_dz[iz] + vpa[ivpa] * dqpar_dz[iz])
                                    / (2 * ppar[iz]) - vpa[ivpa]^2 * dvth_dz[iz])
    end

    for index ∈ eachindex(electron_source_settings)
        if electron_source_settings[index].active
            @views source_density_amplitude = moments.electron.external_source_density_amplitude[:, ir, index]
            @views source_momentum_amplitude = moments.electron.external_source_momentum_amplitude[:, ir, index]
            @views source_pressure_amplitude = moments.electron.external_source_pressure_amplitude[:, ir, index]
            @loop_z iz begin
                term1 = source_density_amplitude[iz] * upar[iz]/(density[iz]*vth[iz])
                term2_over_vpa =
                    -0.5 * (source_pressure_amplitude[iz] +
                            2.0 * upar[iz] * source_momentum_amplitude[iz]) /
                        ppar[iz] +
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
function update_electron_speed_vpa!(advect, density, upar, ppar, moments, vpa,
                                    electron_source_settings)
    @loop_r ir begin
        @views update_electron_speed_vpa!(advect, density[:,ir], upar[:,ir], ppar[:,ir],
                                          moments, vpa, electron_source_settings, ir)
    end
    return nothing
end

function add_electron_vpa_advection_to_Jacobian!(jacobian_matrix, f, dens, upar, ppar,
                                                 vth, third_moment, dpdf_dvpa, ddens_dz,
                                                 dppar_dz, dthird_moment_dz, moments, me,
                                                 z, vperp, vpa, z_spectral, vpa_spectral,
                                                 vpa_advect, z_speed, scratch_dummy,
                                                 external_source_settings, dt, ir,
                                                 include=:all,
                                                 include_qpar_integral_terms=true,
                                                 separate_moments=true; f_offset=0,
                                                 third_moment_offset=0, ppar_offset=0)
    if separate_moments
        @boundscheck if !allunique((f_offset, third_moment_offset, ppar_offset))
            error("Some offsets are the same. Two variables cannot be in the same place in "
                  * "the state vector. Got f_offset=$f_offset, "
                  * "third_moment_offset=$third_moment_offset, "
                  * "ppar_offset=$ppar_offset.")
        end
    else
        @boundscheck if f_offset == ppar_offset
            error("Got f_offset=$f_offset the same as ppar_offset=$ppar_offset. f and ppar "
                  * "cannot be in same place in state vector.")
        end
    end
    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @boundscheck size(jacobian_matrix, 1) ≥ f_offset + z.n * vperp.n * vpa.n || error("f_offset=$f_offset is too big")
    @boundscheck !separate_moments || size(jacobian_matrix, 1) ≥ third_moment_offset + z.n || error("ppar_offset=$ppar_offset is too big")
    @boundscheck size(jacobian_matrix, 1) ≥ ppar_offset + z.n || error("ppar_offset=$ppar_offset is too big")
    @boundscheck include ∈ (:all, :explicit_z, :explicit_v) || error("Unexpected value for include=$include")

    v_size = vperp.n * vpa.n
    source_density_amplitude = @view moments.electron.external_source_density_amplitude[:,ir,:]
    source_momentum_amplitude = @view moments.electron.external_source_momentum_amplitude[:,ir,:]
    source_pressure_amplitude = @view moments.electron.external_source_pressure_amplitude[:,ir,:]

    if !isa(vpa_spectral, gausslegendre_info)
        error("Only gausslegendre_pseudospectral vpa-coordinate type is supported by "
              * "add_electron_vpa_advection_to_Jacobian!() preconditioner because we "
              * "need differentiation matrices.")
    end

    z_deriv_matrix = z_spectral.D_matrix_csr
    vpa_Dmat = vpa_spectral.lobatto.Dmat
    vpa_element_scale = vpa.element_scale

    @begin_z_vperp_vpa_region()
    @loop_z_vperp_vpa iz ivperp ivpa begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = (iz - 1) * v_size + (ivperp - 1) * vpa.n + ivpa + f_offset

        ielement_vpa = vpa.ielement[ivpa]
        igrid_vpa = vpa.igrid[ivpa]
        icolumn_min_vpa = vpa.imin[ielement_vpa] - (ielement_vpa != 1) + f_offset
        icolumn_max_vpa = vpa.imax[ielement_vpa] + f_offset

        vpa_speed = vpa_advect[1].speed[ivpa,ivperp,iz,ir]

        # Contributions from
        #   (1/2*vth/p*dp/dz + 1/2*w_∥/p*dq/dz - w_∥^2*dvth/dz
        #    + source_density_amplitude*u/n/vth
        #    - w_∥*1/2*(source_pressure_amplitude + 2*u*source_momentum_amplitude)/p
        #    + w_∥*1/2*source_density_amplitude/n) * dg/dw_∥
        if include ∈ (:all, :explicit_v)
            if ielement_vpa == 1 && igrid_vpa == 1
                jacobian_matrix[row,(iz-1)*v_size+(ivperp-1)*vpa.n+icolumn_min_vpa:(iz-1)*v_size+(ivperp-1)*vpa.n+icolumn_max_vpa] .+=
                    dt * vpa_speed * vpa_Dmat[1,:] ./ vpa_element_scale[ielement_vpa]
            elseif ielement_vpa == vpa.nelement_local && igrid_vpa == vpa.ngrid
                jacobian_matrix[row,(iz-1)*v_size+(ivperp-1)*vpa.n+icolumn_min_vpa:(iz-1)*v_size+(ivperp-1)*vpa.n+icolumn_max_vpa] .+=
                    dt * vpa_speed * vpa_Dmat[end,:] ./ vpa_element_scale[ielement_vpa]
            elseif igrid_vpa == vpa.ngrid
                # Note igrid_vpa is only ever 1 when ielement_vpa==1, because
                # of the way element boundaries are counted.
                icolumn_min_vpa_next = vpa.imin[ielement_vpa+1] - 1
                icolumn_max_vpa_next = vpa.imax[ielement_vpa+1]
                if vpa_speed < 0.0
                    jacobian_matrix[row,(iz-1)*v_size+(ivperp-1)*vpa.n+icolumn_min_vpa_next:(iz-1)*v_size+(ivperp-1)*vpa.n+icolumn_max_vpa_next] .+=
                        dt * vpa_speed * vpa_Dmat[1,:] ./ vpa_element_scale[ielement_vpa+1]
                elseif vpa_speed > 0.0
                    jacobian_matrix[row,(iz-1)*v_size+(ivperp-1)*vpa.n+icolumn_min_vpa:(iz-1)*v_size+(ivperp-1)*vpa.n+icolumn_max_vpa] .+=
                        dt * vpa_speed * vpa_Dmat[end,:] ./ vpa_element_scale[ielement_vpa]
                else
                    jacobian_matrix[row,(iz-1)*v_size+(ivperp-1)*vpa.n+icolumn_min_vpa:(iz-1)*v_size+(ivperp-1)*vpa.n+icolumn_max_vpa] .+=
                        dt * vpa_speed * 0.5 * vpa_Dmat[end,:] ./ vpa_element_scale[ielement_vpa]
                    jacobian_matrix[row,(iz-1)*v_size+(ivperp-1)*vpa.n+icolumn_min_vpa_next:(iz-1)*v_size+(ivperp-1)*vpa.n+icolumn_max_vpa_next] .+=
                        dt * vpa_speed * 0.5 * vpa_Dmat[1,:] ./ vpa_element_scale[ielement_vpa+1]
                end
            else
                jacobian_matrix[row,(iz-1)*v_size+(ivperp-1)*vpa.n+icolumn_min_vpa:(iz-1)*v_size+(ivperp-1)*vpa.n+icolumn_max_vpa] .+=
                    dt * vpa_speed * vpa_Dmat[igrid_vpa,:] ./ vpa_element_scale[ielement_vpa]
            end
        end
        # q = 2*p*vth*∫dw_∥ w_∥^3 g
        #   = 2*p^(3/2)*sqrt(2/n/me)*∫dw_∥ w_∥^3 g
        # dq/dz = 3*sqrt(2*p/n/me)*∫dw_∥ w_∥^3 g * dp/dz
        #         - p^(3/2)*sqrt(2/me)/n^(3/2)*∫dw_∥ w_∥^3 g * dn/dz
        #         + 2*p*vth*∫dw_∥ w_∥^3 dg/dz
        # w_∥*0.5/p*dq/dz = w_∥*1.5*sqrt(2/p/n/me)*∫dw_∥ w_∥^3 g * dp/dz
        #                   - w_∥*0.5*sqrt(2*p/me)/n^(3/2)*∫dw_∥ w_∥^3 g * dn/dz
        #                   + w_∥*sqrt(2*p/n/me)*∫dw_∥ w_∥^3 dg/dz
        #                 = w_∥*1.5*sqrt(2/p/n/me)*∫dw_∥ w_∥^3 g * dp/dz
        #                   - w_∥*0.5*sqrt(2*p/me)/n^(3/2)*∫dw_∥ w_∥^3 g * dn/dz
        #                   + w_∥*vth*∫dw_∥ w_∥^3 dg/dz
        # d(w_∥*0.5/p*dq/dz[irowz])/d(g[icolvpa,icolvperp,icolz]) =
        #   w_∥*(1.5*sqrt(2/p/n/me)*dp/dz - 0.5*sqrt(2*p/me)/n^(3/2)*dn/dz) * delta(irowz,icolz) * vpa.wgts[icolvpa]/sqrt(π) * vpa.grid[icolvpa]^3
        #   + w_∥*vth * vpa.wgts[icolvpa]/sqrt(π) * vpa.grid[icolvpa]^3 * z_deriv_matrix[irowz,icolz]
        # d(w_∥*0.5/p*dq/dz[irowz])/d(p[icolz]) =
        #   (-w_∥*3/4*sqrt(2/n/me)/p^(3/2)*∫dw_∥ w_∥^3 g * dp/dz - w_∥*1/4*sqrt(2/me)/sqrt(p)/n^(3/2)*∫dw_∥ w_∥^3 g * dn/dz + w_∥*1/2*sqrt(2/n/me)/sqrt(p)*∫dw_∥ w_∥^3 dg/dz)[irowz] * delta(irowz,icolz)
        #   + w_∥*(1.5*sqrt(2/p/n/me)*∫dw_∥ w_∥^3 g)[irowz] * z_deriv_matrix[irowz,icolz]
        if include ∈ (:all, :explicit_v)
            if separate_moments
                col = iz + third_moment_offset
                jacobian_matrix[row,col] += dt * dpdf_dvpa[ivpa,ivperp,iz] *
                    vpa.grid[ivpa] * (1.5*sqrt(2.0/ppar[iz]/dens[iz]/me)*dppar_dz[iz]
                                      - 0.5*sqrt(2.0*ppar[iz]/me)/dens[iz]^1.5*ddens_dz[iz])
            else
                for icolvperp ∈ 1:vperp.n, icolvpa ∈ 1:vpa.n
                    col = (iz - 1) * v_size + (icolvperp - 1) * vpa.n + icolvpa + f_offset
                    jacobian_matrix[row,col] += dt * dpdf_dvpa[ivpa,ivperp,iz] *
                        vpa.grid[ivpa] * (1.5*sqrt(2.0/ppar[iz]/dens[iz]/me)*dppar_dz[iz]
                                          - 0.5*sqrt(2.0*ppar[iz]/me)/dens[iz]^1.5*ddens_dz[iz]) *
                                       vpa.wgts[icolvpa]/sqrt(π) * vpa.grid[icolvpa]^3
                end
            end
        end
        z_deriv_row_startind = z_deriv_matrix.rowptr[iz]
        z_deriv_row_endind = z_deriv_matrix.rowptr[iz+1] - 1
        z_deriv_colinds = @view z_deriv_matrix.colval[z_deriv_row_startind:z_deriv_row_endind]
        z_deriv_row_nonzeros = @view z_deriv_matrix.nzval[z_deriv_row_startind:z_deriv_row_endind]
        if separate_moments
            for (icolz, z_deriv_entry) ∈ zip(z_deriv_colinds, z_deriv_row_nonzeros)
                col = icolz + third_moment_offset
                jacobian_matrix[row,col] += dt * dpdf_dvpa[ivpa,ivperp,iz] *
                    vpa.grid[ivpa] * vth[iz] * z_deriv_entry
            end
        elseif include_qpar_integral_terms
            for (icolz, z_deriv_entry) ∈ zip(z_deriv_colinds, z_deriv_row_nonzeros), icolvperp ∈ 1:vperp.n, icolvpa ∈ 1:vpa.n
                col = (icolz - 1) * v_size + (icolvperp - 1) * vpa.n + icolvpa + f_offset
                jacobian_matrix[row,col] += dt * dpdf_dvpa[ivpa,ivperp,iz] *
                    vpa.grid[ivpa] * vth[iz] * vpa.wgts[icolvpa]/sqrt(π) * vpa.grid[icolvpa]^3 * z_deriv_entry
            end
        end
        if include ∈ (:all, :explicit_v)
            jacobian_matrix[row,ppar_offset+iz] += dt * dpdf_dvpa[ivpa,ivperp,iz] * vpa.grid[ivpa] *
                (-0.75*sqrt(2.0/dens[iz]/me)/ppar[iz]^1.5*third_moment[iz]*dppar_dz[iz]
                 - 0.25*sqrt(2.0/me/ppar[iz])/dens[iz]^1.5*third_moment[iz]*ddens_dz[iz]
                 + 0.5*sqrt(2.0/dens[iz]/me/ppar[iz])*dthird_moment_dz[iz])
        end
        for (icolz, z_deriv_entry) ∈ zip(z_deriv_colinds, z_deriv_row_nonzeros)
            col = ppar_offset + icolz
            jacobian_matrix[row,col] += dt * dpdf_dvpa[ivpa,ivperp,iz] * vpa.grid[ivpa] * 1.5*sqrt(2.0/ppar[iz]/dens[iz]/me)*third_moment[iz] * z_deriv_entry
        end
        #   (1/2*vth/p*dp/dz - w_∥^2*dvth/dz
        #    + source_density_amplitude*u/n/vth
        #    - w_∥*1/2*(source_pressure_amplitude + 2*u*source_momentum_amplitude)/p
        #    + w_∥*1/2*source_density_amplitude/n)
        # = (1/2*sqrt(2/p/n)*dp/dz - w_∥^2*dvth/dz
        #    + source_density_amplitude*u/sqrt(2*p*n)
        #    - w_∥*1/2*(source_pressure_amplitude + 2*u*source_momentum_amplitude)/p
        #    + w_∥*1/2*source_density_amplitude/n)
        #
        # dvth/dz = d/dz(sqrt(2*p/n/me))
        #         = 1/n/me/sqrt(2*p/n/me)*dp/dz - p/n^2/me/sqrt(2*p/n/me)*dn/dz
        #         = 1/sqrt(2*p*n*me)*dp/dz - 1/2*sqrt(2*p/n/me)/n*dn/dz
        # d(dvth/dz[irowz])/d(ppar[icolz]) =
        #   (-1/2/sqrt(2*n*me)/p^(3/2)*dp/dz - 1/4*sqrt(2/me)/p^(1/2)/n^(3/2)*dn/dz)[irowz] * delta(irowz,icolz)
        #   +1/sqrt(2*p*n*me)[irowz] * z_deriv_matrix[irowz,icolz]
        #
        # ⇒ d((1/2*vth/p*dp/dz - w_∥^2*dvth/dz
        #      + source_density_amplitude*u/n/vth
        #      - w_∥*1/2*(source_pressure_amplitude + 2*u*source_momentum_amplitude)/p
        #      + w_∥*1/2*source_density_amplitude/n)[irowz]/d(ppar[icolz])
        # = (-1/4*sqrt(2/n/me)/p^(3/2)*dp/dz
        #    - w_∥^2*(-1/2/sqrt(2*n*me)/p^(3/2)*dp/dz - 1/4*sqrt(2/me)/p^(1/2)/n^(3/2)*dn/dz)
        #    - 1/2*source_density_amplitude*u/sqrt(2*n)/p^(3/2)
        #    + w_∥*1/2*(source_pressure_amplitude + 2*u*source_momentum_amplitude)/p^2)[irowz] * delta(irowz,icolz)
        #   + (1/2*sqrt(2/p/n/me) - w_∥^2/sqrt(2*p*n*me))[irowz] * z_deriv_matrix[irowz,icolz]
        if include ∈ (:all, :explicit_v)
            jacobian_matrix[row,ppar_offset+iz] += dt * (
                -0.25*sqrt(2.0/dens[iz]/me)/ppar[iz]^1.5*dppar_dz[iz]
                - vpa.grid[ivpa]^2*(-0.5/sqrt(2.0*dens[iz]*me)/ppar[iz]^1.5*dppar_dz[iz] - 0.25*sqrt(2.0/me/ppar[iz])/dens[iz]^1.5*ddens_dz[iz])
               ) * dpdf_dvpa[ivpa,ivperp,iz]
            for index ∈ eachindex(external_source_settings.electron)
                electron_source = external_source_settings.electron[index]
                if electron_source.active
                    jacobian_matrix[row,ppar_offset+iz] += dt * (
                        -0.5*source_density_amplitude[iz,index]*upar[iz]/sqrt(2.0*dens[iz])/ppar[iz]^1.5
                        + vpa.grid[ivpa]*0.5*(source_pressure_amplitude[iz,index]
                                              + 2.0*upar[iz]*source_momentum_amplitude[iz,index])/ppar[iz]^2
                       ) * dpdf_dvpa[ivpa,ivperp,iz]
                end
            end
        end
        for (icolz, z_deriv_entry) ∈ zip(z_deriv_colinds, z_deriv_row_nonzeros)
            col = ppar_offset + icolz
            jacobian_matrix[row,col] += dt * (
                0.5*sqrt(2.0/ppar[iz]/dens[iz]/me)
                - vpa.grid[ivpa]^2/sqrt(2.0*ppar[iz]*dens[iz]*me)
               ) * dpdf_dvpa[ivpa,ivperp,iz] * z_deriv_entry
        end
    end

    return nothing
end

function add_electron_vpa_advection_to_v_only_Jacobian!(
        jacobian_matrix, f, dens, upar, ppar, vth, third_moment, dpdf_dvpa, ddens_dz,
        dppar_dz, dthird_moment_dz, moments, me, z, vperp, vpa, z_spectral, vpa_spectral,
        vpa_advect, z_speed, scratch_dummy, external_source_settings, dt, ir, iz)

    @boundscheck size(jacobian_matrix, 1) == size(jacobian_matrix, 2) || error("Jacobian is not square")
    @boundscheck size(jacobian_matrix, 1) == vperp.n * vpa.n + 1 || error("Jacobian matrix size is wrong")

    source_density_amplitude = @view moments.electron.external_source_density_amplitude[iz,ir,:]
    source_momentum_amplitude = @view moments.electron.external_source_momentum_amplitude[iz,ir,:]
    source_pressure_amplitude = @view moments.electron.external_source_pressure_amplitude[iz,ir,:]

    if !isa(vpa_spectral, gausslegendre_info)
        error("Only gausslegendre_pseudospectral vpa-coordinate type is supported by "
              * "add_electron_vpa_advection_to_Jacobian!() preconditioner because we "
              * "need differentiation matrices.")
    end

    vpa_Dmat = vpa_spectral.lobatto.Dmat
    vpa_element_scale = vpa.element_scale

    @loop_vperp_vpa ivperp ivpa begin
        if skip_f_electron_bc_points_in_Jacobian(iz, ivperp, ivpa, z, vperp, vpa, z_speed)
            continue
        end

        # Rows corresponding to pdf_electron
        row = (ivperp - 1) * vpa.n + ivpa

        ielement_vpa = vpa.ielement[ivpa]
        igrid_vpa = vpa.igrid[ivpa]
        icolumn_min_vpa = vpa.imin[ielement_vpa] - (ielement_vpa != 1)
        icolumn_max_vpa = vpa.imax[ielement_vpa]

        vpa_speed = vpa_advect[1].speed[ivpa,ivperp,iz,ir]

        if ielement_vpa == 1 && igrid_vpa == 1
            jacobian_matrix[row,(ivperp-1)*vpa.n+icolumn_min_vpa:(ivperp-1)*vpa.n+icolumn_max_vpa] .+=
                dt * vpa_speed * vpa_Dmat[1,:] ./ vpa_element_scale[ielement_vpa]
        elseif ielement_vpa == vpa.nelement_local && igrid_vpa == vpa.ngrid
            jacobian_matrix[row,(ivperp-1)*vpa.n+icolumn_min_vpa:(ivperp-1)*vpa.n+icolumn_max_vpa] .+=
                dt * vpa_speed * vpa_Dmat[end,:] ./ vpa_element_scale[ielement_vpa]
        elseif igrid_vpa == vpa.ngrid
            # Note igrid_vpa is only ever 1 when ielement_vpa==1, because
            # of the way element boundaries are counted.
            icolumn_min_vpa_next = vpa.imin[ielement_vpa+1] - 1
            icolumn_max_vpa_next = vpa.imax[ielement_vpa+1]
            if vpa_speed < 0.0
                jacobian_matrix[row,(ivperp-1)*vpa.n+icolumn_min_vpa_next:(ivperp-1)*vpa.n+icolumn_max_vpa_next] .+=
                    dt * vpa_speed * vpa_Dmat[1,:] ./ vpa_element_scale[ielement_vpa+1]
            elseif vpa_speed > 0.0
                jacobian_matrix[row,(ivperp-1)*vpa.n+icolumn_min_vpa:(ivperp-1)*vpa.n+icolumn_max_vpa] .+=
                    dt * vpa_speed * vpa_Dmat[end,:] ./ vpa_element_scale[ielement_vpa]
            else
                jacobian_matrix[row,(ivperp-1)*vpa.n+icolumn_min_vpa:(ivperp-1)*vpa.n+icolumn_max_vpa] .+=
                    dt * vpa_speed * 0.5 * vpa_Dmat[end,:] ./ vpa_element_scale[ielement_vpa]
                jacobian_matrix[row,(ivperp-1)*vpa.n+icolumn_min_vpa_next:(ivperp-1)*vpa.n+icolumn_max_vpa_next] .+=
                    dt * vpa_speed * 0.5 * vpa_Dmat[1,:] ./ vpa_element_scale[ielement_vpa+1]
            end
        else
            jacobian_matrix[row,(ivperp-1)*vpa.n+icolumn_min_vpa:(ivperp-1)*vpa.n+icolumn_max_vpa] .+=
                dt * vpa_speed * vpa_Dmat[igrid_vpa,:] ./ vpa_element_scale[ielement_vpa]
        end
        for icolvperp ∈ 1:vperp.n, icolvpa ∈ 1:vpa.n
            col = (icolvperp - 1) * vpa.n + icolvpa
            jacobian_matrix[row,col] += dt * dpdf_dvpa[ivpa,ivperp] *
                vpa.grid[ivpa] * (1.5*sqrt(2.0/ppar/dens/me)*dppar_dz
                                  - 0.5*sqrt(2.0*ppar/me)/dens^1.5*ddens_dz) *
                               vpa.wgts[icolvpa]/sqrt(π) * vpa.grid[icolvpa]^3
        end
        jacobian_matrix[row,end] += dt * dpdf_dvpa[ivpa,ivperp] * vpa.grid[ivpa] *
            (-0.75*sqrt(2.0/dens/me)/ppar^1.5*third_moment*dppar_dz
             - 0.25*sqrt(2.0/me/ppar)/dens^1.5*third_moment*ddens_dz
             + 0.5*sqrt(2.0/dens/me/ppar)*dthird_moment_dz)
        jacobian_matrix[row,end] += dt * (
            -0.25*sqrt(2.0/dens/me)/ppar^1.5*dppar_dz
            - vpa.grid[ivpa]^2*(-0.5/sqrt(2.0*dens*me)/ppar^1.5*dppar_dz - 0.25*sqrt(2.0/me/ppar)/dens^1.5*ddens_dz)
           ) * dpdf_dvpa[ivpa,ivperp]
        for index ∈ eachindex(external_source_settings.electron)
            electron_source = external_source_settings.electron[index]
            if electron_source.active
                jacobian_matrix[row,end] += dt * (
                    -0.5*source_density_amplitude[index]*upar/sqrt(2.0*dens)/ppar^1.5
                    + vpa.grid[ivpa]*0.5*(source_pressure_amplitude[index]
                                          + 2.0*upar*source_momentum_amplitude[index])/ppar^2
                   ) * dpdf_dvpa[ivpa,ivperp]
            end
        end
    end

    return nothing
end

end
