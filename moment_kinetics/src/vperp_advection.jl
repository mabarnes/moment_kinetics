module vperp_advection

export vperp_advection!
export update_speed_vperp!

using ..advection: advance_f_local!
using ..chebyshev: chebyshev_info
using ..debugging
using ..looping
using ..timer_utils
using ..z_advection: update_speed_z!
using ..r_advection: update_speed_r!

# do a single stage time advance (potentially as part of a multi-stage RK scheme)
@timeit global_timer vperp_advection!(
                         f_out, fvec_in, vperp_advect, r, z, vperp, vpa, dt,
                         vperp_spectral, composition, z_advect, r_advect, geometry,
                         moments, fields, t) = begin
    
    # if appropriate, update z and r speeds (except for the 'Spitzer test', this function
    # does nothing because the r- and z-speeds have already been calculated).
    update_z_r_speeds!(z_advect, r_advect, fvec_in, moments, fields, geometry, vpa, vperp,
                       z, r, t)
    # get the updated speed along the vperp direction using the current f
    update_speed_vperp!(vperp_advect, fvec_in, vpa, vperp, z, r, 
                        z_advect, r_advect, geometry, moments)
    
    @begin_s_r_z_vpa_region()
    @loop_s is begin
        @loop_r_z_vpa ir iz ivpa begin
            @views advance_f_local!(f_out[ivpa,:,iz,ir,is], fvec_in.pdf[ivpa,:,iz,ir,is],
                                    vperp_advect[is], ivpa, iz, ir, vperp, dt, vperp_spectral)            
        end
    end
end

# calculate the advection speed in the vperp-direction at each grid point
# note that the vperp advection speed depends on the z and r advection speeds
# It is important to ensure that z_advect and r_advect are updated before vperp_advect
function update_speed_vperp!(vperp_advect, fvec, vpa, vperp, z, r, z_advect, r_advect, geometry, moments)
    @debug_consistency_checks z.n == size(vperp_advect[1].speed,3) || throw(BoundsError(vperp_advect[1]))
    @debug_consistency_checks vperp.n == size(vperp_advect[1].speed,1) || throw(BoundsError(vperp_advect[1]))
    @debug_consistency_checks vpa.n == size(vperp_advect[1].speed,2) || throw(BoundsError(vperp_advect[1]))
    @debug_consistency_checks r.n == size(vperp_advect[1].speed,4) || throw(BoundsError(vperp_advect[1]))
    @begin_s_r_z_vpa_region()
    if moments.evolve_p
        update_speed_vperp_n_u_p_evolution!(vperp_advect, fvec, vpa, vperp, z, r, z_advect, r_advect, geometry, moments)
    elseif moments.evolve_upar
        update_speed_vperp_n_u_evolution!(vperp_advect, vpa, vperp, z, r, z_advect, r_advect, geometry, moments)
    elseif moments.evolve_density
        update_speed_vperp_n_evolution!(vperp_advect, vpa, vperp, z, r, z_advect, r_advect, geometry, moments)
    else
        update_speed_vperp_DK!(vperp_advect, vpa, vperp, z, r, z_advect, r_advect, geometry, moments)
    end

    return nothing
end

"""
update vperp advection speed when n, u, p are evolved separately
"""
function update_speed_vperp_n_u_p_evolution!(vperp_advect, fvec, vpa, vperp, z, r, z_advect, r_advect, geometry, moments)
    # update perpendicular advection speed, which has an extra term apart due to
    # normalisation by thermal speed, so wperp grid is constantly stretching and
    # compressing to account for changing local temperatures while maintaining
    # a normalised perpendicular speed.
    vth = moments.ion.vth
    dvth_dr = moments.ion.dvth_dr
    dvth_dz = moments.ion.dvth_dz
    dvth_dt = moments.ion.dvth_dt
    wperp = vperp.grid
    dBdr = geometry.dBdr
    dBdz = geometry.dBdz
    Bmag = geometry.Bmag
    rfac = 0.0
    if r.n > 1
        rfac = 1.0
    end
    @loop_s is begin
        r_speed = r_advect[is].speed
        z_speed = z_advect[is].speed
        @loop_r ir begin
            @loop_z_vpa iz ivpa begin
                @loop_vperp ivperp begin
                    dzdt = z_speed[iz,ivpa,ivperp,ir]
                    drdt = rfac*r_speed[ir,ivpa,ivperp,iz]
                    vperp_advect[is].speed[ivperp,ivpa,iz,ir] =
                                        - (1/vth[iz,ir,is]) * (wperp[ivperp] *
                                            (dvth_dt[iz,ir,is]
                                             + drdt*dvth_dr[iz,ir,is] + dzdt*dvth_dz[iz,ir,is])
                                             - ((0.5 * wperp[ivperp] * vth[iz,ir,is]/Bmag[iz,ir])
                                             * (drdt*dBdr[iz,ir] + dzdt*dBdz[iz,ir]))
                                           )

                end
            end
        end
    end

    return nothing
end

"""
update vperp advection speed when n, u are evolved separately
"""
function update_speed_vperp_n_u_evolution!(vperp_advect, vpa, vperp, z, r, z_advect, r_advect, geometry, moments)
    dBdr = geometry.dBdr
    dBdz = geometry.dBdz
    Bmag = geometry.Bmag
    rfac = 0.0
    if r.n > 1
        rfac = 1.0
    end
    @loop_s is begin
        z_speed = z_advect[is].speed
        r_speed = r_advect[is].speed
        @loop_r_z_vpa ir iz ivpa begin
            @loop_vperp ivperp begin
                dzdt = z_speed[iz,ivpa,ivperp,ir]
                drdt = rfac*r_speed[ir,ivpa,ivperp,iz]
                vperp_advect[is].speed[ivperp,ivpa,iz,ir] = (0.5*vperp.grid[ivperp]/Bmag[iz,ir])*
                                                            (dzdt*dBdz[iz,ir] + drdt*dBdr[iz,ir])
            end
        end
    end
    return nothing
end

"""
update vperp advection speed when n is evolved separately
"""
function update_speed_vperp_n_evolution!(vperp_advect, vpa, vperp, z, r, z_advect, r_advect, geometry, moments)
    dBdr = geometry.dBdr
    dBdz = geometry.dBdz
    Bmag = geometry.Bmag
    rfac = 0.0
    if r.n > 1
        rfac = 1.0
    end
    @loop_s is begin
        z_speed = z_advect[is].speed
        r_speed = r_advect[is].speed
        @loop_r_z_vpa ir iz ivpa begin
            @loop_vperp ivperp begin
                dzdt = z_speed[iz,ivpa,ivperp,ir]
                drdt = rfac*r_speed[ir,ivpa,ivperp,iz]
                vperp_advect[is].speed[ivperp,ivpa,iz,ir] = (0.5*vperp.grid[ivperp]/Bmag[iz,ir])*
                                                            (dzdt*dBdz[iz,ir] + drdt*dBdr[iz,ir])
            end
        end
    end
    return nothing
end

function update_speed_vperp_DK!(vperp_advect, vpa, vperp, z, r, z_advect, r_advect, geometry, moments)
    # with no moments evolved, only vperp advection possible is due to magnetic trapping
    dzdt = vperp.scratch
    drdt = vperp.scratch2
    dBdr = geometry.dBdr
    dBdz = geometry.dBdz
    Bmag = geometry.Bmag
    rfac = 0.0
    if r.n > 1
        rfac = 1.0
    end
    @loop_s is begin
        z_speed = z_advect[is].speed
        r_speed = r_advect[is].speed
        @loop_r_z_vpa ir iz ivpa begin
            @loop_vperp ivperp begin
                dzdt = z_speed[iz,ivpa,ivperp,ir]
                drdt = rfac*r_speed[ir,ivpa,ivperp,iz]
                vperp_advect[is].speed[ivperp,ivpa,iz,ir] = (0.5*vperp.grid[ivperp]/Bmag[iz,ir])*
                                                            (dzdt*dBdz[iz,ir] + drdt*dBdr[iz,ir])
            end
        end
    end
    return nothing
end

function update_z_r_speeds!(z_advect, r_advect, fvec_in, moments, fields,
                            geometry, vpa, vperp, z, r, t)
    update_z_speed = (z.n == 1 && geometry.input.option == "0D-Spitzer-test")
    if update_z_speed
        # make sure z speed is physical despite
        # 0D nature of simulation
        @begin_s_r_vperp_vpa_region()
        @loop_s is begin
            # get the updated speed along the z direction using the current f
            @views update_speed_z!(z_advect[is], fvec_in.upar[:,:,is],
                                   moments.ion.vth[:,:,is], moments.evolve_upar,
                                   moments.evolve_ppar, fields, vpa, vperp, z, r, t, geometry, is)
        end
    end
    
    update_r_speed = (r.n == 1 && geometry.input.option == "0D-Spitzer-test")
    if update_r_speed
        # make sure r speed is physical despite
        # 0D nature of simulation
        @begin_s_z_vperp_vpa_region()
        @loop_s is begin
            # get the updated speed along the r direction using the current f
            @views update_speed_r!(r_advect[is], fields, moments.evolve_upar,
                                   moments.evolve_ppar, vpa, vperp, z, r, geometry, is)
        end
    end
    return nothing
end    

end
