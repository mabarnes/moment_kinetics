module vperp_advection

export vperp_advection!
export update_speed_vperp!

using ..advection: advance_f_local!
using ..chebyshev: chebyshev_info
using ..debugging
using ..looping
using ..timer_utils
using ..type_definitions
using ..z_advection: update_speed_z!
using ..alpha_advection: update_speed_alpha!
using ..r_advection: update_speed_r!

# do a single stage time advance (potentially as part of a multi-stage RK scheme)
@timeit global_timer vperp_advection!(
                         f_out, fvec_in, vperp_advect, r, z, vperp, vpa, dt,
                         vperp_spectral, composition, z_advect, alpha_advect, r_advect,
                         geometry, moments, fields) = begin
    return vperp_advection!(f_out, fvec_in, vperp_advect, r, z, vperp, vpa, dt,
                            vperp_spectral, composition, z_advect, alpha_advect, r_advect,
                            geometry, moments, fields, Val(moments.evolve_density),
                            Val(moments.evolve_upar), Val(moments.evolve_p))
end
function vperp_advection!(f_out, fvec_in, vperp_advect, r, z, vperp, vpa, dt,
                          vperp_spectral, composition, z_advect, alpha_advect, r_advect,
                          geometry, moments, fields, evolve_density::Val,
                          evolve_upar::Val, evolve_p::Val)
    
    # if appropriate, update z and r speeds (except for the 'Spitzer test', this function
    # does nothing because the r- and z-speeds have already been calculated).
    update_z_alpha_r_speeds!(z_advect, alpha_advect, r_advect, fvec_in, moments, fields,
                             geometry, vpa, vperp, z, r)

    update_speed_vperp!(vperp_advect, fvec_in, vpa, vperp, z, r, z_advect, alpha_advect,
                        r_advect, geometry, moments)

    f_in = fvec_in.pdf
    @begin_s_r_z_vpa_region()
    @loop_s_r_z_vpa is ir iz ivpa begin
        @views advance_f_local!(f_out[ivpa,:,iz,ir,is], f_in[ivpa,:,iz,ir,is],
                                vperp_advect[ivpa,:,iz,ir,is], vperp, dt, vperp_spectral)
    end

    return nothing
end

# calculate the advection speed in the vperp-direction at each grid point
# note that the vperp advection speed depends on the z and r advection speeds
# It is important to ensure that z_advect and r_advect are updated before vperp_advect
function update_speed_vperp!(vperp_advect, fvec, vpa, vperp, z, r, z_advect, alpha_advect,
                             r_advect, geometry, moments)
    return update_speed_vperp!(vperp_advect, fvec, vpa, vperp, z, r, z_advect,
                               alpha_advect, r_advect, geometry, moments,
                               Val(moments.evolve_density), Val(moments.evolve_upar),
                               Val(moments.evolve_p))
end
function update_speed_vperp!(vperp_advect, fvec, vpa, vperp, z, r, z_advect, alpha_advect,
                             r_advect, geometry, moments, evolve_density::Val,
                             evolve_upar::Val, evolve_p::Val)
    @debug_consistency_checks r.n == size(vperp_advect,4) || throw(BoundsError(vperp_advect))
    @debug_consistency_checks z.n == size(vperp_advect,3) || throw(BoundsError(vperp_advect))
    @debug_consistency_checks vperp.n == size(vperp_advect,2) || throw(BoundsError(vperp_advect))
    @debug_consistency_checks vpa.n == size(vperp_advect,1) || throw(BoundsError(vperp_advect))
    @begin_s_r_z_vperp_region()
    speed_args = get_speed_vperp_inner_args(vperp_advect, moments, geometry, r_advect,
                                            alpha_advect, z_advect, r, vperp,
                                            evolve_density, evolve_upar, evolve_p)
    @loop_s_r is ir begin
        speed_args_sr = get_speed_vperp_inner_views_sr(is, ir, speed_args...)
        @loop_z iz begin
            speed_args_z = get_speed_vperp_inner_views_z(iz, speed_args_sr...)
            @loop_vperp ivperp begin
                update_speed_vperp_inner!(get_speed_vperp_inner_views_vperp(ivperp, speed_args_z...)...)
            end
        end
    end

    return nothing
end

@inline function get_speed_vperp_inner_args(vperp_advect, moments, geometry, r_advect, alpha_advect,
                                            z_advect, r, vperp, evolve_density::Val,
                                            evolve_upar::Val, evolve_p::Val)
    rfac = (r.n == 1) ? 0.0 : 1.0
    if evolve_p === Val(true)
        return vperp_advect, moments.ion.vth, moments.ion.dvth_dt, moments.ion.dvth_dr,
               moments.ion.dvth_dz, geometry.Bmag, geometry.dBdr, geometry.dBdz,
               vperp.grid, rfac, r_advect, alpha_advect, z_advect, evolve_density,
               evolve_upar, evolve_p
    elseif evolve_density === Val(true)
        return vperp_advect, geometry.Bmag, geometry.dBdr, geometry.dBdz, vperp.grid,
               rfac, r_advect, alpha_advect, z_advect, evolve_density, evolve_upar,
               evolve_p
    else
        return vperp_advect, geometry.Bmag, geometry.dBdr, geometry.dBdz, vperp.grid,
               rfac, r_advect, alpha_advect, z_advect, evolve_density, evolve_upar,
               evolve_p
    end
end

@inline function get_speed_vperp_inner_views_sr(is, ir, vperp_advect, vth, dvth_dt,
                                                dvth_dr, dvth_dz, Bmag, dBdr, dBdz,
                                                wperp, rfac, r_advect, alpha_advect,
                                                z_advect, evolve_density::Val{true},
                                                evolve_upar::Val{true},
                                                evolve_p::Val{true})
    return @views vperp_advect[:,:,:,ir,is], vth[:,ir,is], dvth_dt[:,ir,is],
                  dvth_dr[:,ir,is], dvth_dz[:,ir,is], Bmag[:,ir], dBdr[:,ir], dBdz[:,ir],
                  wperp, rfac, r_advect[:,:,:,ir,is], alpha_advect[:,:,:,ir,is],
                  z_advect[:,:,:,ir,is], evolve_density, evolve_upar,
                  evolve_p
end

@inline function get_speed_vperp_inner_views_z(iz, vperp_advect, vth, dvth_dt, dvth_dr,
                                               dvth_dz, Bmag, dBdr, dBdz, wperp, rfac,
                                               r_advect, alpha_advect, z_advect,
                                               evolve_density::Val{true},
                                               evolve_upar::Val{true},
                                               evolve_p::Val{true})
    return @views vperp_advect[:,:,iz], vth[iz], dvth_dt[iz], dvth_dr[iz], dvth_dz[iz],
                  Bmag[iz], dBdr[iz], dBdz[iz], wperp, rfac, r_advect[:,:,iz],
                  alpha_advect[:,:,iz], z_advect[:,:,iz], evolve_density, evolve_upar,
                  evolve_p
end

@inline function get_speed_vperp_inner_views_vperp(ivperp, vperp_advect, vth, dvth_dt,
                                                   dvth_dr, dvth_dz, Bmag, dBdr, dBdz,
                                                   wperp, rfac, r_advect, alpha_advect,
                                                   z_advect, evolve_density::Val{true},
                                                   evolve_upar::Val{true},
                                                   evolve_p::Val{true})
    return @views vperp_advect[:,ivperp], vth, dvth_dt, dvth_dr, dvth_dz, Bmag, dBdr,
                  dBdz, wperp[ivperp], rfac, r_advect[:,ivperp], alpha_advect[:,ivperp],
                  z_advect[:,ivperp], evolve_density, evolve_upar, evolve_p
end

"""
update vperp advection speed when n, u, p are evolved separately
"""
function update_speed_vperp_inner!(vperp_advect, vth, dvth_dt, dvth_dr, dvth_dz, Bmag,
                                   dBdr, dBdz, wperp, rfac, r_advect, alpha_advect,
                                   z_advect, evolve_density::Val{true},
                                   evolve_upar::Val{true}, evolve_p::Val{true})
    # update perpendicular advection speed, which has an extra term apart due to
    # normalisation by thermal speed, so wperp grid is constantly stretching and
    # compressing to account for changing local temperatures while maintaining
    # a normalised perpendicular speed.
    @. vperp_advect =
            - 1.0 / vth * (wperp * (dvth_dt + rfac * r_advect * dvth_dr
                                    + (z_advect + alpha_advect) * dvth_dz)
                           - (0.5 * wperp * vth / Bmag)
                             * (rfac * r_advect * dBdr + (z_advect + alpha_advect) * dBdz)
                          )

    return nothing
end

@inline function get_speed_vperp_inner_views_sr(is, ir, vperp_advect, Bmag, dBdr, dBdz,
                                                vperp, rfac, r_advect, alpha_advect,
                                                z_advect, evolve_density::Val{true},
                                                evolve_upar::Val,
                                                evolve_p::Val{false})
    return @views vperp_advect[:,:,:,ir,is], Bmag[:,ir], dBdr[:,ir], dBdz[:,ir], vperp,
                  rfac, r_advect[:,:,:,ir,is], alpha_advect[:,:,:,ir,is],
                  z_advect[:,:,:,ir,is], evolve_density, evolve_upar,
                  evolve_p
end

@inline function get_speed_vperp_inner_views_z(iz, vperp_advect, Bmag, dBdr, dBdz, vperp,
                                               rfac, r_advect, alpha_advect, z_advect,
                                               evolve_density::Val{true},
                                               evolve_upar::Val, evolve_p::Val{false})
    return @views vperp_advect[:,:,iz], Bmag[iz], dBdr[iz], dBdz[iz], vperp, rfac,
                  r_advect[:,:,iz], alpha_advect[:,:,iz], z_advect[:,:,iz],
                  evolve_density, evolve_upar, evolve_p
end

@inline function get_speed_vperp_inner_views_vperp(ivperp, vperp_advect, Bmag, dBdr, dBdz,
                                                   vperp, rfac, r_advect, alpha_advect,
                                                   z_advect, evolve_density::Val{true},
                                                   evolve_upar::Val, evolve_p::Val{false})
    return @views vperp_advect[:,ivperp], Bmag, dBdr, dBdz, vperp[ivperp], rfac,
                  r_advect[:,ivperp], alpha_advect[:,ivperp], z_advect[:,ivperp],
                  evolve_density, evolve_upar, evolve_p
end

function update_speed_vperp_inner!(vperp_advect, Bmag, dBdr, dBdz, vperp, rfac, r_advect,
                                   alpha_advect, z_advect, evolve_density::Val{true},
                                   evolve_upar::Val, evolve_p::Val{false})
    @. vperp_advect = 0.5 * vperp / Bmag * ((z_advect + alpha_advect) * dBdz
                                            + rfac * r_advect * dBdr)
end

@inline function get_speed_vperp_inner_views_sr(is, ir, vperp_advect, Bmag, dBdr, dBdz,
                                                vperp, rfac, r_advect, alpha_advect,
                                                z_advect, evolve_density::Val{false},
                                                evolve_upar::Val{false},
                                                evolve_p::Val{false})
    return @views vperp_advect[:,:,:,ir,is], Bmag[:,ir], dBdr[:,ir], dBdz[:,ir], vperp,
                  rfac, r_advect[:,:,:,ir,is], alpha_advect[:,:,:,ir,is],
                  z_advect[:,:,:,ir,is], evolve_density, evolve_upar, evolve_p
end

@inline function get_speed_vperp_inner_views_z(iz, vperp_advect, Bmag, dBdr, dBdz, vperp,
                                               rfac, r_advect, alpha_advect, z_advect,
                                               evolve_density::Val{false},
                                               evolve_upar::Val{false},
                                               evolve_p::Val{false})
    return @views vperp_advect[:,:,iz], Bmag[iz], dBdr[iz], dBdz[iz], vperp, rfac,
                  r_advect[:,:,iz], alpha_advect[:,:,iz], z_advect[:,:,iz],
                  evolve_density, evolve_upar, evolve_p
end

@inline function get_speed_vperp_inner_views_vperp(ivperp, vperp_advect, Bmag, dBdr, dBdz,
                                                   vperp, rfac, r_advect, alpha_advect,
                                                   z_advect, evolve_density::Val{false},
                                                   evolve_upar::Val{false},
                                                   evolve_p::Val{false})
    return @views vperp_advect[:,ivperp], Bmag, dBdr, dBdz, vperp[ivperp], rfac,
                  r_advect[:,ivperp], alpha_advect[:,ivperp], z_advect[:,ivperp],
                  evolve_density, evolve_upar, evolve_p
end

function update_speed_vperp_inner!(vperp_advect, Bmag, dBdr, dBdz, vperp, rfac, r_advect,
                                   alpha_advect, z_advect, evolve_density::Val{false},
                                   evolve_upar::Val{false}, evolve_p::Val{false})
    # advection of vperp due to conservation of 
    # the adiabatic invariant mu = vperp^2 / 2 B
    @. vperp_advect = (0.5 * vperp / Bmag) * ((alpha_advect + z_advect) * dBdz + rfac * r_advect * dBdr)

    return nothing
end

function update_z_alpha_r_speeds!(z_advect, alpha_advect, r_advect, fvec_in, moments,
                                  fields, geometry, vpa, vperp, z, r)
    update_z_speed = (z.n == 1 && geometry.input.option == "0D-Spitzer-test")
    if update_z_speed
        # make sure z speed is physical despite
        # 0D nature of simulation
        @begin_s_r_vperp_vpa_region()
        # get the updated speed along the z direction using the current f
        @views update_speed_z!(z_advect, fvec_in.upar, moments.ion.vth,
                               moments.evolve_upar, moments.evolve_ppar, vpa, vperp, z, r,
                               geometry)
        # get the updated speed along the binormal direction using the current f
        @views update_speed_alpha!(alpha_advect, moments.evolve_upar, moments.evolve_ppar,
                                   fields, vpa, vperp, z, r, geometry)
    end
    
    update_r_speed = (r.n == 1 && geometry.input.option == "0D-Spitzer-test")
    if update_r_speed
        # make sure r speed is physical despite
        # 0D nature of simulation
        @begin_s_z_vperp_vpa_region()
        # get the updated speed along the r direction using the current f
        @views update_speed_r!(r_advect, fields, moments.evolve_upar, moments.evolve_ppar,
                               vpa, vperp, z, r, geometry)
    end
    return nothing
end    

end
