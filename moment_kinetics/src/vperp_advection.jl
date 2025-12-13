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

    speed_args = get_speed_vperp_inner_args(vperp_advect, moments, geometry, r_advect,
                                            alpha_advect, z_advect, r, vperp,
                                            evolve_density, evolve_upar, evolve_p)
    f_in = fvec_in.pdf
    @begin_s_r_z_vpa_region()
    @loop_s_r is ir begin
        speed_args_sr = get_speed_vperp_inner_views_sr(is, ir, speed_args...)
        this_f_out = @view f_out[:,:,:,ir,is]
        this_f_in = @view f_in[:,:,:,ir,is]
        @loop_z iz begin
            speed_args_z = get_speed_vperp_inner_views_z(iz, speed_args_sr...)
            this_iz_f_out = @view this_f_out[:,:,iz]
            this_iz_f_in = @view this_f_in[:,:,iz]
            @loop_vpa ivpa begin
                speed_args_vperp = get_speed_vperp_inner_views_vpa(ivpa, speed_args_z...)
                # get the updated speed along the vperp direction using the current f
                update_speed_vperp_inner!(speed_args_vperp...)
                @views advance_f_local!(this_iz_f_out[ivpa,:], this_iz_f_in[ivpa,:],
                                        first(speed_args_vperp), vperp, dt,
                                        vperp_spectral)
            end
        end
    end

    return nothing
end

# calculate the advection speed in the vperp-direction at each grid point
# note that the vperp advection speed depends on the z and r advection speeds
# It is important to ensure that z_advect and r_advect are updated before vperp_advect
function update_speed_vperp!(vperp_advect, fvec, vpa, vperp, z, r, z_advect, alpha_advect,
                             r_advect, geometry, moments)
    return update_speed_vperp!(vperp_advect, fvec, vpa, vperp, z, r, z_advect, alpha_advect,
                             r_advect, geometry, moments, Val(moments.evolve_density),
                             Val(moments.evolve_upar), Val(moments.evolve_p))
end
function update_speed_vperp!(vperp_advect, fvec, vpa, vperp, z, r, z_advect, alpha_advect,
                             r_advect, geometry, moments, evolve_density::Val,
                             evolve_upar::Val, evolve_p::Val)
    @debug_consistency_checks r.n == size(vperp_advect,4) || throw(BoundsError(vperp_advect))
    @debug_consistency_checks z.n == size(vperp_advect,3) || throw(BoundsError(vperp_advect))
    @debug_consistency_checks vperp.n == size(vperp_advect,2) || throw(BoundsError(vperp_advect))
    @debug_consistency_checks vpa.n == size(vperp_advect,1) || throw(BoundsError(vperp_advect))
    @begin_s_r_z_vpa_region()
    speed_args = get_speed_vperp_inner_args(vperp_advect, moments, geometry, r_advect,
                                            alpha_advect, z_advect, r, vperp,
                                            evolve_density, evolve_upar, evolve_p)
    @loop_s_r is ir begin
        speed_args_sr = get_speed_vperp_inner_views_sr(is, ir, speed_args...)
        @loop_z iz begin
            speed_args_z = get_speed_vperp_inner_views_z(iz, speed_args_sr...)
            @loop_vpa ivpa begin
                update_speed_vperp_inner!(get_speed_vperp_inner_views_vpa(ivpa, speed_args_z...)...)
            end
        end
    end

    return nothing
end

@inline function get_speed_vperp_inner_args(vperp_advect, moments, geometry, r_advect, alpha_advect,
                                            z_advect, r, vperp, evolve_density::Val,
                                            evolve_upar::Val, evolve_p::Val)
    if evolve_p === Val(true)
        return vperp_advect, moments.ion.vth, moments.ion.dvth_dt, moments.ion.dvth_dr,
               moments.ion.dvth_dz, r_advect, alpha_advect, z_advect, vperp.grid,
               evolve_density, evolve_upar, evolve_p
    elseif evolve_upar === Val(true)
        return vperp_advect, evolve_density, evolve_upar, evolve_p
    elseif evolve_density === Val(true)
        return vperp_advect, evolve_density, evolve_upar, evolve_p
    else
        rfac = r.n > 1 ? 1.0 : 0.0
        return vperp_advect, geometry.Bmag, geometry.dBdr, geometry.dBdz, vperp.grid,
               rfac, r_advect, alpha_advect, z_advect, evolve_density, evolve_upar,
               evolve_p
    end
end

@inline function get_speed_vperp_inner_views_sr(is, ir, vperp_advect, vth, dvth_dt,
                                                dvth_dr, dvth_dz, r_advect, alpha_advect,
                                                z_advect, wperp,
                                                evolve_density::Val{true},
                                                evolve_upar::Val{true},
                                                evolve_p::Val{true})
    return @views vperp_advect[:,:,:,ir,is], vth[:,ir,is], dvth_dt[:,ir,is],
                  dvth_dr[:,ir,is], dvth_dz[:,ir,is], r_advect[:,:,:,ir,is],
                  alpha_advect[:,:,:,ir,is], z_advect[:,:,:,ir,is], wperp, evolve_density,
                  evolve_upar, evolve_p
end

@inline function get_speed_vperp_inner_views_z(iz, vperp_advect, vth, dvth_dt, dvth_dr,
                                               dvth_dz, r_advect, alpha_advect, z_advect,
                                               wperp, evolve_density::Val{true},
                                               evolve_upar::Val{true},
                                               evolve_p::Val{true})
    return @views vperp_advect[:,:,iz], vth[iz], dvth_dt[iz], dvth_dr[iz], dvth_dz[iz],
                  r_advect[:,:,iz], alpha_advect[:,:,iz], z_advect[:,:,iz], wperp,
                  evolve_density, evolve_upar, evolve_p
end

@inline function get_speed_vperp_inner_views_vpa(ivpa, vperp_advect, vth, dvth_dt,
                                                 dvth_dr, dvth_dz, r_advect, alpha_advect,
                                                 z_advect, wperp,
                                                 evolve_density::Val{true},
                                                 evolve_upar::Val{true},
                                                 evolve_p::Val{true})
    return @views vperp_advect[ivpa,:], vth, dvth_dt, dvth_dr, dvth_dz, r_advect[ivpa,:],
                  alpha_advect[ivpa,:], z_advect[ivpa,:], wperp, evolve_density,
                  evolve_upar, evolve_p
end

"""
update vperp advection speed when n, u, p are evolved separately
"""
function update_speed_vperp_inner!(vperp_advect, vth, dvth_dt, dvth_dr, dvth_dz, r_advect,
                                   alpha_advect, z_advect, wperp,
                                   evolve_density::Val{true}, evolve_upar::Val{true},
                                   evolve_p::Val{true})
    # update perpendicular advection speed, which is only nonzero because of the
    # normalisation by thermal speed, so wperp grid is constantly stretching and
    # compressing to account for changing local temperatures while maintaining a
    # normalised perpendicular speed.
    @. vperp_advect =
           - (1/vth) * wperp * (
               dvth_dt
               + r_advect * dvth_dr
               + (alpha_advect + z_advect) * dvth_dz)

    return nothing
end

@inline function get_speed_vperp_inner_views_sr(is, ir, vperp_advect,
                                                evolve_density::Val{true},
                                                evolve_upar::Val,
                                                evolve_p::Val{false})
    return @views vperp_advect[:,:,:,ir,is], evolve_density, evolve_upar, evolve_p
end

@inline function get_speed_vperp_inner_views_z(iz, vperp_advect,
                                               evolve_density::Val{true},
                                               evolve_upar::Val, evolve_p::Val{false})
    return @views vperp_advect[:,:,iz], evolve_density, evolve_upar, evolve_p
end

@inline function get_speed_vperp_inner_views_vpa(ivpa, vperp_advect,
                                                 evolve_density::Val{true},
                                                 evolve_upar::Val,
                                                 evolve_p::Val{false})
    return @views vperp_advect[ivpa,:], evolve_density, evolve_upar, evolve_p
end

"""
update vperp advection speed when n, u are evolved separately, or only n is evolved separately.
u does not affect the wperp coordinate, so these two cases can be handled together.
"""
function update_speed_vperp_inner!(vperp_advect, evolve_density::Val{true},
                                   evolve_upar::Val, evolve_p::Val{false})
    # with no perpendicular advection terms, the advection speed is zero
    return nothing
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

@inline function get_speed_vperp_inner_views_vpa(ivpa, vperp_advect, Bmag, dBdr, dBdz,
                                                 vperp, rfac, r_advect, alpha_advect,
                                                 z_advect, evolve_density::Val{false},
                                                 evolve_upar::Val{false},
                                                 evolve_p::Val{false})
    return @views vperp_advect[ivpa,:], Bmag, dBdr, dBdz, vperp, rfac, r_advect[ivpa,:],
                  alpha_advect[ivpa,:], z_advect[ivpa,:], evolve_density, evolve_upar,
                  evolve_p
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

function get_ion_vperp_advection_term_evolve_nup(sub_terms::IonSubTerms)
    vth = sub_terms.vth
    dvth_dt = sub_terms.dvth_dt
    dvth_dr = sub_terms.dvth_dr
    dvth_dz = sub_terms.dvth_dz
    wperp = sub_terms.wperp
    r_speed = sub_terms.r_speed
    alpha_speed = sub_terms.alpha_speed
    z_speed = sub_terms.z_speed
    df_dvperp = sub_terms.df_dvperp

    speed = -vth^(-1) * wperp * (dvth_dt
                                 + r_speed * dvth_dr
                                 + (alpha_speed + z_speed) * dvth_dz)
    term = speed * df_dvperp

    return term
end

end
