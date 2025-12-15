"""
"""
module z_advection

export z_advection!
export update_speed_z!

using ..advection: advance_f_df_precomputed!
using ..chebyshev: chebyshev_info
using ..debugging
using ..looping
using ..timer_utils
using ..derivatives: derivative_z!

"""
do a single stage time advance (potentially as part of a multi-stage RK scheme)
"""
@timeit global_timer z_advection!(
                         f_out, fvec_in, moments, fields, advect, z, vpa, vperp, r, dt, t,
                         spectral, composition, geometry, scratch_dummy) = begin

    @begin_s_r_vperp_vpa_region()

    # get the updated speed along the z direction using the current f
    update_speed_z!(advect, fvec_in.upar, moments.ion.vth, moments.evolve_upar,
                    moments.evolve_p, vpa, vperp, z, r, geometry)
    #calculate the upwind derivative
    derivative_z!(scratch_dummy.buffer_vpavperpzrs_1, fvec_in.pdf, advect,
                  scratch_dummy.buffer_vpavperprs_1, scratch_dummy.buffer_vpavperprs_2,
                  scratch_dummy.buffer_vpavperprs_3, scratch_dummy.buffer_vpavperprs_4,
                  scratch_dummy.buffer_vpavperprs_5, scratch_dummy.buffer_vpavperprs_6,
                  spectral, z)

    # advance z-advection equation
    @loop_s_r_vperp_vpa is ir ivperp ivpa begin
        @. @views z.scratch = scratch_dummy.buffer_vpavperpzrs_1[ivpa,ivperp,:,ir,is]
        @views advance_f_df_precomputed!(f_out[ivpa,ivperp,:,ir,is], z.scratch,
                                         advect[:,ivpa,ivperp,ir,is], z, dt)
    end
end

"""
calculate the advection speed in the z-direction at each grid point
"""
function update_speed_z!(advect, upar, vth, evolve_upar, evolve_p, vpa, vperp, z, r,
                         geometry)
    @debug_consistency_checks r.n == size(advect,4) || throw(BoundsError(advect))
    @debug_consistency_checks vperp.n == size(advect,3) || throw(BoundsError(advect))
    @debug_consistency_checks vpa.n == size(advect,2) || throw(BoundsError(advect))
    @debug_consistency_checks z.n == size(advect,1) || throw(BoundsError(advect))

    bzed = geometry.bzed
    if evolve_p
        @loop_s_r_vperp_vpa is ir ivperp ivpa begin
            @. @views advect[:,ivpa,ivperp,ir,is] = (vth[:,ir,is] * vpa.grid[ivpa] + upar[:,ir,is]) * bzed[:,ir]
        end
    elseif evolve_upar
        @loop_s_r_vperp_vpa is ir ivperp ivpa begin
            @. @views advect[:,ivpa,ivperp,ir,is] = (vpa.grid[ivpa] + upar[:,ir,is]) * bzed[:,ir]
        end
    else
        @loop_s_r_vperp_vpa is ir ivperp ivpa begin
            # vpa bzed
            @. @views advect[:,ivpa,ivperp,ir,is] = vpa.grid[ivpa]*bzed[:,ir]
        end
    end

    return nothing
end

end
