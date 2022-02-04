"""
"""
module r_advection

export r_advection!
export update_speed_r!

using ..semi_lagrange: find_approximate_characteristic!
using ..advection: advance_f_local!, update_boundary_indices!
using ..chebyshev: chebyshev_info
using ..looping

"""
do a single stage time advance (potentially as part of a multi-stage RK scheme)
"""
function r_advection!(f_out, fvec_in, ff, moments, SL, advect, r, z, vpa,
                      use_semi_lagrange, dt, t, spectral, composition, istage)
    @loop_s is begin
        # get the updated speed along the r direction using the current f
        @views update_speed_r!(advect[is], fvec_in.upar[:,:,is], moments.vth[:,:,is],
                               moments.evolve_upar, moments.evolve_ppar, vpa, z, r, t)
        # update the upwind/downwind boundary indices and upwind_increment
        @views update_boundary_indices!(advect[is], loop_ranges[].vpa, loop_ranges[].z)

        # if using interpolation-free Semi-Lagrange,
        # follow characteristics backwards in time from level m+1 to level m
        # to get departure points.  then find index of grid point nearest
        # the departure point at time level m and use this to define
        # an approximate characteristic
        if use_semi_lagrange
            print("SL NOT SUPPORTED in: function r_advection!")
        end
        # # advance z-advection equation
        # if moments.evolve_density
        #     for ivpa ∈ 1:vpa.n
        #         @views @. advect[is].speed[:,ivpa] /= fvec_in.density[:,is]
        #         @views @. advect[is].modified_speed[:,ivpa] /= fvec_in.density[:,is]
        #         @views advance_f_local!(f_out[:,ivpa,is], fvec_in.density[:,is] .* fvec_in.pdf[:,ivpa,is],
        #             ff[:,ivpa,is], SL[ivpa], advect[is], ivpa, z, dt, istage, spectral, use_semi_lagrange)
        #     end
        # else
        #     for ivpa ∈ 1:vpa.n
        #         @views advance_f_local!(f_out[:,ivpa,is], fvec_in.pdf[:,ivpa,is],
        #             ff[:,ivpa,is], SL[ivpa], advect[is], ivpa, z, dt, istage, spectral, use_semi_lagrange)
        #     end
        # end
        # advance r-advection equation
        @loop_z_vpa iz ivpa begin
            @views adjust_advection_speed!(advect[is].speed[:,ivpa,iz], advect[is].modified_speed[:,ivpa,iz],
                                           fvec_in.density[iz,:,is], moments.vth[iz,:,is],
                                           moments.evolve_density, moments.evolve_ppar)
            # take the normalized pdf contained in fvec_in.pdf and remove the normalization,
            # returning the true (un-normalized) particle distribution function in r.scratch

            @views unnormalize_pdf!(r.scratch, fvec_in.pdf[ivpa,iz,:,is], fvec_in.density[iz,:,is], moments.vth[iz,:,is],
                                    moments.evolve_density, moments.evolve_ppar)
            @views advance_f_local!(f_out[ivpa,iz,:,is], r.scratch, ff[ivpa,iz,:,is], SL, advect[is], ivpa,
                                    r, dt, istage, spectral, use_semi_lagrange)
        end
    end
end

"""
"""
function adjust_advection_speed!(speed, mod_speed, dens, vth, evolve_density, evolve_ppar)
    if evolve_ppar
        @. speed *= vth/dens
        @. mod_speed *= vth/dens
    elseif evolve_density
        @. speed /= dens
        @. mod_speed /= dens
    end
    return nothing
end

"""
"""
function unnormalize_pdf!(unnorm, norm, dens, vth, evolve_density, evolve_ppar)
    if evolve_ppar
        @. unnorm = norm * dens/vth
    elseif evolve_density
        @. unnorm = norm * dens
    else
        @. unnorm = norm
    end
    return nothing
end

"""
calculate the advection speed in the r-direction at each grid point
"""
function update_speed_r!(advect, upar, vth, evolve_upar, evolve_ppar, vpa, z, r, t)
    @boundscheck z.n == size(advect.speed,3) || throw(BoundsError(advect))
    @boundscheck vpa.n == size(advect.speed,2) || throw(BoundsError(advect))
    @boundscheck r.n == size(advect.speed,1) || throw(BoundsError(speed))
    if r.advection.option == "default" || r.advection.option == "constant"
        @inbounds begin
            @loop_z_vpa iz ivpa begin
                @views advect.speed[:,ivpa,iz] .= r.advection.constant_speed
            end
        end
    end
    # the default for modified_speed is simply speed.
    # will be modified later if semi-Lagrange scheme used
    @inbounds begin
        @loop_z_vpa iz ivpa begin
            @views advect.modified_speed[:,ivpa,iz] .= advect.speed[:,ivpa,iz]
        end
    end
    return nothing
end

end
