"""
"""
module z_advection

export z_advection!
export update_speed_z!

using ..semi_lagrange: find_approximate_characteristic!
using ..advection: advance_f_local!, update_boundary_indices!
using ..chebyshev: chebyshev_info
using ..looping

"""
do a single stage time advance (potentially as part of a multi-stage RK scheme)
"""
function z_advection!(f_out, fvec_in, ff, moments, SL, advect, z, vpa, r,
                      use_semi_lagrange, dt, t, spectral, composition, istage)

    begin_s_r_vpa_region()

    @loop_s is begin
        # get the updated speed along the z direction using the current f
        @views update_speed_z!(advect[is], fvec_in.upar[:,:,is], moments.vth[:,:,is],
                               moments.evolve_upar, moments.evolve_ppar, vpa, z, r, t)
        # update the upwind/downwind boundary indices and upwind_increment
        @views update_boundary_indices!(advect[is], loop_ranges[].vpa, loop_ranges[].r)
        # if using interpolation-free Semi-Lagrange,
        # follow characteristics backwards in time from level m+1 to level m
        # to get departure points.  then find index of grid point nearest
        # the departure point at time level m and use this to define
        # an approximate characteristic
        if use_semi_lagrange
            # MRH NOT SUPPORTED
            @loop_r_vpa ir ivpa begin
                find_approximate_characteristic!(SL[ivpa], advect[is], ivpa, ir, z, dt)
            end
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
        # advance z-advection equation
        @loop_r_vpa ir ivpa begin
            @views adjust_advection_speed!(advect[is].speed[:,ivpa,ir], advect[is].modified_speed[:,ivpa,ir],
                                           fvec_in.density[:,ir,is], moments.vth[:,ir,is],
                                           moments.evolve_density, moments.evolve_ppar)
            # take the normalized pdf contained in fvec_in.pdf and remove the normalization,
            # returning the true (un-normalized) particle distribution function in z.scratch
            @views unnormalize_pdf!(z.scratch, fvec_in.pdf[ivpa,:,ir,is], fvec_in.density[:,ir,is], moments.vth[:,ir,is],
                                    moments.evolve_density, moments.evolve_ppar)
            @views advance_f_local!(f_out[ivpa,:,ir,is], z.scratch, ff[ivpa,:,ir,is], SL, advect[is], ivpa, ir,
                                    z, dt, istage, spectral, use_semi_lagrange)
        end
    end
end

"""
"""
function adjust_advection_speed!(speed, mod_speed, dens, vth, evolve_density, evolve_ppar)
    if evolve_ppar
        for i in eachindex(speed)
            factor = vth[i]/dens[i]
            speed[i] *= factor
            mod_speed[i] *= factor
        end
    elseif evolve_density
        for i in eachindex(speed)
            factor = 1.0 / dens[i]
            speed[i] *= factor
            mod_speed[i] *= factor
        end
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
calculate the advection speed in the z-direction at each grid point
"""
function update_speed_z!(advect, upar, vth, evolve_upar, evolve_ppar, vpa, z, r, t)
    @boundscheck r.n == size(advect.speed,3) || throw(BoundsError(advect))
    @boundscheck vpa.n == size(advect.speed,2) || throw(BoundsError(advect))
    @boundscheck z.n == size(advect.speed,1) || throw(BoundsError(speed))
    if z.advection.option == "default"
        @inbounds begin
            @loop_r_vpa ir ivpa begin
                @. @views advect.speed[:,ivpa,ir] = vpa.grid[ivpa]
            end
            if evolve_ppar
                @loop_r_vpa ir ivpa begin
                    @. @views advect.speed[:,ivpa,ir] = advect.speed[:,ivpa,ir] * vth
                end
            end
            if evolve_upar
                @loop_r_vpa ir ivpa begin
                    @. @views advect.speed[:,ivpa,ir] += upar
                end
            end
        end
    elseif z.advection.option == "constant"
        @inbounds begin
            @loop_r_vpa ir ivpa begin
                @views advect.speed[:,ivpa,ir] .= z.advection.constant_speed
            end
        end
    elseif z.advection.option == "linear"
        @inbounds begin
            @loop_r_vpa ir ivpa begin
                @views advect.speed[:,ivpa,ir] .= z.advection.constant_speed*(z.grid[i]+0.5*z.L)
            end
        end
    elseif z.advection.option == "oscillating"
        @inbounds begin
            @loop_r_vpa ir ivpa begin
                @views advect.speed[:,ivpa,ir] .= z.advection.constant_speed*(1.0
                        + z.advection.oscillation_amplitude*sinpi(t*z.advection.frequency))
            end
        end
    end
    # the default for modified_speed is simply speed.
    # will be modified later if semi-Lagrange scheme used
    @inbounds begin
        @loop_r_vpa ir ivpa begin
            @views advect.modified_speed[:,ivpa,ir] .= advect.speed[:,ivpa,ir]
        end
    end
    return nothing
end

end
