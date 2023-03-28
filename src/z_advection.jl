"""
"""
module z_advection

export z_advection!
export update_speed_z!

using ..advection: advance_f_local!
using ..chebyshev: chebyshev_info
using ..looping

"""
do a single stage time advance (potentially as part of a multi-stage RK scheme)
"""
function z_advection!(f_out, fvec_in, ff, moments, advect, z, vpa, r, dt, t, spectral,
                      composition)

    begin_s_r_vpa_region()

    @loop_s is begin
        # get the updated speed along the z direction using the current f
        @views update_speed_z!(advect[is], fvec_in.upar[:,:,is], moments.vth[:,:,is],
                               moments.evolve_upar, moments.evolve_ppar, vpa, z, r, t)
        # # advance z-advection equation
        # if moments.evolve_density
        #     for ivpa ∈ 1:vpa.n
        #         @views @. advect[is].speed[:,ivpa] /= fvec_in.density[:,is]
        #         @views advance_f_local!(f_out[:,ivpa,is], fvec_in.density[:,is] .* fvec_in.pdf[:,ivpa,is],
        #             ff[:,ivpa,is], advect[is], ivpa, z, dt, spectral)
        #     end
        # else
        #     for ivpa ∈ 1:vpa.n
        #         @views advance_f_local!(f_out[:,ivpa,is], fvec_in.pdf[:,ivpa,is],
        #             ff[:,ivpa,is], advect[is], ivpa, z, dt, spectral)
        #     end
        # end
        # advance z-advection equation
        @loop_r_vpa ir ivpa begin
            @views adjust_advection_speed!(advect[is].speed[:,ivpa,ir],
                                           fvec_in.density[:,ir,is], moments.vth[:,ir,is],
                                           moments.evolve_density, moments.evolve_ppar)
            # take the normalized pdf contained in fvec_in.pdf and remove the normalization,
            # returning the true (un-normalized) particle distribution function in z.scratch
            @views unnormalize_pdf!(z.scratch, fvec_in.pdf[ivpa,:,ir,is], fvec_in.density[:,ir,is], moments.vth[:,ir,is],
                                    moments.evolve_density, moments.evolve_ppar)
            @views advance_f_local!(f_out[ivpa,:,ir,is], z.scratch, ff[ivpa,:,ir,is], advect[is], ivpa, ir,
                                    z, dt, spectral)
        end
    end
end

"""
"""
function adjust_advection_speed!(speed, dens, vth, evolve_density, evolve_ppar)
    if evolve_ppar
        for i in eachindex(speed)
            speed[i] *= vth[i]/dens[i]
        end
    elseif evolve_density
        for i in eachindex(speed)
            speed[i] /= dens[i]
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
    return nothing
end

end
