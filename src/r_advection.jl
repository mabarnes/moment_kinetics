"""
"""
module r_advection

export r_advection!
export update_speed_r!

using ..advection: advance_f_local!
using ..chebyshev: chebyshev_info
using ..looping

"""
do a single stage time advance (potentially as part of a multi-stage RK scheme)
"""
function r_advection!(f_out, fvec_in, ff, moments, advect, r, z, vpa, dt, t, spectral,
                      composition)
    @loop_s is begin
        # get the updated speed along the r direction using the current f
        @views update_speed_r!(advect[is], fvec_in.upar[:,:,is], moments.vth[:,:,is],
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
        # advance r-advection equation
        @loop_z_vpa iz ivpa begin
            @views adjust_advection_speed!(advect[is].speed[:,ivpa,iz],
                                           fvec_in.density[iz,:,is], moments.vth[iz,:,is],
                                           moments.evolve_density, moments.evolve_ppar)
            # take the normalized pdf contained in fvec_in.pdf and remove the normalization,
            # returning the true (un-normalized) particle distribution function in r.scratch

            @views unnormalize_pdf!(r.scratch, fvec_in.pdf[ivpa,iz,:,is], fvec_in.density[iz,:,is], moments.vth[iz,:,is],
                                    moments.evolve_density, moments.evolve_ppar)
            @views advance_f_local!(f_out[ivpa,iz,:,is], r.scratch, ff[ivpa,iz,:,is], advect[is], ivpa,
                                    r, dt, spectral)
        end
    end
end

"""
"""
function adjust_advection_speed!(speed, dens, vth, evolve_density, evolve_ppar)
    if evolve_ppar
        @. speed *= vth/dens
    elseif evolve_density
        @. speed /= dens
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
    return nothing
end

end
