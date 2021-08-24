module z_advection

export z_advection!
export update_speed_z!

using ..semi_lagrange: find_approximate_characteristic!
using ..advection: advance_f_local!, update_boundary_indices!
using ..chebyshev: chebyshev_info
using ..optimization
using TimerOutputs
using ..moment_kinetics: global_timer

# do a single stage time advance (potentially as part of a multi-stage RK scheme)
function z_advection!(f_out, fvec_in, ff, moments, SL_vec, advect, z_vec, vpa_vec,
                      use_semi_lagrange, dt, t, spectral_vec, n_species, istage)
    nvpa = vpa_vec[1].n
    @timeit global_timer "z_advection pre-loop" begin
    @outerloop for is ∈ 1:n_species
        ithread = threadid()
        z = z_vec[ithread]
        vpa = vpa_vec[ithread]
        # get the updated speed along the z direction using the current f
        @views update_speed_z!(advect[:,is], fvec_in.upar[:,is], moments.vth[:,is],
                               moments.evolve_upar, moments.evolve_ppar, vpa, z, t)
        # update the upwind/downwind boundary indices and upwind_increment
        @views update_boundary_indices!(advect[:,is])
    end
    end
    if use_semi_lagrange
        # if using interpolation-free Semi-Lagrange,
        # follow characteristics backwards in time from level m+1 to level m
        # to get departure points.  then find index of grid point nearest
        # the departure point at time level m and use this to define
        # an approximate characteristic
        @outerloop for is ∈ 1:n_species, ivpa ∈ 1:nvpa
            find_approximate_characteristic!(SL[ivpa], advect[ivpa,is], z, dt)
        end
    end
    @timeit global_timer "z_advection" begin
    @outerloop for is ∈ 1:n_species, ivpa ∈ 1:nvpa
        ithread = threadid()
        SL = SL_vec[ithread]
        z = z_vec[ithread]
        vpa = vpa_vec[ithread]
        spectral = spectral_vec[ithread]
        @views adjust_advection_speed!(advect[ivpa,is].speed, advect[ivpa,is].modified_speed,
                                       fvec_in.density[:,is], moments.vth[:,is],
                                       moments.evolve_density, moments.evolve_ppar)
        # take the normalized pdf contained in fvec_in.pdf and remove the normalization,
        # returning the true (un-normalized) particle distribution function in z.scratch
        @views unnormalize_pdf!(z.scratch, fvec_in.pdf[ivpa,:,is], fvec_in.density[:,is], moments.vth[:,is],
                                moments.evolve_density, moments.evolve_ppar)
        @views advance_f_local!(f_out[ivpa,:,is], z.scratch, ff[ivpa,:,is], SL[ivpa], advect[ivpa,is],
                                z, dt, istage, spectral, use_semi_lagrange)
    end
    end
end
function adjust_advection_speed!(speed, mod_speed, dens, vth, evolve_density, evolve_ppar)
    if evolve_ppar
        @. speed *= vth/dens
        @. mod_speed *= vth/dens
    elseif evolve_density
        @. speed /= dens
        @. mod_speed /= dens
    end
end
function unnormalize_pdf!(unnorm, norm, dens, vth, evolve_density, evolve_ppar)
    if evolve_ppar
        @. unnorm = norm * dens/vth
    elseif evolve_density
        @. unnorm = norm * dens
    else
        @. unnorm = norm
    end
end
# calculate the advection speed in the z-direction at each grid point
function update_speed_z!(advect, upar, vth, evolve_upar, evolve_ppar, vpa, z, t)
    @boundscheck vpa.n == size(advect,1) || throw(BoundsError(advect))
    @boundscheck z.n == size(advect[1].speed,1) || throw(BoundsError(speed))
    if z.advection.option == "default"
        @inbounds begin
            for ivpa ∈ 1:vpa.n
                advect[ivpa].speed .= vpa.grid[ivpa]
            end
            if evolve_upar
                if evolve_ppar
                    for ivpa ∈ 1:vpa.n
                        advect[ivpa].speed .*= vth
                    end
                end
                for ivpa ∈ 1:vpa.n
                    advect[ivpa].speed .+= upar
                end
            end
        end
    elseif z.advection.option == "constant"
        @inbounds begin
            for ivpa ∈ 1:vpa.n
                advect[ivpa].speed .= z.advection.constant_speed
            end
        end
    elseif z.advection.option == "linear"
        @inbounds begin
            for ivpa ∈ 1:vpa.n
                advect[ivpa].speed .= z.advection.constant_speed*(z.grid[i]+0.5*z.L)
            end
        end
    elseif z.advection.option == "oscillating"
        @inbounds begin
            for ivpa ∈ 1:vpa.n
                advect[ivpa].speed .= z.advection.constant_speed*(1.0
                        + z.advection.oscillation_amplitude*sinpi(t*z.advection.frequency))
            end
        end
    end
    # the default for modified_speed is simply speed.
    # will be modified later if semi-Lagrange scheme used
    @inbounds begin
        for ivpa ∈ 1:vpa.n
            advect[ivpa].modified_speed .= advect[ivpa].speed
        end
    end
    return nothing
end

end
