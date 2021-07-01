# do a single stage time advance (potentially as part of a multi-stage RK scheme)
function z_advection!(f_out, fvec_in, ff, moments, SL, advect, z, vpa,
                      use_semi_lagrange, dt, t, spectral, n_species, istage)
    for is ∈ 1:n_species
        # get the updated speed along the z direction using the current f
        @views update_speed_z!(advect[:,is], fvec_in.upar[:,is], moments.vth[:,is],
                               moments.evolve_upar, moments.evolve_ppar, vpa, z, t)
        # update the upwind/downwind boundary indices and upwind_increment
        @views update_boundary_indices!(advect[:,is])
        # if using interpolation-free Semi-Lagrange,
        # follow characteristics backwards in time from level m+1 to level m
        # to get departure points.  then find index of grid point nearest
        # the departure point at time level m and use this to define
        # an approximate characteristic
        if use_semi_lagrange
            for ivpa ∈ 1:vpa.n
                find_approximate_characteristic!(SL[ivpa], advect[ivpa,is], z, dt)
            end
        end
        # # advance z-advection equation
        # if moments.evolve_density
        #     for ivpa ∈ 1:vpa.n
        #         @. advect[ivpa,is].speed /= fvec_in.density[:,is]
        #         @. advect[ivpa,is].modified_speed /= fvec_in.density[:,is]
        #         @views advance_f_local!(f_out[:,ivpa,is], fvec_in.density[:,is] .* fvec_in.pdf[:,ivpa,is],
        #             ff[:,ivpa,is], SL[ivpa], advect[ivpa,is], z, dt, istage, spectral, use_semi_lagrange)
        #     end
        # else
        #     for ivpa ∈ 1:vpa.n
        #         @views advance_f_local!(f_out[:,ivpa,is], fvec_in.pdf[:,ivpa,is],
        #             ff[:,ivpa,is], SL[ivpa], advect[ivpa,is], z, dt, istage, spectral, use_semi_lagrange)
        #     end
        # end
        # advance z-advection equation
        for ivpa ∈ 1:vpa.n
            @views adjust_advection_speed!(advect[ivpa,is].speed, advect[ivpa,is].modified_speed,
                                           fvec_in.density[:,is], moments.vth[:,is],
                                           moments.evolve_density, moments.evolve_ppar)
            # take the normalized pdf contained in fvec_in.pdf and remove the normalization,
            # returning the true (un-normalized) particle distribution function in z.scratch
            @views unnormalize_pdf!(z.scratch, fvec_in.pdf[:,ivpa,is], fvec_in.density[:,is], moments.vth[:,is],
                                    moments.evolve_density, moments.evolve_ppar)
            @views advance_f_local!(f_out[:,ivpa,is], z.scratch, ff[:,ivpa,is], SL[ivpa], advect[ivpa,is],
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
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    advect[j].speed[i] = vpa.grid[j]
                end
            end
            if evolve_upar
                if evolve_ppar
                    for j ∈ 1:vpa.n
                        @. advect[j].speed *= vth
                    end
                end
                for j ∈ 1:vpa.n
                    @. advect[j].speed += upar
                end
            end
        end
    elseif z.advection.option == "constant"
        @inbounds begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    advect[j].speed[i] = z.advection.constant_speed
                end
            end
        end
    elseif z.advection.option == "linear"
        @inbounds begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    advect[j].speed[i] = z.advection.constant_speed*(z.grid[i]+0.5*z.L)
                end
            end
        end
    elseif z.advection.option == "oscillating"
        @inbounds begin
            for j ∈ 1:vpa.n
                for i ∈ 1:z.n
                    advect[j].speed[i] = z.advection.constant_speed*(1.0
                        + z.advection.oscillation_amplitude*sinpi(t*z.advection.frequency))
                end
            end
        end
    end
    # the default for modified_speed is simply speed.
    # will be modified later if semi-Lagrange scheme used
    @inbounds begin
        for j ∈ 1:vpa.n
            @. advect[j].modified_speed = advect[j].speed
        end
    end
    return nothing
end
