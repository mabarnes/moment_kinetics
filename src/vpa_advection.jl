module vpa_advection

export vpa_advection!
export update_speed_vpa!

using ..semi_lagrange: find_approximate_characteristic!
using ..advection: update_boundary_indices!
using ..advection: advance_f_local!
using ..communication
using ..calculus: derivative!
using ..initial_conditions: enforce_vpa_boundary_condition!
using ..looping

function vpa_advection!(f_out, fvec_in, ff, fields, moments, SL, advect,
        vpa, z, use_semi_lagrange, dt, t, vpa_spectral, z_spectral, composition, CX_frequency, istage)

    # only have a parallel acceleration term for neutrals if using the peculiar velocity
    # wpar = vpar - upar as a variable; i.e., d(wpar)/dt /=0 for neutrals even though d(vpar)/dt = 0.

    # calculate the advection speed corresponding to current f
    update_speed_vpa!(advect, fields, fvec_in, moments, vpa, z, composition, CX_frequency, t, z_spectral)
    @s_z_loop_s is begin
        if !moments.evolve_upar && is in composition.neutral_species_range
            # No acceleration for neutrals when not evolving upar
            continue
        end
        # update the upwind/downwind boundary indices and upwind_increment
        # NB: not sure if this will work properly with SL method at the moment
        # NB: if the speed is actually time-dependent
        update_boundary_indices!(advect[is], loop_ranges[].s_z_range_z)
        # if using interpolation-free Semi-Lagrange,
        # follow characteristics backwards in time from level m+1 to level m
        # to get departure points.  then find index of grid point nearest
        # the departure point at time level m and use this to define
        # an approximate characteristic
        if use_semi_lagrange
            @s_z_loop_z iz begin
                find_approximate_characteristic!(SL, advect[is], iz, vpa, dt)
            end
        end
        @s_z_loop_z iz begin
            @views advance_f_local!(f_out[:,iz,is], fvec_in.pdf[:,iz,is], ff[:,iz,is],
                                    SL, advect[is], iz, vpa, dt, istage,
                                    vpa_spectral, use_semi_lagrange)
        end
        #@views enforce_vpa_boundary_condition!(f_out[:,:,is], vpa.bc, advect[is])
    end
end
# calculate the advection speed in the z-direction at each grid point
function update_speed_vpa!(advect, fields, fvec, moments, vpa, z, composition, CX_frequency, t, z_spectral)
    @boundscheck z.n == size(advect[1].speed,2) || throw(BoundsError(advect))
    #@boundscheck composition.n_ion_species == size(advect,2) || throw(BoundsError(advect))
    @boundscheck composition.n_species == size(advect,1) || throw(BoundsError(advect))
    @boundscheck vpa.n == size(advect[1].speed,1) || throw(BoundsError(speed))
    if vpa.advection.option == "default"
        # dvpa/dt = Ze/m ⋅ E_parallel
        update_speed_default!(advect, fields, fvec, moments, vpa, z, composition, CX_frequency, t, z_spectral)
    elseif vpa.advection.option == "constant"
        @serial_region begin
            # Not usually used - just run in serial
            #
            # dvpa/dt = constant
            s_range = ifelse(moments.evolve_upar, 1:composition.n_species,
                             composition.ion_species_range)
            for is ∈ s_range
                update_speed_constant!(advect[is], vpa, z_range)
            end
        end
        block_sychronize()
    elseif vpa.advection.option == "linear"
        @serial_region begin
            # Not usually used - just run in serial
            #
            # dvpa/dt = constant ⋅ (vpa + L_vpa/2)
            s_range = ifelse(moments.evolve_upar, 1:composition.n_species,
                             composition.ion_species_range)
            for is ∈ s_range
                update_speed_linear!(advect[is], vpa, z_range)
            end
        end
        block_sychronize()
    end
    @s_z_loop_s is begin
        if !moments.evolve_upar && is in composition.neutral_species_range
            # No acceleration for neutrals when not evolving upar
            continue
        end
        @s_z_loop_z iz begin
            @views @. advect[is].modified_speed[:,iz] = advect[is].speed[:,iz]
        end
    end
    return nothing
end
function update_speed_default!(advect, fields, fvec, moments, vpa, z, composition, CX_frequency, t, z_spectral)
    if moments.evolve_ppar
        @s_z_loop_s is begin
            # get d(ppar)/dz
            derivative!(z.scratch, view(fvec.ppar,:,is), z, z_spectral)
            # update parallel acceleration to account for parallel derivative of parallel pressure
            # NB: no vpa-dependence so compute as a scalar and broadcast to all entries
            @s_z_loop_z iz begin
                @views advect[is].speed[:,iz] .= z.scratch[iz]/(fvec.density[iz,is]*moments.vth[iz,is])
            end
            # calculate d(qpar)/dz
            derivative!(z.scratch, view(moments.qpar,:,is), z, z_spectral)
            # update parallel acceleration to account for (wpar/2*ppar)*dqpar/dz
            @s_z_loop_z iz begin
                @views @. advect[is].speed[:,iz] += 0.5*vpa.grid*z.scratch[iz]/fvec.ppar[iz,is]
            end
            # calculate d(vth)/dz
            derivative!(z.scratch, view(moments.vth,:,is), z, z_spectral)
            # update parallel acceleration to account for -wpar^2 * d(vth)/dz term
            @s_z_loop_z iz begin
                @views @. advect[is].speed[:,iz] -= vpa.grid^2*z.scratch[iz]
            end
        end
        # add in contributions from charge exchange collisions
        if composition.n_neutral_species > 0 && abs(CX_frequency) > 0.0
            @s_z_loop_s is begin
                if is ∈ composition.ion_species_range
                    for isp ∈ composition.neutral_species_range
                        @s_z_loop_z iz begin
                            @views @. advect[is].speed[:,iz] += CX_frequency *
                            (0.5*vpa.grid/fvec.ppar[iz,is] * (fvec.density[iz,isp]*fvec.ppar[iz,is]
                                                              - fvec.density[iz,is]*fvec.ppar[iz,isp])
                             - fvec.density[iz,isp] * (fvec.upar[iz,isp]-fvec.upar[iz,is])/moments.vth[iz,is])
                        end
                    end
                end
                if is ∈ composition.neutral_species_range
                    for isp ∈ composition.ion_species_range
                        @s_z_loop_z iz begin
                            @views @. advect[is].speed[:,iz] += CX_frequency *
                            (0.5*vpa.grid/fvec.ppar[iz,is] * (fvec.density[iz,isp]*fvec.ppar[iz,is]
                                                              - fvec.density[iz,is]*fvec.ppar[iz,isp])
                             - fvec.density[iz,isp] * (fvec.upar[iz,isp]-fvec.upar[iz,is])/moments.vth[iz,is])
                        end
                    end
                end
            end
        end
    elseif moments.evolve_upar
        @s_z_loop_s is begin
            # get d(ppar)/dz
            derivative!(z.scratch, view(fvec.ppar,:,is), z, z_spectral)
            # update parallel acceleration to account for parallel derivative of parallel pressure
            # NB: no vpa-dependence so compute as a scalar and broadcast to all entries
            @s_z_loop_z iz begin
                @views advect[is].speed[:,iz] .= z.scratch[iz]/fvec.density[iz,is]
            end
            # calculate d(upar)/dz
            derivative!(z.scratch, view(fvec.upar,:,is), z, z_spectral)
            # update parallel acceleration to account for -wpar*dupar/dz
            @s_z_loop_z iz begin
                @views @. advect[is].speed[:,iz] -= vpa.grid*z.scratch[iz]
            end
        end
        # if neutrals present and charge exchange frequency non-zero,
        # account for collisional friction between ions and neutrals
        if composition.n_neutral_species > 0 && abs(CX_frequency) > 0.0
            # include contribution to ion acceleration due to collisional friction with neutrals
            @s_z_loop_s is begin
                if is ∈ composition.ion_species_range
                    for isp ∈ composition.neutral_species_range
                        @s_z_loop_z iz begin
                            @views advect[is].speed[:,iz] .+= -CX_frequency*fvec.density[iz,isp]*(fvec.upar[iz,isp]-fvec.upar[iz,is])
                        end
                    end
                end
                # include contribution to neutral acceleration due to collisional friction with ions
                if is ∈ composition.neutral_species_range
                    for isp ∈ composition.ion_species_range
                        # get the absolute species index for the neutral species
                        @s_z_loop_z iz begin
                            @views advect[is].speed[:,iz] .+= -CX_frequency*fvec.density[iz,isp]*(fvec.upar[iz,isp]-fvec.upar[iz,is])
                        end
                    end
                end
            end
        end
    else
        # update the electrostatic potential phi
        # calculate the derivative of phi with respect to z;
        # the value at element boundaries is taken to be the average of the values
        # at neighbouring elements
        derivative!(z.scratch, fields.phi, z, z_spectral)
        # advection velocity in vpa is -dphi/dz = -z.scratch
        @inbounds @fastmath begin
            @s_z_loop_s is begin
                if !moments.evolve_upar && is in composition.neutral_species_range
                    # No acceleration for neutrals when not evolving upar
                    continue
                end
                @s_z_loop_z iz begin
                    @views advect[is].speed[:,iz] .= -0.5*z.scratch[iz]
                end
            end
        end
    end
end
# update the advection speed dvpa/dt = constant
function update_speed_constant!(advect, vpa, z_range)
    #@inbounds @fastmath begin
    for iz ∈ z_range
        @views advect.speed[:,iz] .= vpa.advection.constant_speed
    end
    #end
end
# update the advection speed dvpa/dt = const*(vpa + L/2)
function update_speed_linear(advect, vpa, z_range)
    @inbounds @fastmath begin
        for iz ∈ z_range
            @views @. advect.speed[:,iz] = vpa.advection.constant_speed*(vpa.grid+0.5*vpa.L)
        end
    end
end

end
