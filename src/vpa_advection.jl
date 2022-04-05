"""
"""
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

"""
"""
function vpa_advection!(f_out, fvec_in, ff, fields, moments, SL, advect,
        vpa, vperp, z, r, use_semi_lagrange, dt, t,
        vpa_spectral, z_spectral, composition, CX_frequency, istage)

    # only have a parallel acceleration term for neutrals if using the peculiar velocity
    # wpar = vpar - upar as a variable; i.e., d(wpar)/dt /=0 for neutrals even though d(vpar)/dt = 0.

    # calculate the advection speed corresponding to current f
    update_speed_vpa!(advect, fields, fvec_in, moments, vpa, vperp,
    z, r, composition, CX_frequency, t, z_spectral)
    @loop_s is begin
        if !moments.evolve_upar && is in composition.neutral_species_range
            # No acceleration for neutrals when not evolving upar
            continue
        end
        # update the upwind/downwind boundary indices and upwind_increment
        update_boundary_indices!(advect[is], loop_ranges[].vperp, loop_ranges[].z, loop_ranges[].r)

        @loop_r_z_vperp ir iz ivperp begin
            @views advance_f_local!(f_out[:,ivperp,iz,ir,is], fvec_in.pdf[:,ivperp,iz,ir,is],
                                    ff[:,ivperp,iz,ir,is],
                                    SL, advect[is], ivperp, iz, ir, vpa, dt, istage,
                                    vpa_spectral, use_semi_lagrange)
        end
        #@views enforce_vpa_boundary_condition!(f_out[:,:,is], vpa.bc, advect[is])
    end
end

"""
calculate the advection speed in the vpa-direction at each grid point
"""
function update_speed_vpa!(advect, fields, fvec, moments, vpa, vperp, z, r, composition, CX_frequency, t, z_spectral)
    @boundscheck r.n == size(advect[1].speed,4) || throw(BoundsError(advect))
    @boundscheck z.n == size(advect[1].speed,3) || throw(BoundsError(advect))
    @boundscheck vperp.n == size(advect[1].speed,2) || throw(BoundsError(advect))
    #@boundscheck composition.n_ion_species == size(advect,2) || throw(BoundsError(advect))
    @boundscheck composition.n_species == size(advect,1) || throw(BoundsError(advect))
    @boundscheck vpa.n == size(advect[1].speed,1) || throw(BoundsError(speed))
    if vpa.advection.option == "default"
        # dvpa/dt = Ze/m ⋅ E_parallel
        update_speed_default!(advect, fields, fvec, moments, vpa, vperp, z, r, composition, CX_frequency, t, z_spectral)
    elseif vpa.advection.option == "constant"
        @serial_region begin
            # Not usually used - just run in serial
            #
            # dvpa/dt = constant
            s_range = ifelse(moments.evolve_upar, 1:composition.n_species,
                             composition.ion_species_range)
            for is ∈ s_range
                update_speed_constant!(advect[is], vpa, 1:vperp.n, 1:z.n, 1:r.n)
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
                update_speed_linear!(advect[is], vpa, 1:vperp.n, 1:z.n, 1:r.n)
            end
        end
        block_sychronize()
    end
    @loop_s is begin
        if !moments.evolve_upar && is in composition.neutral_species_range
            # No acceleration for neutrals when not evolving upar
            continue
        end
        @loop_r_z_vperp ir iz ivperp begin
            @views @. advect[is].modified_speed[:,ivperp,iz,ir] = advect[is].speed[:,ivperp,iz,ir]
        end
    end
    return nothing
end

"""
"""
function update_speed_default!(advect, fields, fvec, moments, vpa, vperp, z, r, composition, CX_frequency, t, z_spectral)

    @inbounds @fastmath begin
        @loop_s is begin
            if !moments.evolve_upar && is in composition.neutral_species_range
                # No acceleration for neutrals when not evolving upar
                continue
            end
            @loop_r ir begin
                # update the electrostatic potential phi
                # calculate the derivative of phi with respect to z;
                # the value at element boundaries is taken to be the average of the values
                # at neighbouring elements
                derivative!(z.scratch, view(fields.phi,:,ir), z, z_spectral)
                # advection velocity in vpa is -dphi/dz = -z.scratch
                @loop_z_vperp iz ivperp begin
                    @views advect[is].speed[:,ivperp,iz,ir] .= -0.5*z.scratch[iz]
                end
            end
        end
    end

end

"""
update the advection speed dvpa/dt = constant
"""
function update_speed_constant!(advect, vpa, vperp_range, z_range, r_range)
    #@inbounds @fastmath begin
    for ir ∈ r_range
        for iz ∈ z_range
            for ivperp ∈ vperp_range
                @views advect.speed[:,ivperp,iz,ir] .= vpa.advection.constant_speed
            end
        end
    end
    #end
end

"""
update the advection speed dvpa/dt = const*(vpa + L/2)
"""
function update_speed_linear(advect, vpa, z_range, r_range)
    @inbounds @fastmath begin
        for ir ∈ r_range
            for iz ∈ z_range
                for ivperp ∈ vperp_range
                    @views @. advect.speed[:,ivperp,iz,ir] = vpa.advection.constant_speed*(vpa.grid+0.5*vpa.L)
                end
            end
        end
    end
end

end
