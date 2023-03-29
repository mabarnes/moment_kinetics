"""
"""
module vpa_advection

export vpa_advection!
export update_speed_vpa!

using ..advection: advance_f_df_precomputed!
using ..calculus: derivative!
using ..communication
using ..looping

"""
"""
function vpa_advection!(f_out, fvec_in, fields, advect,
        vpa, vperp, z, r, dt, vpa_spectral, composition, geometry)

    begin_s_r_z_vperp_region()
    
    # calculate the advection speed corresponding to current f
    update_speed_vpa!(advect, fields, vpa, vperp, z, r, composition, geometry)
    @loop_s_r_z_vperp is ir iz ivperp begin
        # update advection factor 
        @views @. advect[is].adv_fac[:,ivperp,iz,ir] = -dt*advect[is].speed[:,ivperp,iz,ir]
        # calculate the upwind derivative along vpa
        @views derivative!(vpa.scratch, fvec_in.pdf[:,ivperp,iz,ir,is],
                        vpa, advect[is].adv_fac[:,ivperp,iz,ir], vpa_spectral)
    	# advance vpa-advection equation
        @views advance_f_df_precomputed!(f_out[:,ivperp,iz,ir,is],
          vpa.scratch, advect[is], ivperp, iz, ir, vpa, dt, vpa_spectral)
    end
    
end

"""
calculate the advection speed in the vpa-direction at each grid point
"""
function update_speed_vpa!(advect, fields, vpa, vperp, z, r, composition, geometry)
    @boundscheck r.n == size(advect[1].speed,4) || throw(BoundsError(advect))
    @boundscheck z.n == size(advect[1].speed,3) || throw(BoundsError(advect))
    @boundscheck vperp.n == size(advect[1].speed,2) || throw(BoundsError(advect))
    #@boundscheck composition.n_ion_species == size(advect,2) || throw(BoundsError(advect))
    @boundscheck composition.n_ion_species == size(advect,1) || throw(BoundsError(advect))
    @boundscheck vpa.n == size(advect[1].speed,1) || throw(BoundsError(speed))
    if vpa.advection.option == "default"
        # dvpa/dt = Ze/m ⋅ E_parallel
        update_speed_default!(advect, fields, vpa, vperp, z, r, composition, geometry)
    elseif vpa.advection.option == "constant"
        @serial_region begin
            # Not usually used - just run in serial
            # dvpa/dt = constant
            for is ∈ 1:composition.n_ion_species
                update_speed_constant!(advect[is], vpa, 1:vperp.n, 1:z.n, 1:r.n)
            end
        end
        block_sychronize()
    elseif vpa.advection.option == "linear"
        @serial_region begin
            # Not usually used - just run in serial
            # dvpa/dt = constant ⋅ (vpa + L_vpa/2)
            for is ∈ 1:composition.n_ion_species
                update_speed_linear!(advect[is], vpa, 1:vperp.n, 1:z.n, 1:r.n)
            end
        end
        block_sychronize()
    end
    @loop_s is begin
        @loop_r_z_vperp ir iz ivperp begin
            @views @. advect[is].modified_speed[:,ivperp,iz,ir] = advect[is].speed[:,ivperp,iz,ir]
        end
    end
    return nothing
end

"""
"""
function update_speed_default!(advect, fields, vpa, vperp, z, r, composition, geometry)
    bzed = geometry.bzed
    @inbounds @fastmath begin
        @loop_s is begin
            @loop_r ir begin
                # bzed = B_z/B
                @loop_z_vperp iz ivperp begin
                    @views advect[is].speed[:,ivperp,iz,ir] .= 0.5*bzed*fields.Ez[iz,ir]
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
