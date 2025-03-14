module vperp_advection

export vperp_advection!
export update_speed_vperp!

using ..advection: advance_f_local!
using ..chebyshev: chebyshev_info
using ..looping
using ..timer_utils
using ..z_advection: update_speed_z!
using ..r_advection: update_speed_r!

# do a single stage time advance (potentially as part of a multi-stage RK scheme)
@timeit global_timer vperp_advection!(
                         f_out, fvec_in, vperp_advect, r, z, vperp, vpa, dt,
                         vperp_spectral, composition, z_advect, r_advect, geometry,
                         moments, fields, t) = begin
    
    # if appropriate, update z and r speeds
    update_z_r_speeds!(z_advect, r_advect, fvec_in, moments, fields,
                            geometry, vpa, vperp, z, r, t)
    
    @begin_s_r_z_vpa_region()
    @loop_s is begin
        # get the updated speed along the vperp direction using the current f
        @views update_speed_vperp!(vperp_advect[is], vpa, vperp, z, r, z_advect[is], r_advect[is], geometry)
        @loop_r_z_vpa ir iz ivpa begin
            @views advance_f_local!(f_out[ivpa,:,iz,ir,is], fvec_in.pdf[ivpa,:,iz,ir,is],
                                    vperp_advect[is], ivpa, iz, ir, vperp, dt, vperp_spectral)            
        end
    end
end

# calculate the advection speed in the vperp-direction at each grid point
# note that the vperp advection speed depends on the z and r advection speeds
# It is important to ensure that z_advect and r_advect are updated before vperp_advect
function update_speed_vperp!(vperp_advect, vpa, vperp, z, r, z_advect, r_advect, geometry)
    @boundscheck z.n == size(vperp_advect.speed,3) || throw(BoundsError(vperp_advect))
    @boundscheck vperp.n == size(vperp_advect.speed,1) || throw(BoundsError(vperp_advect))
    @boundscheck vpa.n == size(vperp_advect.speed,2) || throw(BoundsError(vperp_advect))
    @boundscheck r.n == size(vperp_advect.speed,4) || throw(BoundsError(vperp_speed))
    if vperp.advection.option == "default"
    # advection of vperp due to conservation of 
    # the adiabatic invariant mu = vperp^2 / 2 B
        dzdt = vperp.scratch
        drdt = vperp.scratch2
        dBdr = geometry.dBdr
        dBdz = geometry.dBdz
        Bmag = geometry.Bmag
        rfac = 0.0
        if r.n > 1
            rfac = 1.0
        end
        @inbounds begin
            @loop_r_z_vpa ir iz ivpa begin
                @. @views dzdt = z_advect.speed[iz,ivpa,:,ir]
                @. @views drdt = rfac*r_advect.speed[ir,ivpa,:,iz]
                @. @views vperp_advect.speed[:,ivpa,iz,ir] = (0.5*vperp.grid[:]/Bmag[iz,ir])*(dzdt[:]*dBdz[iz,ir] + drdt[:]*dBdr[iz,ir])
            end
        end
    elseif vperp.advection.option == "constant"
        @inbounds begin
            @loop_r_z_vpa ir iz ivpa begin
                @views vperp_advect.speed[:,ivpa,iz,ir] .= vperp.advection.constant_speed
            end
        end
    end
    return nothing
end

function update_z_r_speeds!(z_advect, r_advect, fvec_in, moments, fields,
                            geometry, vpa, vperp, z, r, t)
    update_z_speed = (z.n == 1 && geometry.input.option == "0D-Spitzer-test")
    if update_z_speed
    # make sure z speed is physical despite
    # 0D nature of simulation
        @begin_s_r_vperp_vpa_region()
        @loop_s is begin
            # get the updated speed along the z direction using the current f
            @views update_speed_z!(z_advect[is], fvec_in.upar[:,:,is],
                                   moments.ion.vth[:,:,is], moments.evolve_upar,
                                   moments.evolve_ppar, fields, vpa, vperp, z, r, t, geometry, is)
        end
    end
    
    update_r_speed = (r.n == 1 && geometry.input.option == "0D-Spitzer-test")
    if update_r_speed
    # make sure r speed is physical despite
    # 0D nature of simulation
        @begin_s_z_vperp_vpa_region()
        @loop_s is begin
            # get the updated speed along the r direction using the current f
            update_speed_r!(r_advect[is], fvec_in.upar[:,:,is],
                               moments.ion.vth[:,:,is], fields, moments.evolve_upar,
                               moments.evolve_ppar, vpa, vperp, z, r, geometry, is)
        end
    end
    return nothing
end    

end
