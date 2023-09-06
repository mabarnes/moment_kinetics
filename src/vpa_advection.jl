"""
"""
module vpa_advection

export vpa_advection!
export update_speed_vpa!

using ..advection: advance_f_local!
using ..communication
using ..looping

"""
"""
function vpa_advection!(f_out, fvec_in, fields, moments, advect, vpa, vperp, z, r, dt, t,
                        vpa_spectral, composition, collisions, geometry)

    begin_s_r_z_vperp_region()

    # only have a parallel acceleration term for neutrals if using the peculiar velocity
    # wpar = vpar - upar as a variable; i.e., d(wpar)/dt /=0 for neutrals even though d(vpar)/dt = 0.

    # calculate the advection speed corresponding to current f
    update_speed_vpa!(advect, fields, fvec_in, moments, vpa, vperp, z, r, composition, collisions, t, geometry)
    @loop_s is begin
        @loop_r_z_vperp ir iz ivperp begin
            @views advance_f_local!(f_out[:,ivperp,iz,ir,is], fvec_in.pdf[:,ivperp,iz,ir,is],
                                    advect[is], ivperp, iz, ir, vpa, dt, vpa_spectral)
        end
    end
end

"""
calculate the advection speed in the vpa-direction at each grid point
"""
function update_speed_vpa!(advect, fields, fvec, moments, vpa, vperp, z, r, composition, collisions, t, geometry)
    @boundscheck r.n == size(advect[1].speed,4) || throw(BoundsError(advect))
    @boundscheck z.n == size(advect[1].speed,3) || throw(BoundsError(advect))
    @boundscheck vperp.n == size(advect[1].speed,2) || throw(BoundsError(advect))
    #@boundscheck composition.n_ion_species == size(advect,2) || throw(BoundsError(advect))
    @boundscheck composition.n_ion_species == size(advect,1) || throw(BoundsError(advect))
    @boundscheck vpa.n == size(advect[1].speed,1) || throw(BoundsError(speed))
    if vpa.advection.option == "default"
        # dvpa/dt = Ze/m ⋅ E_parallel
        update_speed_default!(advect, fields, fvec, moments, vpa, z, r, composition, collisions, t, geometry)
    elseif vpa.advection.option == "constant"
        begin_serial_region()
        @serial_region begin
            # Not usually used - just run in serial
            # dvpa/dt = constant
            for is ∈ 1:composition.n_ion_species
                update_speed_constant!(advect[is], vpa, 1:vperp.n, 1:z.n, 1:r.n)
            end
        end
    elseif vpa.advection.option == "linear"
        begin_serial_region()
        @serial_region begin
            # Not usually used - just run in serial
            # dvpa/dt = constant ⋅ (vpa + L_vpa/2)
            for is ∈ 1:composition.n_ion_species
                update_speed_linear!(advect[is], vpa, 1:vperp.n, 1:z.n, 1:r.n)
            end
        end
    end
    return nothing
end

"""
"""
function update_speed_default!(advect, fields, fvec, moments, vpa, z, r, composition, collisions, t, geometry)
    if moments.evolve_ppar && moments.evolve_upar
        update_speed_n_u_p_evolution!(advect, fvec, moments, vpa, z, r, composition, collisions)
    elseif moments.evolve_ppar
        update_speed_n_p_evolution!(advect, fields, fvec, moments, vpa, z, r, composition, collisions)
    elseif moments.evolve_upar
        update_speed_n_u_evolution!(advect, fvec, moments, vpa, z, r, composition, collisions)
    else
        bzed = geometry.bzed
        @inbounds @fastmath begin
            @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
                # bzed = B_z/B
                advect[is].speed[ivpa,ivperp,iz,ir] = 0.5*bzed*fields.Ez[iz,ir]
            end
        end
    end
end

"""
update the advection speed in the parallel velocity coordinate for the case
where density, flow and pressure are evolved independently from the pdf;
in this case, the parallel velocity coordinate is the normalized peculiar velocity
wpahat = (vpa - upar)/vth
"""
function update_speed_n_u_p_evolution!(advect, fvec, moments, vpa, z, r, composition, collisions)
    @loop_s is begin
        @loop_r ir begin
            # update parallel acceleration to account for:
            # • parallel derivative of parallel pressure
            # • (wpar/2*ppar)*dqpar/dz
            # • -wpar^2 * d(vth)/dz term
            @loop_z_vperp iz ivperp begin
                @views @. advect[is].speed[:,ivperp,iz,ir] =
                    moments.charged.dppar_dz[iz,ir,is]/(fvec.density[iz,ir,is]*moments.charged.vth[iz,ir,is]) +
                    0.5*vpa.grid*moments.charged.dqpar_dz[iz,ir,is]/fvec.ppar[iz,ir,is] -
                    vpa.grid^2*moments.charged.dvth_dz[iz,ir,is]
            end
        end
    end
    # add in contributions from charge exchange and ionization collisions
    if composition.n_neutral_species > 0 &&
            (abs(collisions.charge_exchange) > 0.0 || abs(collisions.ionization) > 0.0)

        @loop_s is begin
            @loop_r_z_vperp ir iz ivperp begin
                @views @. advect[is].speed[:,ivperp,iz,ir] +=
                    collisions.charge_exchange *
                    (0.5*vpa.grid/fvec.ppar[iz,ir,is]
                     * (fvec.density_neutral[iz,ir,is]*fvec.ppar[iz,ir,is]
                        - fvec.density[iz,ir,is]*fvec.pz_neutral[iz,ir,is]
                        - fvec.density[iz,ir,is]*fvec.density_neutral[iz,ir,is]
                          * (fvec.upar[iz,ir,is]-fvec.uz_neutral[iz,ir,is])^2)
                     - fvec.density_neutral[iz,ir,is]
                       * (fvec.uz_neutral[iz,ir,is]-fvec.upar[iz,ir,is])
                       / moments.charged.vth[iz,ir,is]) +
                    collisions.ionization *
                    (0.5*vpa.grid
                       * (fvec.density_neutral[iz,ir,is]
                          - fvec.density[iz,ir,is]*fvec.pz_neutral[iz,ir,is]
                            / fvec.ppar[iz,ir,is]
                          - fvec.density[iz,ir,is]*fvec.density_neutral[iz,ir,is]
                            * (fvec.uz_neutral[iz,ir,is] - fvec.upar[iz,ir,is])^2
                            / fvec.ppar[iz,ir,is])
                     - fvec.density_neutral[iz,ir,is]
                       * (fvec.uz_neutral[iz,ir,is] - fvec.upar[iz,ir,is])
                       / moments.charged.vth[iz,ir,is])
            end
        end
    end
end

"""
update the advection speed in the parallel velocity coordinate for the case
where density and pressure are evolved independently from the pdf;
in this case, the parallel velocity coordinate is the normalized velocity
vpahat = vpa/vth
"""
function update_speed_n_p_evolution!(advect, fields, fvec, moments, vpa, z, r, composition, collisions)
    @loop_s is begin
        # include contributions common to both ion and neutral species
        @loop_r ir begin
            # update parallel acceleration to account for:
            # • (vpahat/2*ppar)*dqpar/dz
            # • vpahat*(upar/vth-vpahat) * d(vth)/dz term
            # • vpahat*d(upar)/dz
            # • -(1/2)*(dphi/dz)/vthi
            @loop_z_vperp iz ivperp begin
                @views @. advect[is].speed[:,ivperp,iz,ir] = 0.5*vpa.grid*moments.charged.dqpar_dz[iz,ir,is]/fvec.ppar[iz,ir,is] +
                                                             vpa.grid*moments.charged.dvth_dz[iz] * (fvec.upar[iz,ir,is]/moments.vth[iz,ir,is] - vpa.grid) +
                                                             vpa.grid*moments.charged.dupar_dz[iz,ir,is] +
                                                             0.5*fields.Ez[iz,ir]/moments.vth[iz,ir,is]
            end
        end
    end
    # add in contributions from charge exchange and ionization collisions
    if composition.n_neutral_species > 0
        error("suspect the charge exchange and ionization contributions here may be "
              * "wrong because (upar[is]-upar[isp])^2 type terms were missed in the "
              * "energy equation when it was substituted in to derive them.")
        if abs(collisions.charge_exchange + collisions.ionization) > 0.0
            @loop_s is begin
                @loop_r_z_vperp ir iz ivperp begin
                    @views @. advect[is].speed[:,ivperp,iz,ir] += (collisions.charge_exchange + collisions.ionization) *
                            0.5*vpa.grid*fvec.density[iz,ir,is] * (1.0-fvec.pz_neutral[iz,ir,is]/fvec.ppar[iz,ir,is])
                end
            end
        end
    end
end

"""
update the advection speed in the parallel velocity coordinate for the case
where density and flow are evolved independently from the pdf;
in this case, the parallel velocity coordinate is the peculiar velocity
wpa = vpa-upar
"""
function update_speed_n_u_evolution!(advect, fvec, moments, vpa, z, r, composition, collisions)
    @loop_s is begin
        @loop_r ir begin
            # update parallel acceleration to account for:
            # • parallel derivative of parallel pressure
            # • -wpar*dupar/dz
            @loop_z_vperp iz ivperp begin
                @views @. advect[is].speed[:,ivperp,iz,ir] =
                    moments.charged.dppar_dz[iz,ir,is]/fvec.density[iz,ir,is] -
                    vpa.grid*moments.charged.dupar_dz[iz,ir,is]
            end
        end
    end
    # if neutrals present compute contribution to parallel acceleration due to charge exchange
    # and/or ionization collisions betweens ions and neutrals
    if composition.n_neutral_species > 0
        # account for collisional charge exchange friction between ions and neutrals
        if abs(collisions.charge_exchange) > 0.0
            @loop_s is begin
                @loop_r_z_vperp ir iz ivperp begin
                    @views @. advect[is].speed[:,ivperp,iz,ir] -= collisions.charge_exchange*fvec.density_neutral[iz,ir,is]*(fvec.uz_neutral[iz,ir,is]-fvec.upar[iz,ir,is])
                end
            end
        end
        if abs(collisions.ionization) > 0.0
            @loop_s is begin
                @loop_r_z_vperp ir iz ivperp begin
                    @views @. advect[is].speed[:,ivperp,iz,ir] -= collisions.ionization*fvec.density_neutral[iz,ir,is]*(fvec.uz_neutral[iz,ir,is]-fvec.upar[iz,ir,is])
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
function update_speed_linear(advect, vpa, vperp_range, z_range, r_range)
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
