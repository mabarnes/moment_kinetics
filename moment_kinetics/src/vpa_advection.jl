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
                        vpa_spectral, composition, collisions, ion_source_settings, geometry)

    begin_s_r_z_vperp_region()

    # only have a parallel acceleration term for neutrals if using the peculiar velocity
    # wpar = vpar - upar as a variable; i.e., d(wpar)/dt /=0 for neutrals even though d(vpar)/dt = 0.

    # calculate the advection speed corresponding to current f
    update_speed_vpa!(advect, fields, fvec_in, moments, vpa, vperp, z, r, composition,
                      collisions, ion_source_settings, t, geometry)
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
function update_speed_vpa!(advect, fields, fvec, moments, vpa, vperp, z, r, composition,
                           collisions, ion_source_settings, t, geometry)
    @boundscheck r.n == size(advect[1].speed,4) || throw(BoundsError(advect))
    @boundscheck z.n == size(advect[1].speed,3) || throw(BoundsError(advect))
    @boundscheck vperp.n == size(advect[1].speed,2) || throw(BoundsError(advect))
    #@boundscheck composition.n_ion_species == size(advect,2) || throw(BoundsError(advect))
    @boundscheck composition.n_ion_species == size(advect,1) || throw(BoundsError(advect))
    @boundscheck vpa.n == size(advect[1].speed,1) || throw(BoundsError(speed))
    if vpa.advection.option == "default"
        # dvpa/dt = Ze/m ⋅ E_parallel - (vperp^2/2B) bz dB/dz
        # magnetic mirror term only supported for standard DK implementation
        update_speed_default!(advect, fields, fvec, moments, vpa, vperp, z, r, composition,
                              collisions, ion_source_settings, t, geometry)
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
function update_speed_default!(advect, fields, fvec, moments, vpa, vperp, z, r, composition,
                               collisions, ion_source_settings, t, geometry)
    if moments.evolve_ppar && moments.evolve_upar
        update_speed_n_u_p_evolution!(advect, fvec, moments, vpa, z, r, composition,
                                      collisions, ion_source_settings)
    elseif moments.evolve_ppar
        update_speed_n_p_evolution!(advect, fields, fvec, moments, vpa, z, r, composition,
                                    collisions, ion_source_settings)
    elseif moments.evolve_upar
        update_speed_n_u_evolution!(advect, fvec, moments, vpa, z, r, composition,
                                    collisions, ion_source_settings)
    else
        bzed = geometry.bzed
        dBdz = geometry.dBdz
        Bmag = geometry.Bmag
        @inbounds @fastmath begin
            @loop_s_r_z_vperp_vpa is ir iz ivperp ivpa begin
                # mu, the adiabatic invariant
                mu = 0.5*(vperp.grid[ivperp]^2)/Bmag[iz,ir]
                # bzed = B_z/B
                advect[is].speed[ivpa,ivperp,iz,ir] = (0.5*bzed[iz,ir]*fields.Ez[iz,ir] - 
                                                       mu*bzed[iz,ir]*dBdz[iz,ir])
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
function update_speed_n_u_p_evolution!(advect, fvec, moments, vpa, z, r, composition,
                                       collisions, ion_source_settings)
    @loop_s is begin
        @loop_r ir begin
            # update parallel acceleration to account for:
            # • parallel derivative of parallel pressure
            # • (wpar/2*ppar)*dqpar/dz
            # • -wpar^2 * d(vth)/dz term
            @loop_z_vperp iz ivperp begin
                @views @. advect[is].speed[:,ivperp,iz,ir] =
                    moments.ion.dppar_dz[iz,ir,is]/(fvec.density[iz,ir,is]*moments.ion.vth[iz,ir,is]) +
                    0.5*vpa.grid*moments.ion.dqpar_dz[iz,ir,is]/fvec.ppar[iz,ir,is] -
                    vpa.grid^2*moments.ion.dvth_dz[iz,ir,is]
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
                       / moments.ion.vth[iz,ir,is]) +
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
                       / moments.ion.vth[iz,ir,is])
            end
        end
    end
    if ion_source_settings.active
        source_density_amplitude = moments.ion.external_source_density_amplitude
        source_momentum_amplitude = moments.ion.external_source_momentum_amplitude
        source_pressure_amplitude = moments.ion.external_source_pressure_amplitude
        density = fvec.density
        upar = fvec.upar
        ppar = fvec.ppar
        vth = moments.ion.vth
        vpa_grid = vpa.grid
        @loop_s_r_z is ir iz begin
            term1 = source_density_amplitude[iz,ir] * upar[iz,ir,is]/(density[iz,ir,is]*vth[iz,ir,is])
            term2_over_vpa =
                -0.5 * (source_pressure_amplitude[iz,ir] +
                        2.0 * upar[iz,ir,is] * source_momentum_amplitude[iz,ir]) /
                       ppar[iz,ir,is] +
                0.5 * source_density_amplitude[iz,ir] / density[iz,ir,is]
            @loop_vperp_vpa ivperp ivpa begin
                advect[is].speed[ivpa,ivperp,iz,ir] += term1 + vpa_grid[ivpa] * term2_over_vpa
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
function update_speed_n_p_evolution!(advect, fields, fvec, moments, vpa, z, r,
                                     composition, collisions, ion_source_settings)
    @loop_s is begin
        # include contributions common to both ion and neutral species
        @loop_r ir begin
            # update parallel acceleration to account for:
            # • (vpahat/2*ppar)*dqpar/dz
            # • vpahat*(upar/vth-vpahat) * d(vth)/dz term
            # • vpahat*d(upar)/dz
            # • -(1/2)*(dphi/dz)/vthi
            @loop_z_vperp iz ivperp begin
                @views @. advect[is].speed[:,ivperp,iz,ir] = 0.5*vpa.grid*moments.ion.dqpar_dz[iz,ir,is]/fvec.ppar[iz,ir,is] +
                                                             vpa.grid*moments.ion.dvth_dz[iz] * (fvec.upar[iz,ir,is]/moments.vth[iz,ir,is] - vpa.grid) +
                                                             vpa.grid*moments.ion.dupar_dz[iz,ir,is] +
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
    if ion_source_settings.active
        error("External source not implemented for evolving n and ppar case")
    end
end

"""
update the advection speed in the parallel velocity coordinate for the case
where density and flow are evolved independently from the pdf;
in this case, the parallel velocity coordinate is the peculiar velocity
wpa = vpa-upar
"""
function update_speed_n_u_evolution!(advect, fvec, moments, vpa, z, r, composition,
                                     collisions, ion_source_settings)
    @loop_s is begin
        @loop_r ir begin
            # update parallel acceleration to account for:
            # • parallel derivative of parallel pressure
            # • -wpar*dupar/dz
            @loop_z_vperp iz ivperp begin
                @views @. advect[is].speed[:,ivperp,iz,ir] =
                    moments.ion.dppar_dz[iz,ir,is]/fvec.density[iz,ir,is] -
                    vpa.grid*moments.ion.dupar_dz[iz,ir,is]
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
    if ion_source_settings.active
        source_density_amplitude = moments.ion.external_source_density_amplitude
        source_strength = ion_source_settings.source_strength
        r_amplitude = ion_source_settings.r_amplitude
        z_amplitude = ion_source_settings.z_amplitude
        density = fvec.density
        upar = fvec.upar
        vth = moments.ion.vth
        @loop_s_r_z is ir iz begin
            term = source_density_amplitude[iz,ir] * upar[iz,ir,is] / density[iz,ir,is]
            @loop_vperp_vpa ivperp ivpa begin
                advect[is].speed[ivpa,ivperp,iz,ir] += term
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
