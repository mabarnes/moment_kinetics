"""
"""
module velocity_moments

export integrate_over_vspace
export integrate_over_positive_vpa, integrate_over_negative_vpa
export create_moments
export update_moments!
export update_density!
export update_upar!
export update_ppar!
export update_qpar!
export reset_moments_status!

using ..type_definitions: mk_float
using ..array_allocation: allocate_shared_float, allocate_bool
using ..calculus: integral
using ..communication
using ..looping

#global tmpsum1 = 0.0
#global tmpsum2 = 0.0
#global dens_hist = zeros(17,1)
#global n_hist = 0

"""
"""
mutable struct moments
    # this is the particle density
    dens::MPISharedArray{mk_float,3}
    # flag that keeps track of if the density needs updating before use
    # Note: may not be set for all species on this process, but this process only ever
    # sets/uses the value for the same subset of species. This means dens_update does
    # not need to be a shared memory array.
    dens_updated::Vector{Bool}
    # flag that indicates if the density should be evolved via continuity equation
    evolve_density::Bool
    # flag that indicates if particle number should be conserved for each species
    # effects like ionisation or net particle flux from the domain would lead to
    # non-conservation
    particle_number_conserved::Array{Bool,1}
    # flag that indicates if exact particle conservation should be enforced
    enforce_conservation::Bool
    # this is the parallel flow
    upar::MPISharedArray{mk_float,3}
    # flag that keeps track of whether or not upar needs updating before use
    # Note: may not be set for all species on this process, but this process only ever
    # sets/uses the value for the same subset of species. This means upar_update does
    # not need to be a shared memory array.
    upar_updated::Vector{Bool}
    # flag that indicates if the parallel flow should be evolved via force balance
    evolve_upar::Bool
    # this is the parallel pressure
    ppar::MPISharedArray{mk_float,3}
    # flag that keeps track of whether or not ppar needs updating before use
    # Note: may not be set for all species on this process, but this process only ever
    # sets/uses the value for the same subset of species. This means ppar_update does
    # not need to be a shared memory array.
    ppar_updated::Vector{Bool}
    # flag that indicates if the parallel pressure should be evolved via the energy equation
    evolve_ppar::Bool
    # this is the parallel heat flux
    qpar::MPISharedArray{mk_float,3}
    # flag that keeps track of whether or not qpar needs updating before use
    # Note: may not be set for all species on this process, but this process only ever
    # sets/uses the value for the same subset of species. This means qpar_update does
    # not need to be a shared memory array.
    qpar_updated::Vector{Bool}
    # this is the thermal speed based on the parallel temperature Tpar = ppar/dens: vth = sqrt(2*Tpar/m)
    vth::MPISharedArray{mk_float,3}
    # if evolve_ppar = true, then the velocity variable is (vpa - upa)/vth, which introduces
    # a factor of vth for each power of wpa in velocity space integrals.
    # vpa_norm_fac accounts for this: it is vth if using the above definition for the parallel velocity,
    # and it is one otherwise
    vpa_norm_fac::MPISharedArray{mk_float,3}
    # flag that indicates if the drift kinetic equation should be formulated in advective form
    #advective_form::Bool
end

"""
"""
function create_moments(nz, nr, n_species, evolve_moments, ionization, z_bc)
    # allocate array used for the particle density
    density = allocate_shared_float(nz, nr, n_species)
    # allocate array of Bools that indicate if the density is updated for each species
    density_updated = allocate_bool(n_species)
    density_updated .= false
    # allocate array used for the parallel flow
    parallel_flow = allocate_shared_float(nz, nr, n_species)
    # allocate array of Bools that indicate if the parallel flow is updated for each species
    parallel_flow_updated = allocate_bool(n_species)
    parallel_flow_updated .= false
    # allocate array used for the parallel pressure
    parallel_pressure = allocate_shared_float(nz, nr, n_species)
    # allocate array of Bools that indicate if the parallel pressure is updated for each species
    parallel_pressure_updated = allocate_bool(n_species)
    parallel_pressure_updated .= false
    # allocate array used for the parallel flow
    parallel_heat_flux = allocate_shared_float(nz, nr, n_species)
    # allocate array of Bools that indicate if the parallel flow is updated for each species
    parallel_heat_flux_updated = allocate_bool(n_species)
    parallel_heat_flux_updated .= false
    # allocate array used for the thermal speed
    thermal_speed = allocate_shared_float(nz, nr, n_species)
    if evolve_moments.parallel_pressure
        vpa_norm_fac = thermal_speed
    else
        vpa_norm_fac = allocate_shared_float(nz, nr, n_species)
        @serial_region begin
            vpa_norm_fac .= 1.0
        end
    end
    # allocate array of Bools that indicate if particle number for each species should be conserved
    particle_number_conserved = allocate_bool(n_species)
    # by default, assumption is that particle number should be conserved for each species
    particle_number_conserved .= true
    # if ionization collisions are included or wall BCs are enforced,
    # then particle number is not conserved within each species
    if abs(ionization) > 0.0 || z_bc == "wall"
        particle_number_conserved .= false
    end

    # return struct containing arrays needed to update moments
    return moments(density, density_updated, evolve_moments.density, particle_number_conserved,
        evolve_moments.conservation,
        parallel_flow, parallel_flow_updated, evolve_moments.parallel_flow,
        parallel_pressure, parallel_pressure_updated, evolve_moments.parallel_pressure,
        parallel_heat_flux, parallel_heat_flux_updated, thermal_speed, vpa_norm_fac)
end

"""
calculate the updated density (dens) and parallel pressure (ppar) for all species
"""
function update_moments!(moments, ff, vpa, z, r, composition)
    begin_s_r_z_region()
    n_species = size(ff,4)
    @boundscheck n_species == size(moments.dens,3) || throw(BoundsError(moments))
    @loop_s is begin
        if moments.dens_updated[is] == false
            @views update_density_species!(moments.dens[:,:,is], ff[:,:,:,is], vpa, z, r)
            moments.dens_updated[is] = true
        end
        if moments.upar_updated[is] == false
            # Can pass moments.ppar here even though it has not been updated yet,
            # because moments.ppar is only needed if evolve_ppar=true, in which case it
            # will not be updated because it is not calculated from the distribution
            # function
            @views update_upar_species!(moments.upar[:,:,is], moments.dens[:,:,is],
                                        moments.ppar[:,:,is], ff[:,:,:,is], vpa, z, r,
                                        moments.evolve_density, moments.evolve_ppar)
            moments.upar_updated[is] = true
        end
        if moments.ppar_updated[is] == false
            @views update_ppar_species!(moments.ppar[:,:,is], moments.dens[:,:,is],
                                        moments.upar[:,:,is], ff[:,:,:,is], vpa, z, r,
                                        moments.evolve_density, moments.evolve_upar)
            moments.ppar_updated[is] = true
        end
        @loop_r_z ir iz begin
            moments.vth[iz,ir,is] = sqrt(2*moments.ppar[iz,ir,is]/moments.dens[iz,ir,is])
        end
        if moments.qpar_updated[is] == false
            @views update_qpar_species!(moments.qpar[:,is], moments.dens[:,:,is],
                                        moments.vth[:,:,is], ff[:,:,is], vpa, z, r,
                                        moments.evolve_density, moments.evolve_upar,
                                        moments.evolve_ppar)
            moments.qpar_updated[is] = true
        end
    end
    return nothing
end

"""
NB: if this function is called and if dens_updated is false, then
the incoming pdf is the un-normalized pdf that satisfies int dv pdf = density
"""
function update_density!(dens, dens_updated, pdf, vpa, z, r, composition)

    begin_s_r_z_region()

    n_species = size(pdf,4)
    @boundscheck n_species == size(dens,3) || throw(BoundsError(dens))
    @loop_s is begin
        if dens_updated[is] == false
            @views update_density_species!(dens[:,:,is], pdf[:,:,:,is], vpa, z, r)
            dens_updated[is] = true
        end
    end
end

"""
calculate the updated density (dens) for a given species;
should only be called when evolve_density = false,
in which case the vpa coordinate is vpa/c_s
"""
function update_density_species!(dens, ff, vpa, z, r)
    @boundscheck z.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck z.n == size(dens, 1) || throw(BoundsError(dens))
    @loop_r_z ir iz begin
        # When evolve_density = false, the evolved pdf is the 'true' pdf, and the vpa
        # coordinate is (dz/dt) / c_s.
        # Integrating calculates n_s / N_e = (1/√π)∫d(vpa/c_s) (√π f_s c_s / N_e)
        dens[iz,ir] = integrate_over_vspace(@view(ff[:,iz,ir]), vpa.wgts)
    end
    return nothing
end

"""
NB: if this function is called and if upar_updated is false, then
the incoming pdf is the un-normalized pdf that satisfies int dv pdf = density
"""
function update_upar!(upar, upar_updated, density, ppar, pdf, vpa, z, r, composition,
                      evolve_density, evolve_ppar)

    begin_s_r_z_region()

    n_species = size(pdf,4)
    @boundscheck n_species == size(upar,3) || throw(BoundsError(upar))
    @loop_s is begin
        if upar_updated[is] == false
            @views update_upar_species!(upar[:,:,is], density[:,:,is], ppar[:,:,is],
                                        pdf[:,:,:,is], vpa, z, r, evolve_density,
                                        evolve_ppar)
            upar_updated[is] = true
        end
    end
end

"""
calculate the updated parallel flow (upar) for a given species
"""
function update_upar_species!(upar, density, ppar, ff, vpa, z, r, evolve_density,
                              evolve_ppar)
    @boundscheck z.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck z.n == size(upar, 1) || throw(BoundsError(upar))
    if evolve_density && evolve_ppar
        # this is the case where the density and parallel pressure are evolved
        # separately from the normalized pdf, g_s = (√π f_s vth_s / n_s); the vpa
        # coordinate is (dz/dt) / vth_s.
        # Integrating calculates
        # (upar_s / vth_s) = (1/√π)∫d(vpa/vth_s) * (vpa/vth_s) * (√π f_s vth_s / n_s)
        # so convert from upar_s / vth_s to upar_s / c_s
        @loop_r_z ir iz begin
            vth = sqrt(2.0*ppar[iz,ir]/density[iz,ir])
            upar[iz,ir] = integrate_over_vspace(@view(ff[:,iz,ir]), vpa.grid, vpa.wgts) * vth
        end
    elseif evolve_density
        # corresponds to case where only the density is evolved separately from the
        # normalised pdf, given by g_s = (√π f_s c_s / n_s); the vpa coordinate is
        # (dz/dt) / c_s.
        # Integrating calculates
        # (upar_s / c_s) = (1/√π)∫d(vpa/c_s) * (vpa/c_s) * (√π f_s c_s / n_s)
        @loop_r_z ir iz begin
            upar[iz,ir] = integrate_over_vspace(@view(ff[:,iz,ir]), vpa.grid, vpa.wgts)
        end
    else
        # When evolve_density = false, the evolved pdf is the 'true' pdf,
        # and the vpa coordinate is (dz/dt) / c_s.
        # Integrating calculates
        # (n_s / N_e) * (upar_s / c_s) = (1/√π)∫d(vpa/c_s) * (vpa/c_s) * (√π f_s c_s / N_e)
        @loop_r_z ir iz begin
            upar[iz,ir] = integrate_over_vspace(@view(ff[:,iz,ir]), vpa.grid, vpa.wgts) /
                          density[iz,ir]
        end
    end
    return nothing
end

"""
NB: if this function is called and if ppar_updated is false, then
the incoming pdf is the un-normalized pdf that satisfies int dv pdf = density
"""
function update_ppar!(ppar, ppar_updated, density, upar, pdf, vpa, z, r, composition,
                      evolve_density, evolve_upar)
    @boundscheck composition.n_species == size(ppar,3) || throw(BoundsError(ppar))
    @boundscheck r.n == size(ppar,2) || throw(BoundsError(ppar))
    @boundscheck z.n == size(ppar,1) || throw(BoundsError(ppar))

    begin_s_r_z_region()

    @loop_s is begin
        if ppar_updated[is] == false
            @views update_ppar_species!(ppar[:,:,is], density[:,:,is], upar[:,:,is],
                                        pdf[:,:,:,is], vpa, z, r, evolve_density,
                                        evolve_upar)
            ppar_updated[is] = true
        end
    end
end

"""
calculate the updated energy density (or parallel pressure, ppar) for a given species;
which of these is calculated depends on the definition of the vpa coordinate
"""
function update_ppar_species!(ppar, density, upar, ff, vpa, z, r, evolve_density, evolve_upar)
    @boundscheck z.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck z.n == size(ppar, 1) || throw(BoundsError(ppar))
    if evolve_upar
        # this is the case where the parallel flow and density are evolved separately
        # from the normalized pdf, g_s = (√π f_s c_s / n_s); the vpa coordinate is
        # ((dz/dt) - upar_s) / c_s>
        # Integrating calculates (p_parallel/m_s n_s c_s^2) = (1/√π)∫d((vpa-upar_s)/c_s) (1/2)*((vpa-upar_s)/c_s)^2 * (√π f_s c_s / n_s)
        # so convert from p_s / m_s n_s c_s^2 to ppar_s = p_s / m_s N_e c_s^2
        @loop_r_z ir iz begin
            ppar[iz,ir] = integrate_over_vspace(@view(ff[:,iz,ir]), vpa.grid, 2, vpa.wgts) *
                          density[iz,ir]
        end
    elseif evolve_density
        # corresponds to case where only the density is evolved separately from the
        # normalised pdf, given by g_s = (√π f_s c_s / n_s); the vpa coordinate is
        # (dz/dt) / c_s.
        # Integrating calculates
        # (p_parallel/m_s n_s c_s^2) + (upar_s/c_s)^2 = (1/√π)∫d(vpa/c_s) (vpa/c_s)^2 * (√π f_s c_s / n_s)
        # so subtract off the mean kinetic energy and multiply by density to get the
        # internal energy density (aka pressure)
        @loop_r_z ir iz begin
            ppar[iz,ir] = (integrate_over_vspace(@view(ff[:,iz,ir]), vpa.grid, 2, vpa.wgts) -
                           upar[iz,ir]^2) * density[iz,ir]
        end
    else
        # When evolve_density = false, the evolved pdf is the 'true' pdf,
        # and the vpa coordinate is (dz/dt) / c_s.
        # Integrating calculates
        # (p_parallel/m_s N_e c_s^2) + (n_s/N_e)*(upar_s/c_s)^2 = (1/√π)∫d(vpa/c_s) (vpa/c_s)^2 * (√π f_s c_s / N_e)
        # so subtract off the mean kinetic energy density to get the internal energy
        # density (aka pressure)
        @loop_r_z ir iz begin
            ppar[iz,ir] = integrate_over_vspace(@view(ff[:,iz,ir]), vpa.grid, 2, vpa.wgts) -
                          density[iz,ir]*upar[iz,ir]^2
        end
    end
    return nothing
end

"""
NB: the incoming pdf is the normalized pdf
"""
function update_qpar!(qpar, qpar_updated, density, upar, vth, pdf, vpa, z, r,
                      composition, evolve_density, evolve_upar, evolve_ppar)
    @boundscheck composition.n_species == size(qpar,3) || throw(BoundsError(qpar))

    begin_s_r_z_region()

    @loop_s is begin
        if qpar_updated[is] == false
            @views update_qpar_species!(qpar[:,:,is], density[:,:,is], upar[:,:,is],
                                        vth[:,:,is], pdf[:,:,:,is], vpa, z, r,
                                        evolve_density, evolve_upar, evolve_ppar)
            qpar_updated[is] = true
        end
    end
end

"""
calculate the updated parallel heat flux (qpar) for a given species
"""
function update_qpar_species!(qpar, density, upar, vth, ff, vpa, z, r, evolve_density,
                              evolve_upar, evolve_ppar)
    @boundscheck z.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck z.n == size(qpar, 1) || throw(BoundsError(qpar))
    if evolve_upar && evolve_ppar
        @loop_r_z ir iz begin
            qpar[iz,ir] = integrate_over_vspace(@view(ff[:,iz,ir]), vpa.grid, 3, vpa.wgts) *
                          density[iz,ir] * vth[iz,ir]^3
        end
    elseif evolve_upar
        @loop_r_z ir iz begin
            qpar[iz,ir] = integrate_over_vspace(@view(ff[:,iz,ir]), vpa.grid, 3, vpa.wgts) *
                          density[iz,ir]
        end
    elseif evolve_ppar
        @loop_r_z ir iz begin
            @. vpa.scratch = vpa.grid - upar[iz,ir]
            qpar[iz,ir] = integrate_over_vspace(@view(ff[:,iz,ir]), vpa.scratch, 3, vpa.wgts) *
                          density[iz,ir] * vth[iz,ir]^3
        end
    elseif evolve_density
        @loop_r_z ir iz begin
            @. vpa.scratch = vpa.grid - upar[iz,ir]
            qpar[iz,ir] = integrate_over_vspace(@view(ff[:,iz,ir]), vpa.scratch, 3, vpa.wgts) *
                          density[iz,ir]
        end
    else
        @loop_r_z ir iz begin
            @. vpa.scratch = vpa.grid - upar[iz,ir]
            qpar[iz,ir] = integrate_over_vspace(@view(ff[:,iz,ir]), vpa.scratch, 3, vpa.wgts)
        end
    end
    return nothing
end

"""
computes the integral over vpa of the integrand, using the input vpa_wgts
"""
function integrate_over_vspace(args...)
    return integral(args...)/sqrt(pi)
end

"""
computes the integral over vpa >= 0 of the integrand, using the input vpa_wgts
this could be made more efficient for the case that dz/dt = vpa is time-independent,
but it has been left general for the cases where, e.g., dz/dt = wpa*vth + upar
varies in time
"""
function integrate_over_positive_vpa(integrand, dzdt, vpa_wgts, wgts_mod)
    # define the nvpa variable for convenience
    nvpa = length(dzdt)
    # define an approximation to zero that allows for finite-precision arithmetic
    zero = -1.0e-15
    # if dzdt at the maximum vpa index is negative, then dzdt < 0 everywhere
    # the integral over positive dzdt is thus zero, as we assume the distribution
    # function is zero beyond the simulated vpa domain
    if dzdt[nvpa] < zero
        vpa_integral = 0.0
    else
        # do bounds checks on arrays that will be used in the below loop
        @boundscheck nvpa == length(integrand) || throw(BoundsError(integrand))
        @boundscheck nvpa == length(dzdt) || throw(BoundsError(dzdt))
        @boundscheck nvpa == length(vpa_wgts) || throw(BoundsError(vpa_wgts))
        @boundscheck nvpa == length(wgts_mod) || throw(BoundsError(wgts_mod))
        # initialise the integration weights, wgts_mod, to be the input vpa_wgts
        # this will only change at the dzdt = 0 point, if it exists on the grid
        @. wgts_mod = vpa_wgts
        # ivpa_zero will be the minimum index for which dzdt[ivpa_zero] >= 0
        ivpa_zero = 0
        @inbounds for ivpa ∈ 1:nvpa
            if dzdt[ivpa] >= zero
                ivpa_zero = ivpa
                # if dzdt = 0, need to divide its associated integration
                # weight by a factor of 2 to avoid double-counting
                if abs(dzdt[ivpa]) < abs(zero)
                    wgts_mod[ivpa] /= 2.0
                end
                break
            end
        end
        @views vpa_integral = integral(integrand[ivpa_zero:end], wgts_mod[ivpa_zero:end])/sqrt(pi)
    end
    return vpa_integral
end

"""
computes the integral over vpa <= 0 of the integrand, using the input vpa_wgts
this could be made more efficient for the case that dz/dt = vpa is time-independent,
but it has been left general for the cases where, e.g., dz/dt = wpa*vth + upar
varies in time
"""
function integrate_over_negative_vpa(integrand, dzdt, vpa_wgts, wgts_mod)
    # define the nvpa variable for convenience
    nvpa = length(integrand)
    # define an approximation to zero that allows for finite-precision arithmetic
    zero = 1.0e-15
    # if dzdt at the mimimum vpa index is positive, then dzdt > 0 everywhere
    # the integral over negative dzdt is thus zero, as we assume the distribution
    # function is zero beyond the simulated vpa domain
    if dzdt[1] > zero
        vpa_integral = 0.0
    else
        # do bounds checks on arrays that will be used in the below loop
        @boundscheck nvpa == length(integrand) || throw(BoundsError(integrand))
        @boundscheck nvpa == length(dzdt) || throw(BoundsError(dzdt))
        @boundscheck nvpa == length(vpa_wgts) || throw(BoundsError(vpa_wgts))
        @boundscheck nvpa == length(wgts_mod) || throw(BoundsError(wgts_mod))
        # initialise the integration weights, wgts_mod, to be the input vpa_wgts
        # this will only change at the dzdt = 0 point, if it exists on the grid
        @. wgts_mod = vpa_wgts
        # ivpa_zero will be the maximum index for which dzdt[ivpa_zero] <= 0
        ivpa_zero = 0
        @inbounds for ivpa ∈ nvpa:-1:1
            if dzdt[ivpa] <= zero
                ivpa_zero = ivpa
                # if dzdt = 0, need to divide its associated integration
                # weight by a factor of 2 to avoid double-counting
                if abs(dzdt[ivpa]) < zero
                    wgts_mod[ivpa] /= 2.0
                end
                break
            end
        end
        @views vpa_integral = integral(integrand[1:ivpa_zero], wgts_mod[1:ivpa_zero])/sqrt(pi)
    end
    return vpa_integral
end

"""
"""
function reset_moments_status!(moments, composition, z)
    if moments.evolve_density == false
        moments.dens_updated .= false
    end
    if moments.evolve_upar == false
        moments.upar_updated .= false
    end
    if moments.evolve_ppar == false
        moments.ppar_updated .= false
    end
    moments.qpar_updated .= false
end

end
