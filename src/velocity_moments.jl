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
export enforce_moment_constraints!

using ..type_definitions: mk_float
using ..array_allocation: allocate_shared_float, allocate_bool
using ..calculus: integral
using ..communication
using ..communication: _block_synchronize
using ..looping

#global tmpsum1 = 0.0
#global tmpsum2 = 0.0
#global dens_hist = zeros(17,1)
#global n_hist = 0

mutable struct moments
    # this is the particle density
    dens::MPISharedArray{mk_float,3}
    # MRH dens::MPISharedArray{mk_float,2}
    # flag that keeps track of if the density needs updating before use
    # Note: may not be set for all species on this process, but this process only ever
    # sets/uses the value for the same subset of species. This means dens_update does
    # not need to be a shared memory array.
    dens_updated::Vector{Bool}
    # flag that indicates if the density should be evolved via continuity equation
    evolve_density::Bool
    # flag that indicates if exact particle conservation should be enforced
    enforce_conservation::Bool
    # this is the parallel flow
    upar::MPISharedArray{mk_float,3}
    # MRH upar::MPISharedArray{mk_float,2}
    # flag that keeps track of whether or not upar needs updating before use
    # Note: may not be set for all species on this process, but this process only ever
    # sets/uses the value for the same subset of species. This means upar_update does
    # not need to be a shared memory array.
    upar_updated::Vector{Bool}
    # flag that indicates if the parallel flow should be evolved via force balance
    evolve_upar::Bool
    # this is the parallel pressure
    ppar::MPISharedArray{mk_float,3}
    # MRH ppar::MPISharedArray{mk_float,2}
    # flag that keeps track of whether or not ppar needs updating before use
    # Note: may not be set for all species on this process, but this process only ever
    # sets/uses the value for the same subset of species. This means ppar_update does
    # not need to be a shared memory array.
    ppar_updated::Vector{Bool}
    # flag that indicates if the parallel pressure should be evolved via the energy equation
    evolve_ppar::Bool
    # this is the parallel heat flux
    qpar::MPISharedArray{mk_float,3}
    # MRH qpar::MPISharedArray{mk_float,2}
    # flag that keeps track of whether or not qpar needs updating before use
    # Note: may not be set for all species on this process, but this process only ever
    # sets/uses the value for the same subset of species. This means qpar_update does
    # not need to be a shared memory array.
    qpar_updated::Vector{Bool}
    # this is the thermal speed based on the parallel temperature Tpar = ppar/dens: vth = sqrt(2*Tpar/m)
    vth::MPISharedArray{mk_float,3}
    # MRH vth::MPISharedArray{mk_float,2}
    # if evolve_ppar = true, then the velocity variable is (vpa - upa)/vth, which introduces
    # a factor of vth for each power of wpa in velocity space integrals.
    # vpa_norm_fac accounts for this: it is vth if using the above definition for the parallel velocity,
    # and it is one otherwise
    # MRH vpa_norm_fac::MPISharedArray{mk_float,2}
    vpa_norm_fac::MPISharedArray{mk_float,3}
    # flag that indicates if the drift kinetic equation should be formulated in advective form
    #advective_form::Bool
end
function create_moments(nz, nr, n_species, evolve_moments)
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
    # return struct containing arrays needed to update moments
    return moments(density, density_updated, evolve_moments.density, evolve_moments.conservation,
        parallel_flow, parallel_flow_updated, evolve_moments.parallel_flow,
        parallel_pressure, parallel_pressure_updated, evolve_moments.parallel_pressure,
        parallel_heat_flux, parallel_heat_flux_updated, thermal_speed, vpa_norm_fac)
end
# calculate the updated density (dens) and parallel pressure (ppar) for all species
function update_moments!(moments, ff, vpa, nz, composition)
    n_species = size(ff,3)
    @boundscheck n_species == size(moments.dens,2) || throw(BoundsError(moments))
    @s_z_loop_s is begin
        if moments.dens_updated[is] == false
            @views update_density_species!(moments.dens[:,is], ff[:,:,is], vpa, z)
            moments.dens_updated[is] = true
        end
        if moments.upar_updated[is] == false
            @views update_upar_species!(moments.upar[:,is], ff[:,:,is], vpa, z)
            moments.upar_updated[is] = true
        end
        if moments.ppar_updated[is] == false
            @views update_ppar_species!(moments.ppar[:,is], ff[:,:,is], vpa, z)
            moments.ppar_updated[is] = true
        end
        @. moments.vth = sqrt(2*moments.ppar/moments.dens)
        if moments.qpar_updated[is] == false
            @views update_qpar_species!(moments.qpar[:,is], ff[:,:,is], vpa, z, moments.vpa_norm_fac[:,is])
            moments.qpar_updated[is] = true
        end
    end
    return nothing
end
# NB: if this function is called and if dens_updated is false, then
# the incoming pdf is the un-normalized pdf that satisfies int dv pdf = density
function update_density!(dens, dens_updated, pdf, vpa, z, composition)
    n_species = size(pdf,3)
    @boundscheck n_species == size(dens,2) || throw(BoundsError(dens))
    @s_z_loop_s is begin
        if dens_updated[is] == false
            @views update_density_species!(dens[:,is], pdf[:,:,is], vpa, z)
            dens_updated[is] = true
        end
    end
end
# calculate the updated density (dens) for a given species
function update_density_species!(dens, ff, vpa, z)
    @boundscheck z.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck z.n == length(dens) || throw(BoundsError(dens))
    @s_z_loop_z iz begin
        dens[iz] = integrate_over_vspace(@view(ff[:,iz]), vpa.wgts)
    end
    return nothing
end
# NB: if this function is called and if upar_updated is false, then
# the incoming pdf is the un-normalized pdf that satisfies int dv pdf = density
function update_upar!(upar, upar_updated, pdf, vpa, z, composition)
    n_species = size(pdf,3)
    @boundscheck n_species == size(upar,2) || throw(BoundsError(upar))
    @s_z_loop_s is begin
        if upar_updated[is] == false
            @views update_upar_species!(upar[:,is], pdf[:,:,is], vpa, z)
            upar_updated[is] = true
        end
    end
end
# calculate the updated parallel flow (upar) for a given species
function update_upar_species!(upar, ff, vpa, z)
    @boundscheck z.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck z.n == length(upar) || throw(BoundsError(upar))
    @s_z_loop_z iz begin
        upar[iz] = integrate_over_vspace(@view(ff[:,iz]), vpa.grid, vpa.wgts)
    end
    return nothing
end
# NB: if this function is called and if ppar_updated is false, then
# the incoming pdf is the un-normalized pdf that satisfies int dv pdf = density
function update_ppar!(ppar, ppar_updated, pdf, vpa, z, composition)
    @boundscheck composition.n_species == size(ppar,2) || throw(BoundsError(ppar))
    @s_z_loop_s is begin
        if ppar_updated[is] == false
            @views update_ppar_species!(ppar[:,is], pdf[:,:,is], vpa, z)
            ppar_updated[is] = true
        end
    end
end
# calculate the updated parallel pressure (ppar) for a given species
function update_ppar_species!(ppar, ff, vpa, z)
    @boundscheck z.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck z.n == length(ppar) || throw(BoundsError(ppar))
    @s_z_loop_z iz begin
        ppar[iz] = integrate_over_vspace(@view(ff[:,iz]), vpa.grid, 2, vpa.wgts)
    end
    return nothing
end
# NB: if this function is called and if ppar_updated is false, then
# the incoming pdf is the un-normalized pdf that satisfies int dv pdf = density

function update_qpar!(qpar, qpar_updated, pdf, vpa, z, r, composition, vpanorm)
    @boundscheck composition.n_species == size(qpar,3) || throw(BoundsError(qpar))
    @s_z_loop_s is begin
        if qpar_updated[is] == false
            @views update_qpar_species!(qpar[:,:,is], pdf[:,:,:,is], vpa, z, r, vpanorm[:,:,is])
            qpar_updated[is] = true
        end
    end
end
# calculate the updated parallel heat flux (qpar) for a given species
function update_qpar_species!(qpar, ff, vpa, z, r, vpanorm)
    @boundscheck z.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck z.n == size(qpar, 1) || throw(BoundsError(qpar))
    @s_z_loop_z iz begin
        qpar[iz, ir] = integrate_over_vspace(@view(ff[:,iz, ir]), vpa.grid, 3, vpa.wgts) * vpanorm[iz, ir]^4
    end
    return nothing
end
# computes the integral over vpa of the integrand, using the input vpa_wgts
function integrate_over_vspace(args...)
    return integral(args...)/sqrt(pi)
end
# computes the integral over vpa >= 0 of the integrand, using the input vpa_wgts
# this could be made more efficient for the case that dz/dt = vpa is time-independent,
# but it has been left general for the cases where, e.g., dz/dt = wpa*vth + upar
# varies in time
function integrate_over_positive_vpa(integrand, dzdt, vpa_wgts, wgts_mod)
    # define the nvpa variable for convenience
    nvpa = length(dzdt)
    # define an approximation to zero that allows for finite-precision arithmetic
    zero = -1.0e-8
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
# computes the integral over vpa <= 0 of the integrand, using the input vpa_wgts
# this could be made more efficient for the case that dz/dt = vpa is time-independent,
# but it has been left general for the cases where, e.g., dz/dt = wpa*vth + upar
# varies in time
function integrate_over_negative_vpa(integrand, dzdt, vpa_wgts, wgts_mod)
    # define the nvpa variable for convenience
    nvpa = length(integrand)
    # define an approximation to zero that allows for finite-precision arithmetic
    zero = 1.0e-8
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
function enforce_moment_constraints!(fvec_new, fvec_old, vpa, z, composition, moments)
    #global @. dens_hist += fvec_old.density
    #global n_hist += 1

    # pre-calculate avgdens_ratio so that we don't read fvec_new.density[:,is] on every
    # process in the next loop - that would be an error because different processes
    # write to fvec_new.density[:,is]
    # This loop needs to be @s_z_loop_s because it fills the (not-shared)
    # composition.scratch buffer to be used within the @s_z_loop_s below, so the values
    # of is looped over by this process need to be the same.
    @s_z_loop_s is begin
        @views @. z.scratch = fvec_old.density[:,is] - fvec_new.density[:,is]
        @views composition.scratch[is] = integral(z.scratch, z.wgts)/integral(fvec_old.density[:,is], z.wgts)
    end
    # Need to call _block_synchronize() even though loop type does not change because
    # all spatial ranks read fvec_new.density, but it will be written below.
    _block_synchronize()

    @s_z_loop_s is begin
        #tmp1 = integral(fvec_old.density[:,is], z.wgts)
        #tmp2 = integral(fvec_new.density[:,is], z.wgts)
        #@views avgdens_ratio = integral(fvec_new.density[:,is], z.wgts)/integral(fvec_old.density[:,is], z.wgts)
        avgdens_ratio = composition.scratch[is]
        @s_z_loop_z iz begin
            # Create views once to save overhead
            fnew_view = @view(fvec_new.pdf[:,iz,is])
            fold_view = @view(fvec_old.pdf[:,iz,is])

            # first calculate all of the integrals involving the updated pdf fvec_new.pdf
            density_integral = integrate_over_vspace(fnew_view, vpa.wgts)
            if moments.evolve_upar
                upar_integral = integrate_over_vspace(fnew_view, vpa.grid, vpa.wgts)
            end
            if moments.evolve_ppar
                ppar_integral = integrate_over_vspace(fnew_view, vpa.grid, 2, vpa.wgts) - 0.5*density_integral
            end
            # update the pdf to account for the density-conserving correction
            @. fnew_view += fold_view * (1.0 - density_integral)
            if moments.evolve_upar
                # next form the even part of the old distribution function that is needed
                # to ensure momentum and energy conservation
                @. vpa.scratch = fold_view
                reverse!(vpa.scratch)
                @. vpa.scratch = 0.5*(vpa.scratch + fold_view)
                # calculate the integrals involving this even pdf
                vpa2_moment = integrate_over_vspace(vpa.scratch, vpa.grid, 2, vpa.wgts)
                upar_integral /= vpa2_moment
                # update the pdf to account for the momentum-conserving correction
                @. fnew_view -= vpa.scratch * vpa.grid * upar_integral
                if moments.evolve_ppar
                    ppar_integral /= integrate_over_vspace(vpa.scratch, vpa.grid, 4, vpa.wgts) - 0.5 * vpa2_moment
                    # update the pdf to account for the energy-conserving correction
                    #@. fnew_view -= vpa.scratch * (vpa.grid^2 - 0.5) * ppar_integral
                    # Until julia-1.8 is released, prefer x*x to x^2 to avoid
                    # extra allocations when broadcasting.
                    @. fnew_view -= vpa.scratch * (vpa.grid * vpa.grid - 0.5) * ppar_integral
                end
            end
            fvec_new.density[iz,is] += fvec_old.density[iz,is] * avgdens_ratio
            # update the thermal speed, as the density has changed
            moments.vth[iz,is] = sqrt(2.0*fvec_new.ppar[iz,is]/fvec_new.density[iz,is])
        end
        #global tmpsum1 += avgdens_ratio
        #@views avgdens_ratio2 = integral(fvec_old.density[:,is] .- fvec_new.density[:,is], z.wgts)#/integral(fvec_old.density[:,is], z.wgts)
        #global tmpsum2 += avgdens_ratio2
    end
    # the pdf, density and thermal speed have been changed so the corresponding parallel heat flux must be updated
    moments.qpar_updated .= false
    # update the parallel heat flux
    # NB: no longer need fvec_old.pdf so can use for temporary storage of un-normalised pdf
    if moments.evolve_ppar
        @s_z_loop_s is begin
            @s_z_loop_z iz begin
                fvec_old.temp_z_s[iz,is] = fvec_new.density[iz,is] / moments.vth[iz,is]
            end
            @s_z_loop_z iz begin
                for ivpa ∈ 1:vpa.n
                    fvec_old.pdf[ivpa,iz,is] = fvec_new.pdf[ivpa,iz,is] * fvec_old.temp_z_s[iz,is]
                end
            end
        end
    elseif moments.evolve_density
        @s_z_loop is iz begin
            for ivpa ∈ 1:vpa.n
                fvec_old.pdf[ivpa,iz,is] = fvec_new.pdf[ivpa,iz,is] * fvec_new.density[iz,is]
            end
        end
    else
        @s_z_loop is iz begin
            for ivpa ∈ 1:vpa.n
                fvec_old.pdf[ivpa,iz,is] = fvec_new.pdf[ivpa,iz,is]
            end
        end
    end
    update_qpar!(moments.qpar, moments.qpar_updated, fvec_old.pdf, vpa, z, composition, moments.vpa_norm_fac)
end
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
