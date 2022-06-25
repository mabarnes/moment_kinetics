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
function update_moments!(moments, ff, vpa, nz, nr, composition)
    n_species = size(ff,4)
    @boundscheck n_species == size(moments.dens,3) || throw(BoundsError(moments))
    @loop_s is begin
        if moments.dens_updated[is] == false
            @views update_density_species!(moments.dens[:,:,is], ff[:,:,:,is], vpa, z, r)
            moments.dens_updated[is] = true
        end
        if moments.upar_updated[is] == false
            @views update_upar_species!(moments.upar[:,:,is], ff[:,:,:,is], vpa, z, r)
            moments.upar_updated[is] = true
        end
        if moments.ppar_updated[is] == false
            @views update_ppar_species!(moments.ppar[:,:,is], ff[:,:,:,is], vpa, z, r)
            moments.ppar_updated[is] = true
        end
        @. moments.vth = sqrt(2*moments.ppar/moments.dens)
        if moments.qpar_updated[is] == false
            @views update_qpar_species!(moments.qpar[:,is], ff[:,:,is], vpa, z, r, moments.vpa_norm_fac[:,:,is])
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
        dens[iz,ir] = integrate_over_vspace(@view(ff[:,iz,ir]), vpa.wgts)
    end
    return nothing
end

"""
NB: if this function is called and if upar_updated is false, then
the incoming pdf is the un-normalized pdf that satisfies int dv pdf = density
"""
function update_upar!(upar, upar_updated, pdf, vpa, z, r, composition)
    n_species = size(pdf,4)
    @boundscheck n_species == size(upar,3) || throw(BoundsError(upar))
    @loop_s is begin
        if upar_updated[is] == false
            @views update_upar_species!(upar[:,:,is], pdf[:,:,:,is], vpa, z, r)
            upar_updated[is] = true
        end
    end
end

"""
calculate the updated parallel flow (upar) for a given species
"""
function update_upar_species!(upar, ff, vpa, z, r)
    @boundscheck z.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck z.n == size(upar, 1) || throw(BoundsError(upar))
    @loop_r_z ir iz begin
        upar[iz,ir] = integrate_over_vspace(@view(ff[:,iz,ir]), vpa.grid, vpa.wgts)
    end
    return nothing
end

"""
NB: if this function is called and if ppar_updated is false, then
the incoming pdf is the un-normalized pdf that satisfies int dv pdf = density
"""
function update_ppar!(ppar, ppar_updated, pdf, vpa, z, r, composition)
    @boundscheck composition.n_species == size(ppar,3) || throw(BoundsError(ppar))
    @boundscheck r.n == size(ppar,2) || throw(BoundsError(ppar))
    @boundscheck z.n == size(ppar,1) || throw(BoundsError(ppar))
    @loop_s is begin
        if ppar_updated[is] == false
            @views update_ppar_species!(ppar[:,:,is], pdf[:,:,:,is], vpa, z, r)
            ppar_updated[is] = true
        end
    end
end

"""
calculate the updated energy density (or parallel pressure, ppar) for a given species;
which of these is calculated depends on the definition of the vpa coordinate
"""
function update_ppar_species!(ppar, ff, vpa, z, r)
    @boundscheck z.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck z.n == size(ppar, 1) || throw(BoundsError(ppar))
    @loop_r_z ir iz begin
        ppar[iz,ir] = integrate_over_vspace(@view(ff[:,iz,ir]), vpa.grid, 2, vpa.wgts)
    end
    return nothing
end

"""
NB: if this function is called and if ppar_updated is false, then
the incoming pdf is the un-normalized pdf that satisfies int dv pdf = density
"""
function update_qpar!(qpar, qpar_updated, pdf, vpa, z, r, composition, vpanorm)
    @boundscheck composition.n_species == size(qpar,3) || throw(BoundsError(qpar))
    @loop_s is begin
        if qpar_updated[is] == false
            @views update_qpar_species!(qpar[:,:,is], pdf[:,:,:,is], vpa, z, r, vpanorm[:,:,is])
            qpar_updated[is] = true
        end
    end
end

"""
calculate the updated parallel heat flux (qpar) for a given species
"""
function update_qpar_species!(qpar, ff, vpa, z, r, vpanorm)
    @boundscheck z.n == size(ff, 2) || throw(BoundsError(ff))
    @boundscheck z.n == size(qpar, 1) || throw(BoundsError(qpar))
    @loop_r_z ir iz begin
        qpar[iz,ir] = integrate_over_vspace(@view(ff[:,iz,ir]), vpa.grid, 3, vpa.wgts) * vpanorm[iz,ir]^4
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

"""
    enforce_moment_constraints!(fvec_new, fvec_old, vpa, z, r, composition, moments, dummy_sr)

Force moment constraints to be true for fvec_new if they were true for fvec_old. Should
always be a small correction because the time step is 'small' so fvec_new should only
violate the constraints by a small amount.
"""
function enforce_moment_constraints!(fvec_new, fvec_old, vpa, z, r, composition, moments, dummy_sr)
    # pre-calculate avgdens_ratio so that we don't read fvec_new.density[:,is] on every
    # process in the next loop - that would be an error because different processes
    # write to fvec_new.density[:,is]
    # This loop needs to be @loop_s_r because it fills the (not-shared)
    # dummy_sr buffer to be used within the @loop_s_r below, so the values
    # of is looped over by this process need to be the same.
    @loop_s_r is ir begin
        if moments.particle_number_conserved[is]
            @views @. z.scratch = fvec_old.density[:,ir,is] - fvec_new.density[:,ir,is]
            @views dummy_sr[ir,is] = integral(z.scratch, z.wgts)/integral(fvec_old.density[:,ir,is], z.wgts)
        end
    end
    # Need to call _block_synchronize() even though loop type does not change because
    # all spatial ranks read fvec_new.density, but it will be written below.
    _block_synchronize()

    @loop_s is begin
        # add a small correction to the density for each species to ensure that
        # that particle number is conserved if it should be;
        # ionisation collisions and net particle flux out of the domain due to, e.g.,
        # a wall BC break particle conservation, in which cases it should not be enforced.
        if moments.particle_number_conserved[is]
            @loop_r ir begin
                avgdens_ratio = dummy_sr[ir,is]
                @loop_z iz begin
                    # update the density with the above factor to ensure particle conservation
                    fvec_new.density[iz,ir,is] += fvec_old.density[iz,ir,is] * avgdens_ratio
                    # update the thermal speed, as the density has changed
                    moments.vth[iz,ir,is] = sqrt(2.0*fvec_new.ppar[iz,ir,is]/fvec_new.density[iz,ir,is])
                end
            end
        end
        @loop_r ir begin
            @loop_z iz begin
                # Create views once to save overhead
                fnew_view = @view(fvec_new.pdf[:,iz,ir,is])
                fold_view = @view(fvec_old.pdf[:,iz,ir,is])

                # first calculate all of the integrals involving the updated pdf fvec_new.pdf
                density_integral = integrate_over_vspace(fnew_view, vpa.wgts)
                if moments.evolve_upar
                    upar_integral = integrate_over_vspace(fnew_view, vpa.grid, vpa.wgts)
                end
                if moments.evolve_ppar
                    ppar_integral = integrate_over_vspace(fnew_view, vpa.grid, 2, vpa.wgts)
                end
                # update the pdf to account for the density-conserving correction
                @. fnew_view += fold_view * (1.0 - density_integral)
                if moments.evolve_upar
                    if !moments.evolve_ppar
                        upar_coefficient = upar_integral /
                            integrate_over_vspace(fold_view, vpa.grid, 2, vpa.wgts)
                    else
                        vpa3_moment = integrate_over_vspace(fold_view, vpa.grid, 3, vpa.wgts)
                        vpa4_moment = integrate_over_vspace(fold_view, vpa.grid, 4, vpa.wgts)
                        ppar_coefficient = (-ppar_integral + 0.5*density_integral +
                                            2.0*upar_integral*vpa3_moment) /
                                            (vpa4_moment - 0.25 - 2.0*vpa3_moment*vpa3_moment)
                        upar_coefficient = 2.0 * (upar_integral + ppar_coefficient * vpa3_moment)
                    end
                    # update the pdf to account for the momentum-conserving correction
                    @. fnew_view -= upar_coefficient * vpa.grid * fold_view
                    if moments.evolve_ppar
                        # update the pdf to account for the energy-conserving correction
                        #@. fnew_view += ppar_coefficient * (vpa.grid^2 - 0.5) * fold_view
                        # Until julia-1.8 is released, prefer x*x to x^2 to avoid
                        # extra allocations when broadcasting.
                        @. fnew_view += ppar_coefficient * (vpa.grid * vpa.grid - 0.5) * fold_view
                    end
                end
            end
        end
    end
    # the pdf, density and thermal speed have been changed so the corresponding parallel heat flux must be updated
    moments.qpar_updated .= false
    # update the parallel heat flux
    # NB: no longer need fvec_old.pdf so can use for temporary storage of un-normalised pdf
    if moments.evolve_ppar
        @loop_s is begin
            @loop_r ir begin
                @loop_z iz begin
                    fvec_old.temp_z_s[iz,ir,is] = fvec_new.density[iz,ir,is] / moments.vth[iz,ir,is]
                end
                @loop_z_vpa iz ivpa begin
                    fvec_old.pdf[ivpa,iz,ir,is] = fvec_new.pdf[ivpa,iz,ir,is] * fvec_old.temp_z_s[iz,ir,is]
                end
            end
        end
    elseif moments.evolve_density
        @loop_s_r_z_vpa is ir iz ivpa begin
            fvec_old.pdf[ivpa,iz,ir,is] = fvec_new.pdf[ivpa,iz,ir,is] * fvec_new.density[iz,ir,is]
        end
    else
        @loop_s_r_z_vpa is ir iz ivpa begin
            fvec_old.pdf[ivpa,iz,ir,is] = fvec_new.pdf[ivpa,iz,ir,is]
        end
    end
    update_qpar!(moments.qpar, moments.qpar_updated, fvec_old.pdf, vpa, z, r, composition, moments.vpa_norm_fac)
end

"""

Force the moment constraints needed for the system being evolved to be applied to `f`.
Not guaranteed to be a small correction, if `f` does not approximately obey the
constraints to start with, but can be useful at initialisation to ensure a consistent
initial state, and when applying boundary conditions.

Note this function assumes the input is given at a single spatial position.
"""
function hard_force_moment_constraints!(f, moments, vpa)
    if moments.evolve_ppar
        I0 = integrate_over_vspace(f, vpa.wgts)
        I1 = integrate_over_vspace(f, vpa.grid, vpa.wgts)
        I2 = integrate_over_vspace(f, vpa.grid, 2, vpa.wgts)
        I3 = integrate_over_vspace(f, vpa.grid, 3, vpa.wgts)
        I4 = integrate_over_vspace(f, vpa.grid, 4, vpa.wgts)
        A = ((1.0 - 0.5*I2)*(I2 - I3^2/I4) + 0.5*I3/I4*(I1-I2*I3/I4)) /
            ((I0 - I2^2)*(I2 - I3^2/I4) - (I1 - I2*I3/I4)^2)
        B = -(0.5*I3/I4 + A*(I1 - I2*I3/I4)) / (I2 - I3^2/I4)
        C = -(A*I1 + B*I2) / I3
        @. f = A*f + B*vpa.grid*f + C*vpa.grid*vpa.grid*f
    elseif moments.evolve_upar
        I0 = integrate_over_vspace(f, vpa.wgts)
        I1 = integrate_over_vspace(f, vpa.grid, vpa.wgts)
        I2 = integrate_over_vspace(f, vpa.grid, 2, vpa.wgts)
        A = 1.0 / (I0 + I1*I1/I2)
        B = -I1*A/I2
        @. f = A*f + B*vpa.grid*f
    elseif moments.evolve_density
        I0 = integrate_over_vspace(f, vpa.wgts)
        @. f = f / I0
    end

    return nothing
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
