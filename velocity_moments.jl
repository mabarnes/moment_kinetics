module velocity_moments

export integrate_over_vspace!
export update_moments!
export update_density!
export update_upar!
export update_ppar!
export reset_moments_status!
export enforce_moment_constraints!

using type_definitions: mk_float
using array_allocation: allocate_float, allocate_bool
using calculus: integral

#global tmpsum1 = 0.0
#global tmpsum2 = 0.0
#global dens_hist = zeros(17,1)
#global n_hist = 0

mutable struct moments
    # this is the particle density
    dens::Array{mk_float,2}
    # flag that keeps track of if the density needs updating before use
    dens_updated::Array{Bool,1}
    # flag that indicates if the density should be evolved via continuity equation
    evolve_density::Bool
    # flag that indicates if exact particle conservation should be enforced
    enforce_conservation::Bool
    # this is the parallel flow
    upar::Array{mk_float,2}
    # flag that keeps track of whether or not upar needs updating before use
    upar_updated::Array{Bool,1}
    # flag that indicates if the parallel flow should be evolved via force balance
    evolve_upar::Bool
    # this is the parallel pressure
    ppar::Array{mk_float,2}
    # flag that keeps track of whether or not ppar needs updating before use
    ppar_updated::Array{Bool,1}
    # flag that indicates if the parallel pressure should be evolved via the energy equation
    evolve_ppar::Bool
    # flag that indicates if the drift kinetic equation should be formulated in advective form
    #advective_form::Bool
end
# create and initialise arrays for the density and parallel pressure,
# as well as a scratch array used for intermediate calculations needed
# to later update these moments
function setup_moments(ff, vpa, nz, evolve_moments)
    n_species = size(ff,3)
    # allocate array used for the particle density
    density = allocate_float(nz, n_species)
    # allocate array of Bools that indicate if the density is updated for each species
    density_updated = allocate_bool(n_species)
    # allocate array used for the parallel flow
    parallel_flow = allocate_float(nz, n_species)
    # allocate array of Bools that indicate if the parallel flow is updated for each species
    parallel_flow_updated = allocate_bool(n_species)
    # allocate array used for the parallel pressure
    parallel_pressure = allocate_float(nz, n_species)
    # allocate array of Bools that indicate if the parallel pressure is updated for each species
    parallel_pressure_updated = allocate_bool(n_species)
    # initialise the density and parallel_pressure arrays
    for is ∈ 1:n_species
        @views update_density_species!(density[:,is], vpa.scratch, ff[:,:,is], vpa, nz)
        density_updated[is] = true
        @views update_upar_species!(parallel_flow[:,is], vpa.scratch, ff[:,:,is], vpa, nz)
        # convert from parallel particle flux to parallel flow
        @. parallel_flow[:,is] /= density[:,is]
        parallel_flow_updated[is] = true
        @views update_ppar_species!(parallel_pressure[:,is], vpa.scratch, ff[:,:,is], vpa, nz)
        parallel_pressure_updated[is] = true
    end
    # return struct containing arrays needed to update moments
    return moments(density, density_updated, evolve_moments.density, evolve_moments.conservation,
        parallel_flow, parallel_flow_updated, evolve_moments.parallel_flow,
        parallel_pressure, parallel_pressure_updated, evolve_moments.parallel_pressure)#evolve_moments.advective_form)
end
# calculate the updated density (dens) and parallel pressure (ppar) for all species
function update_moments!(moments, ff, vpa, nz)
    n_species = size(ff,3)
    @boundscheck n_species == size(moments.dens,2) || throw(BoundsError(moments))
    for is ∈ 1:n_species
        if moments.dens_updated[is] == false
            @views update_density_species!(moments.dens[:,is], vpa.scratch, ff[:,:,is], vpa, nz)
            moments.dens_updated[is] = true
        end
        if moments.upar_updated[is] == false
            @views update_upar_species!(moments.upar[:,is], vpa.scratch, ff[:,:,is], vpa, nz)
            moments.upar_updated[is] = true
        end
        if moments.ppar_updated[is] == false
            @views update_ppar_species!(moments.ppar[:,is], vpa.scratch, ff[:,:,is], vpa, nz)
            moments.ppar_updated[is] = true
        end
    end
    return nothing
end
# NB: if this function is called and if dens_updated is false, then
# the incoming pdf is the un-normalized pdf that satisfies int dv pdf = density
function update_density!(dens, dens_updated, pdf, vpa, nz)
    n_species = size(pdf,3)
    @boundscheck n_species == size(dens,2) || throw(BoundsError(dens))
    for is ∈ 1:n_species
        if dens_updated[is] == false
            @views update_density_species!(dens[:,is], vpa.scratch, pdf[:,:,is], vpa, nz)
            dens_updated[is] = true
        end
    end
end
# calculate the updated density (dens) for a given species
function update_density_species!(dens, scratch, ff, vpa, nz)
    @boundscheck nz == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck nz == length(dens) || throw(BoundsError(dens))
    @inbounds for iz ∈ 1:nz
        @views @. scratch = ff[iz,:]
        dens[iz] = integrate_over_vspace(scratch, vpa.wgts)
    end
    return nothing
end
# NB: if this function is called and if upar_updated is false, then
# the incoming pdf is the un-normalized pdf that satisfies int dv pdf = density
function update_upar!(upar, upar_updated, pdf, vpa, nz)
    n_species = size(pdf,3)
    @boundscheck n_species == size(upar,2) || throw(BoundsError(upar))
    for is ∈ 1:n_species
        if upar_updated[is] == false
            @views update_upar_species!(upar[:,is], vpa.scratch, pdf[:,:,is], vpa, nz)
            upar_updated[is] = true
        end
    end
end
# calculate the updated parallel flow (upar) for a given species
function update_upar_species!(upar, scratch, ff, vpa, nz)
    @boundscheck nz == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck nz == length(upar) || throw(BoundsError(upar))
    @inbounds for iz ∈ 1:nz
        @views @. scratch = ff[iz,:] * vpa.grid
        upar[iz] = integrate_over_vspace(scratch, vpa.wgts)
    end
    return nothing
end
# NB: if this function is called and if ppar_updated is false, then
# the incoming pdf is the un-normalized pdf that satisfies int dv pdf = density
function update_ppar!(ppar, ppar_updated, pdf, vpa, nz)
    n_species = size(pdf,3)
    @boundscheck n_species == size(ppar,2) || throw(BoundsError(ppar))
    for is ∈ 1:n_species
        if ppar_updated[is] == false
            @views update_ppar_species!(ppar[:,is], vpa.scratch, pdf[:,:,is], vpa, nz)
            ppar_updated[is] = true
        end
    end
end
# calculate the updated parallel pressure (ppar) for a given species
function update_ppar_species!(ppar, scratch, ff, vpa, nz)
    @boundscheck nz == size(ff, 1) || throw(BoundsError(ff))
    @boundscheck nz == length(ppar) || throw(BoundsError(ppar))
    @inbounds for iz ∈ 1:nz
        #@views @. scratch = ff[iz,:] * 2.0*vpa.grid^2/3.0
        @views @. scratch = ff[iz,:] * vpa.grid^2
        ppar[iz] = integrate_over_vspace(scratch, vpa.wgts)
    end
    return nothing
end
# computes the integral over vpa of the integrand, using the input vpa_wgts
function integrate_over_vspace(integrand, vpa_wgts)
    return integral(integrand, vpa_wgts)/sqrt(pi)
end
function enforce_moment_constraints!(fvec_new, fvec_old, z, vpa, moments)
    #global @. dens_hist += fvec_old.density
    #global n_hist += 1
    for is ∈ 1:size(fvec_new.density,2)
        #tmp1 = integral(fvec_old.density[:,is], z.wgts)
        #tmp2 = integral(fvec_new.density[:,is], z.wgts)
        #@views avgdens_ratio = integral(fvec_new.density[:,is], z.wgts)/integral(fvec_old.density[:,is], z.wgts)
        @views avgdens_ratio = integral(fvec_old.density[:,is] .- fvec_new.density[:,is], z.wgts)/integral(fvec_old.density[:,is], z.wgts)
        #@views avgdens_ratio = integral(fvec_old.density[:,is], z.wgts) - integral(fvec_new.density[:,is], z.wgts)
        #println("tmp1: ", tmp1, "  tmp2: ", tmp2, "  ratio: ", avgdens_ratio)
        for iz ∈ 1:size(fvec_new.density,1)
            @. vpa.scratch = fvec_new.pdf[iz,:,is]
            density_integral = integrate_over_vspace(vpa.scratch, vpa.wgts)
            @. fvec_new.pdf[iz,:,is] += fvec_old.pdf[iz,:,is] * (1.0 - density_integral)
            if moments.evolve_upar
                @. vpa.scratch *= vpa.grid
                upar_integral = integrate_over_vspace(vpa.scratch, vpa.wgts)
                @. vpa.scratch = fvec_old.pdf[iz,:,is] * vpa.grid^2
                upar_integral /= integrate_over_vspace(vpa.scratch, vpa.wgts)
                @. fvec_new.pdf[iz,:,is] -= vpa.grid*fvec_old.pdf[iz,:,is]*upar_integral
            end
            #fvec_new.density[iz,is] += fvec_old.density[iz,is] * (1.0 - avgdens_ratio)
            fvec_new.density[iz,is] += fvec_old.density[iz,is] * avgdens_ratio
        end
        #println("old_dens: ", tmp1, "  mid_dens: ", tmp2, "  new_dens: ", integral(fvec_new.density[:,is], z.wgts), "  fac: ", 1.0-avgdens_ratio)
        #global tmpsum1 += avgdens_ratio
        #@views avgdens_ratio2 = integral(fvec_old.density[:,is] .- fvec_new.density[:,is], z.wgts)#/integral(fvec_old.density[:,is], z.wgts)
        #global tmpsum2 += avgdens_ratio2
        #println("tmpsum1: ", tmpsum1, "  tmpsum2: ", tmpsum2, "  fac1: ", avgdens_ratio, "  fac2: ", avgdens_ratio2)
    end
end
function reset_moments_status!(moments)
    if moments.evolve_density == false
        moments.dens_updated .= false
    end
    if moments.evolve_upar == false
        moments.upar_updated .= false
    end
    if moments.evolve_ppar == false
        moments.ppar_updated .= false
    end
end

end
